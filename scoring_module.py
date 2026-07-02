"""scoring_module.py

Module B: Scoring & Storage (scoring half).

This module contains the business logic that sits between the low-level
NLP primitives in ``parser_module.py`` and the persistence layer in
``database.py``. It is intentionally stateless: every function receives
all the data it needs as arguments (resume text, job description text,
weights, IDs) and returns a plain, explicit result. Nothing is cached or
mutated at module scope here, which keeps this module trivially safe to
call concurrently from multiple FastAPI request handlers or repeatedly
from Streamlit reruns.

Two responsibilities are covered:
    1. ``calculate_weighted_score`` -- pure scoring math: given resume text,
       job-description text, and an HR-chosen weight, compute the semantic
       score, the keyword score, and their weighted combination.
    2. ``process_candidate_submission`` -- the end-to-end service-layer
       workflow: parse an uploaded resume, look up the target job
       description, score the pair, and persist everything.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, IO, Optional, Union

import database
import parser_module

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Default split when the caller (e.g. a brand-new dashboard session before
# the HR user has touched the slider) does not supply an explicit weight.
DEFAULT_SEMANTIC_WEIGHT: float = 0.7


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreBreakdown:
    """An immutable record of a single resume-to-job-description scoring result.

    Attributes:
        semantic_score: Cosine similarity between sentence-embedding
            vectors of the two texts, in the range ``[0.0, 1.0]``.
        keyword_score: Cosine similarity between TF-IDF vectors of the two
            texts, in the range ``[0.0, 1.0]``.
        semantic_weight: The weight applied to ``semantic_score`` when
            computing ``final_score``, in the range ``[0.0, 1.0]``.
        keyword_weight: The weight applied to ``keyword_score`` when
            computing ``final_score``. Always equals ``1.0 - semantic_weight``
            so the two weights sum to exactly 1.0.
        final_score: The weighted combination of the two method scores, in
            the range ``[0.0, 1.0]``.
    """

    semantic_score: float
    keyword_score: float
    semantic_weight: float
    keyword_weight: float
    final_score: float

    def to_dict(self) -> Dict[str, float]:
        """Convert this breakdown into a plain dictionary.

        Returns:
            Dict[str, float]: A dictionary with the same field names and
            values as this dataclass, suitable for JSON serialization in a
            FastAPI response or direct display in Streamlit.
        """
        return asdict(self)


@dataclass(frozen=True)
class SubmissionResult:
    """The outcome of processing one candidate resume submission.

    Using an explicit result object (rather than raising exceptions for
    expected failure modes like "empty PDF" or "job description not
    found") lets calling code -- particularly the Streamlit dashboard --
    display a friendly, specific message to the HR user without needing a
    try/except around every call.

    Attributes:
        success: ``True`` if the resume was parsed, scored, and persisted
            successfully. ``False`` if any step failed.
        error_message: A human-readable explanation of what went wrong.
            Always ``None`` when ``success`` is ``True``.
        resume_id: The primary key of the persisted resume row, or
            ``None`` if persistence did not occur.
        score_id: The primary key of the persisted score row, or ``None``
            if persistence did not occur.
        filename: The original filename of the submitted resume.
        breakdown: The computed :class:`ScoreBreakdown`, or ``None`` if
            scoring did not occur.
    """

    success: bool
    error_message: Optional[str]
    resume_id: Optional[int]
    score_id: Optional[int]
    filename: str
    breakdown: Optional[ScoreBreakdown]

    def to_dict(self) -> Dict[str, Any]:
        """Convert this result into a plain, JSON-serializable dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation with the nested
            ``breakdown`` (if present) also flattened into a plain dict.
        """
        result = asdict(self)
        result["breakdown"] = self.breakdown.to_dict() if self.breakdown else None
        return result


# -----------------------------------------------------------------------------
# Pure scoring logic
# -----------------------------------------------------------------------------


def _clamp_weight(weight: float) -> float:
    """Clamp a weight value into the valid ``[0.0, 1.0]`` range.

    HR-controlled sliders should never produce an out-of-range value, but
    this guard protects against programming errors upstream (e.g. a
    slider accidentally configured with a 0-100 scale instead of 0-1)
    without ever crashing the scoring pipeline.

    Args:
        weight: The raw weight value to validate.

    Returns:
        float: ``weight`` clamped to ``[0.0, 1.0]``. Falls back to
        ``DEFAULT_SEMANTIC_WEIGHT`` if ``weight`` is not a real number at
        all (e.g. ``None``, ``NaN``, or a non-numeric type).
    """
    try:
        numeric_weight = float(weight)
    except (TypeError, ValueError):
        logger.warning(
            "Received a non-numeric semantic_weight (%r); falling back to default %.2f.",
            weight,
            DEFAULT_SEMANTIC_WEIGHT,
        )
        return DEFAULT_SEMANTIC_WEIGHT

    if numeric_weight != numeric_weight:  # NaN check without importing math
        logger.warning(
            "Received NaN semantic_weight; falling back to default %.2f.",
            DEFAULT_SEMANTIC_WEIGHT,
        )
        return DEFAULT_SEMANTIC_WEIGHT

    if numeric_weight < 0.0:
        logger.warning("semantic_weight %.4f below 0.0, clamping to 0.0.", numeric_weight)
        return 0.0
    if numeric_weight > 1.0:
        logger.warning("semantic_weight %.4f above 1.0, clamping to 1.0.", numeric_weight)
        return 1.0
    return numeric_weight


def calculate_weighted_score(
    resume_text: str,
    jd_text: str,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> ScoreBreakdown:
    """Compute a resume's weighted alignment score against a job description.

    This function combines the two independent similarity methods defined
    in ``parser_module.py`` -- semantic (sentence-embedding) similarity and
    keyword (TF-IDF) similarity -- into a single, HR-tunable final score:

        final_score = (semantic_score * semantic_weight)
                     + (keyword_score * keyword_weight)

    where ``keyword_weight`` is always derived as ``1.0 - semantic_weight``
    so the two weights always sum to 1.0 and the final score stays within
    the same ``[0.0, 1.0]`` range as its two inputs.

    Args:
        resume_text: The candidate's resume text. Either raw or
            pre-cleaned text is acceptable; both underlying similarity
            functions degrade gracefully on messier text.
        jd_text: The job description text to compare the resume against.
        semantic_weight: The HR-chosen weight (0.0 to 1.0) to give to the
            semantic similarity method. For example, ``0.7`` means "70%
            semantic, 30% keyword". Values outside ``[0.0, 1.0]`` are
            clamped rather than rejected. Defaults to
            :data:`DEFAULT_SEMANTIC_WEIGHT`.

    Returns:
        ScoreBreakdown: An immutable record containing both individual
        method scores, both weights actually used, and the final combined
        score. If either input text is empty or invalid, the corresponding
        method score(s) will simply be ``0.0`` (per the safe-default
        behavior of the underlying ``parser_module`` functions) rather than
        raising an exception.
    """
    validated_semantic_weight = _clamp_weight(semantic_weight)
    keyword_weight = 1.0 - validated_semantic_weight

    semantic_score = parser_module.compute_semantic_similarity(resume_text, jd_text)
    keyword_score = parser_module.compute_keyword_similarity(resume_text, jd_text)

    raw_final_score = (semantic_score * validated_semantic_weight) + (
        keyword_score * keyword_weight
    )
    # Guard against floating-point drift pushing the result marginally
    # outside [0.0, 1.0] (e.g. 1.0000000000000002) so downstream consumers
    # (progress bars, percentage displays) never receive an invalid value.
    final_score = max(0.0, min(1.0, raw_final_score))

    return ScoreBreakdown(
        semantic_score=semantic_score,
        keyword_score=keyword_score,
        semantic_weight=validated_semantic_weight,
        keyword_weight=keyword_weight,
        final_score=final_score,
    )


# -----------------------------------------------------------------------------
# Service layer
# -----------------------------------------------------------------------------


def process_candidate_submission(
    uploaded_file: Union[str, bytes, bytearray, IO[bytes]],
    filename: str,
    job_description_id: int,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    candidate_name: Optional[str] = None,
) -> SubmissionResult:
    """Run the full pipeline for one uploaded resume: parse, score, persist.

    This is the single entry point that both the FastAPI backend and the
    Streamlit dashboard should call when a new resume is submitted for
    evaluation. It performs, in order:

        1. Look up the target job description in the database.
        2. Extract raw text from the uploaded PDF via
           :func:`parser_module.extract_text_from_pdf`.
        3. Clean both the resume text and the job description text via
           :func:`parser_module.clean_text`.
        4. Persist the resume (raw + cleaned text) via
           :func:`database.insert_resume`.
        5. Compute the weighted score via :func:`calculate_weighted_score`.
        6. Persist the score breakdown via :func:`database.insert_score`.

    Every failure mode (missing job description, unreadable PDF, empty
    extracted text, database errors) is caught and reported through the
    returned :class:`SubmissionResult` rather than raised, so a single bad
    upload can never take down the FastAPI server or crash a Streamlit
    session.

    Args:
        uploaded_file: The resume file to process, in any form accepted by
            :func:`parser_module.extract_text_from_pdf` (a filesystem
            path, raw bytes, or a file-like object with ``.read()``).
        filename: The original filename of the uploaded resume, stored
            alongside the parsed text for display in the HR leaderboard.
        job_description_id: The primary key of the job description (as
            previously created via ``database.insert_job_description``)
            to score this resume against.
        semantic_weight: The HR-chosen weight (0.0 to 1.0) to give to the
            semantic similarity method, forwarded to
            :func:`calculate_weighted_score`. Defaults to
            :data:`DEFAULT_SEMANTIC_WEIGHT`.
        candidate_name: An optional human-readable candidate name to store
            alongside the resume. Defaults to ``None``, in which case
            ``filename`` should be used for display purposes.

    Returns:
        SubmissionResult: A structured outcome describing whether the
        submission succeeded, and if so, the generated database IDs and
        the full score breakdown. If it failed, ``success`` is ``False``
        and ``error_message`` explains why.
    """
    # --- Step 1: Resolve the job description -------------------------------
    try:
        job_description = database.get_job_description(job_description_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Database error while fetching job description %s: %s",
            job_description_id,
            exc,
        )
        return SubmissionResult(
            success=False,
            error_message=(
                "A database error occurred while looking up the job description. "
                "Please try again."
            ),
            resume_id=None,
            score_id=None,
            filename=filename,
            breakdown=None,
        )

    if job_description is None:
        logger.warning("Job description with id=%s was not found.", job_description_id)
        return SubmissionResult(
            success=False,
            error_message=(
                f"No job description exists with id={job_description_id}. "
                "Please select or create a valid job description first."
            ),
            resume_id=None,
            score_id=None,
            filename=filename,
            breakdown=None,
        )

    # --- Step 2: Extract text from the uploaded resume ----------------------
    raw_resume_text = parser_module.extract_text_from_pdf(uploaded_file)
    if not raw_resume_text or not raw_resume_text.strip():
        logger.warning("No extractable text found in uploaded resume '%s'.", filename)
        return SubmissionResult(
            success=False,
            error_message=(
                f"Could not extract any text from '{filename}'. The file may be "
                "corrupted, empty, image-only (scanned without OCR), or password "
                "protected."
            ),
            resume_id=None,
            score_id=None,
            filename=filename,
            breakdown=None,
        )

    # --- Step 3: Clean both texts --------------------------------------------
    cleaned_resume_text = parser_module.clean_text(raw_resume_text)
    # The job description's cleaned_text was already computed and stored at
    # creation time, so we reuse it here instead of re-cleaning on every
    # single resume submission.
    cleaned_jd_text = job_description.get("cleaned_text", "")

    if not cleaned_resume_text:
        logger.warning(
            "Resume '%s' produced no usable text after cleaning.", filename
        )
        return SubmissionResult(
            success=False,
            error_message=(
                f"'{filename}' contained no usable text after cleaning "
                "(it may consist entirely of unsupported characters or symbols)."
            ),
            resume_id=None,
            score_id=None,
            filename=filename,
            breakdown=None,
        )

    # --- Step 4: Persist the resume ------------------------------------------
    try:
        resume_id = database.insert_resume(
            filename=filename,
            raw_text=raw_resume_text,
            cleaned_text=cleaned_resume_text,
            candidate_name=candidate_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist resume '%s': %s", filename, exc)
        return SubmissionResult(
            success=False,
            error_message="A database error occurred while saving the resume.",
            resume_id=None,
            score_id=None,
            filename=filename,
            breakdown=None,
        )

    # --- Step 5: Compute the weighted score -----------------------------------
    breakdown = calculate_weighted_score(
        resume_text=cleaned_resume_text,
        jd_text=cleaned_jd_text,
        semantic_weight=semantic_weight,
    )

    # --- Step 6: Persist the score ---------------------------------------------
    try:
        score_id = database.insert_score(
            resume_id=resume_id,
            job_description_id=job_description_id,
            semantic_score=breakdown.semantic_score,
            keyword_score=breakdown.keyword_score,
            semantic_weight=breakdown.semantic_weight,
            keyword_weight=breakdown.keyword_weight,
            final_score=breakdown.final_score,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to persist score for resume_id=%s, job_description_id=%s: %s",
            resume_id,
            job_description_id,
            exc,
        )
        # The resume itself was already saved successfully; we report a
        # partial failure so the caller/HR user understands the resume is
        # on file even though scoring could not be recorded this time.
        return SubmissionResult(
            success=False,
            error_message=(
                "The resume was saved, but a database error occurred while "
                "saving its score. You can retry scoring this candidate."
            ),
            resume_id=resume_id,
            score_id=None,
            filename=filename,
            breakdown=breakdown,
        )

    return SubmissionResult(
        success=True,
        error_message=None,
        resume_id=resume_id,
        score_id=score_id,
        filename=filename,
        breakdown=breakdown,
    )


def rescore_leaderboard(
    job_description_id: int,
    semantic_weight: float,
) -> Dict[int, ScoreBreakdown]:
    """Recompute weighted scores for every resume already scored against a job.

    This supports the dashboard's "adjustable weighting" requirement: when
    HR drags the semantic/keyword weight slider, the *underlying* semantic
    and keyword similarity numbers for already-submitted resumes do not
    need to be recomputed (they are deterministic functions of the text
    alone), only their weighted combination does. This function reuses the
    existing stored method scores and applies the new weight, which is far
    cheaper than re-running embeddings/TF-IDF for every candidate on every
    slider movement.

    Args:
        job_description_id: The job description whose leaderboard should
            be recalculated.
        semantic_weight: The new HR-chosen semantic weight (0.0 to 1.0) to
            apply to every existing score record for this job description.

    Returns:
        Dict[int, ScoreBreakdown]: A mapping from ``resume_id`` to its
        freshly recomputed :class:`ScoreBreakdown` under the new weight.
        Returns an empty dictionary if no resumes have been scored against
        this job description yet, or if the job description does not
        exist.
    """
    validated_semantic_weight = _clamp_weight(semantic_weight)
    keyword_weight = 1.0 - validated_semantic_weight

    try:
        existing_rows = database.get_leaderboard(job_description_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to fetch existing leaderboard for job_description_id=%s: %s",
            job_description_id,
            exc,
        )
        return {}

    recomputed: Dict[int, ScoreBreakdown] = {}
    for row in existing_rows:
        semantic_score = float(row["semantic_score"])
        keyword_score = float(row["keyword_score"])
        raw_final_score = (semantic_score * validated_semantic_weight) + (
            keyword_score * keyword_weight
        )
        final_score = max(0.0, min(1.0, raw_final_score))

        recomputed[int(row["resume_id"])] = ScoreBreakdown(
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            semantic_weight=validated_semantic_weight,
            keyword_weight=keyword_weight,
            final_score=final_score,
        )

    return recomputed