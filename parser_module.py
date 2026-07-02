"""parser_module.py

Module A: Text Parsing & Word Embedding.

This module is responsible for turning raw, messy input (PDF bytes or file
paths) into clean text, and for turning clean text into the two numeric
representations the rest of the ATS pipeline needs:

    1. Dense semantic embeddings, produced by a pre-trained
       ``sentence-transformers`` model (``all-MiniLM-L6-v2``), which capture
       *meaning* rather than exact word overlap.
    2. Sparse TF-IDF vectors, produced by scikit-learn's
       ``TfidfVectorizer``, which capture *lexical/keyword* overlap.

Both similarity measures are computed using scikit-learn's
``cosine_similarity`` (never implemented from scratch), per the project's
"don't reinvent the wheel" constraint.

This module deliberately contains no database or Streamlit imports so it
stays independently testable and reusable from either the FastAPI backend
or the Streamlit dashboard.
"""

import re
import logging
import threading
from typing import Any, IO, List, Optional, Union

import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# A small, fast, general-purpose sentence embedding model. It produces
# 384-dimensional vectors and offers a strong accuracy/speed trade-off,
# which matters because the evaluator's machine may not have a GPU.
DEFAULT_EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# Known output dimensionality of all-MiniLM-L6-v2. Used as a safe fallback
# vector size when we must return a zero-vector for empty/invalid text
# without having to load the model just to ask it for its own dimension.
DEFAULT_EMBEDDING_DIMENSION: int = 384

# Characters that are meaningful inside technical resumes/job descriptions
# (e.g. "C++", "C#", "Node.js", "CI/CD") and must therefore survive the
# cleaning pipeline instead of being stripped out as "special characters".
_ALLOWED_SPECIAL_CHARACTERS: str = r"\.\+\#\-\/"


# -----------------------------------------------------------------------------
# PDF Extraction
# -----------------------------------------------------------------------------


def extract_text_from_pdf(pdf_source: Union[str, bytes, bytearray, IO[bytes]]) -> str:
    """Extract raw text content from a PDF file.

    Accepts several common input shapes so this function works uniformly
    whether it is called from a FastAPI upload endpoint (which typically
    hands over raw ``bytes``), a Streamlit ``UploadedFile`` (which behaves
    like a file object with a ``.read()`` method), or a plain filesystem
    path (a ``str``).

    Args:
        pdf_source: One of:
            - ``str``: A filesystem path to a PDF file on disk.
            - ``bytes`` / ``bytearray``: The raw binary content of a PDF.
            - A file-like object exposing ``.read()`` that returns bytes
              (e.g. Streamlit's ``UploadedFile``, or a Python ``BytesIO``).

    Returns:
        str: The concatenated text of every page in the PDF, separated by
        newlines. Returns an empty string (never raises) if the PDF is
        missing, corrupted, encrypted without a supplied password, or
        otherwise unreadable, so that calling services (API endpoints,
        the dashboard) never crash on a single bad upload.
    """
    document: Optional["fitz.Document"] = None
    try:
        if isinstance(pdf_source, (bytes, bytearray)):
            document = fitz.open(stream=bytes(pdf_source), filetype="pdf")
        elif hasattr(pdf_source, "read"):
            # File-like object (Streamlit UploadedFile, BytesIO, open(..., "rb")).
            binary_content: bytes = pdf_source.read()
            if not binary_content:
                logger.warning("PDF source file-like object was empty.")
                return ""
            document = fitz.open(stream=binary_content, filetype="pdf")
        elif isinstance(pdf_source, str):
            document = fitz.open(pdf_source)
        else:
            logger.error(
                "Unsupported pdf_source type for extraction: %s", type(pdf_source)
            )
            return ""

        if document.is_encrypted:
            # Attempt a blank-password unlock, which succeeds for PDFs that
            # are encrypted but do not actually require a password to open
            # (a common export artifact). If it fails, we bail out safely.
            unlocked = document.authenticate("")
            if not unlocked:
                logger.warning("PDF is encrypted and could not be unlocked.")
                return ""

        page_texts: List[str] = []
        for page in document:
            # "text" mode gives plain reading-order text, which is more
            # robust for resumes (often multi-column) than raw dict/HTML
            # extraction modes that would need extra parsing downstream.
            page_texts.append(page.get_text("text"))

        return "\n".join(page_texts)

    except Exception as exc:  # noqa: BLE001 - we must never crash the caller
        logger.error("Failed to extract text from PDF: %s", exc)
        return ""
    finally:
        if document is not None:
            try:
                document.close()
            except Exception as close_exc:  # noqa: BLE001
                logger.warning("Failed to close PDF document cleanly: %s", close_exc)


# -----------------------------------------------------------------------------
# Text Cleaning
# -----------------------------------------------------------------------------


def clean_text(raw_text: str) -> str:
    """Normalize raw extracted text into a form suitable for NLP scoring.

    The cleaning pipeline performs, in order:
        1. Lowercasing (so "Python" and "python" are treated identically).
        2. Removal of characters that are not alphanumeric, whitespace, or
           one of a small allow-list of technically meaningful symbols
           (``. + # - /``), which preserves tokens like "c++", "c#",
           "node.js", and "ci/cd" instead of mangling them.
        3. Collapsing all runs of whitespace (including newlines and tabs
           introduced by PDF text extraction) into single spaces.
        4. Stripping leading/trailing whitespace.

    Args:
        raw_text: The unprocessed text, typically fresh output from
            ``extract_text_from_pdf`` or a pasted job description.

    Returns:
        str: The cleaned, lowercased text. Returns an empty string if the
        input is ``None``, not a string, or empty/whitespace-only, so
        downstream vectorizers never receive an invalid type.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    try:
        text = raw_text.lower()

        # Strip out everything except letters, digits, whitespace, and the
        # small set of technically meaningful punctuation marks defined
        # above. This is safer than a hand-rolled character-by-character
        # loop and keeps the regex declarative and easy to audit.
        pattern = rf"[^a-z0-9\s{_ALLOWED_SPECIAL_CHARACTERS}]"
        text = re.sub(pattern, " ", text)

        # Collapse all whitespace runs (spaces, tabs, newlines produced by
        # multi-column PDF layouts) into a single space.
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    except Exception as exc:  # noqa: BLE001 - never crash the caller
        logger.error("Failed to clean text: %s", exc)
        return ""


# -----------------------------------------------------------------------------
# Semantic Embeddings (sentence-transformers)
# -----------------------------------------------------------------------------


class EmbeddingEngine:
    """Thread-safe, lazily-loaded wrapper around a SentenceTransformer model.

    Loading a transformer model from disk/HuggingFace Hub is relatively
    expensive (hundreds of milliseconds to a few seconds). Wrapping it in a
    class that loads once and is reused for every subsequent call avoids
    reloading the model on every single resume/job-description comparison,
    which matters a great deal once the FastAPI backend is handling many
    requests or the Streamlit dashboard reruns on every widget interaction.

    Attributes:
        model_name: The HuggingFace model identifier used for encoding.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL_NAME) -> None:
        """Initialize the engine without eagerly loading the model.

        Args:
            model_name: The HuggingFace/sentence-transformers model
                identifier to load on first use. Defaults to the lightweight
                ``all-MiniLM-L6-v2`` model, which balances speed and
                semantic quality well for CPU-only evaluator environments.
        """
        self.model_name: str = model_name
        self._model: Optional[SentenceTransformer] = None
        self._lock: threading.Lock = threading.Lock()

    def get_model(self) -> SentenceTransformer:
        """Return the underlying SentenceTransformer model, loading it if needed.

        Uses double-checked locking so concurrent callers (e.g. multiple
        FastAPI request handlers) do not each trigger a separate, redundant
        model download/load.

        Returns:
            SentenceTransformer: The loaded, ready-to-use embedding model.

        Raises:
            RuntimeError: If the model fails to load (e.g. no internet
                access on first run and no local cache available). This is
                intentionally allowed to propagate because encoding cannot
                proceed at all without a model, unlike the text-cleaning
                functions above which have safe empty-string fallbacks.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:  # re-check inside the lock
                    try:
                        logger.info("Loading embedding model '%s'...", self.model_name)
                        self._model = SentenceTransformer(self.model_name)
                        logger.info("Embedding model loaded successfully.")
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "Failed to load embedding model '%s': %s",
                            self.model_name,
                            exc,
                        )
                        raise RuntimeError(
                            f"Could not load embedding model '{self.model_name}'. "
                            "Ensure the model is available locally or that the "
                            "evaluator's machine has internet access to download "
                            "it from the HuggingFace Hub on first run."
                        ) from exc
        return self._model

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single piece of text into a dense semantic vector.

        Args:
            text: The (ideally already-cleaned) text to embed.

        Returns:
            np.ndarray: A 1-D float32 array of shape
            ``(embedding_dimension,)``. If ``text`` is empty, ``None``, or
            not a string, a zero-vector of the correct dimensionality is
            returned instead of raising, so callers can safely feed
            unpredictable user input straight into this function.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return np.zeros(DEFAULT_EMBEDDING_DIMENSION, dtype=np.float32)

        try:
            model = self.get_model()
            embedding = model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            return np.asarray(embedding, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to encode text into embedding: %s", exc)
            return np.zeros(DEFAULT_EMBEDDING_DIMENSION, dtype=np.float32)


# A single, module-level engine instance shared across the application.
# Kept private and exposed via ``get_default_embedding_engine`` so callers
# never accidentally instantiate (and therefore load) multiple copies of
# the same multi-hundred-megabyte model.
_default_engine: Optional[EmbeddingEngine] = None
_default_engine_lock: threading.Lock = threading.Lock()


def get_default_embedding_engine() -> EmbeddingEngine:
    """Return the process-wide singleton ``EmbeddingEngine`` instance.

    Returns:
        EmbeddingEngine: A shared engine configured with
        ``DEFAULT_EMBEDDING_MODEL_NAME``. The underlying transformer model
        itself is still loaded lazily on first use, not at singleton
        creation time.
    """
    global _default_engine
    if _default_engine is None:
        with _default_engine_lock:
            if _default_engine is None:
                _default_engine = EmbeddingEngine(DEFAULT_EMBEDDING_MODEL_NAME)
    return _default_engine


def compute_semantic_similarity(
    text_a: str,
    text_b: str,
    engine: Optional[EmbeddingEngine] = None,
) -> float:
    """Compute the semantic (meaning-based) similarity between two texts.

    Encodes both texts with a sentence-transformers model and compares the
    resulting vectors using scikit-learn's ``cosine_similarity``, per the
    project constraint to never hand-roll cosine similarity math.

    Args:
        text_a: The first text to compare (e.g. a candidate's resume).
        text_b: The second text to compare (e.g. a job description).
        engine: An optional pre-configured ``EmbeddingEngine`` to reuse
            (useful for batch scoring many resumes against one job
            description without repeatedly re-fetching the singleton).
            Defaults to the shared process-wide engine when omitted.

    Returns:
        float: A similarity score clamped to the range ``[0.0, 1.0]``.
        Returns ``0.0`` if either input is empty/invalid rather than
        raising, since an empty resume or job description simply has no
        meaningful similarity to compare.
    """
    if not text_a or not text_b:
        return 0.0
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return 0.0

    active_engine = engine if engine is not None else get_default_embedding_engine()

    vector_a = active_engine.encode_text(text_a).reshape(1, -1)
    vector_b = active_engine.encode_text(text_b).reshape(1, -1)

    # A text that fails to encode returns an all-zero vector, whose cosine
    # similarity with anything is undefined (0/0). We guard against that
    # explicitly rather than letting scikit-learn silently emit a NaN/0.0
    # with a runtime warning.
    if not np.any(vector_a) or not np.any(vector_b):
        return 0.0

    try:
        similarity = cosine_similarity(vector_a, vector_b)[0][0]
        return float(np.clip(similarity, 0.0, 1.0))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to compute semantic similarity: %s", exc)
        return 0.0


# -----------------------------------------------------------------------------
# Keyword / Lexical Embeddings (TF-IDF)
# -----------------------------------------------------------------------------


def compute_keyword_similarity(text_a: str, text_b: str) -> float:
    """Compute lexical (keyword-overlap) similarity between two texts.

    Builds a TF-IDF vector space from exactly the two supplied documents
    and measures their cosine similarity in that space. Fitting the
    vectorizer fresh on each pair (rather than on a large shared corpus)
    is intentional for a hackathon-scale ATS: it requires no pre-built
    vocabulary/corpus management and still produces a meaningful relative
    measure of keyword overlap between a specific resume/job-description
    pair.

    Args:
        text_a: The first text to compare (e.g. a candidate's resume).
        text_b: The second text to compare (e.g. a job description).

    Returns:
        float: A similarity score clamped to the range ``[0.0, 1.0]``.
        Returns ``0.0`` if either input is empty/invalid, or if the two
        texts share no usable vocabulary after stop-word removal (which
        would otherwise raise a ``ValueError`` from an empty TF-IDF
        vocabulary).
    """
    if not text_a or not text_b:
        return 0.0
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return 0.0
    if not text_a.strip() or not text_b.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])

        # A fitted vocabulary of size zero (e.g. both texts consisted
        # entirely of English stop words) means there is nothing left to
        # compare; treat that as "no keyword overlap" rather than letting
        # cosine_similarity operate on an empty sparse matrix.
        if tfidf_matrix.shape[1] == 0:
            return 0.0

        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(similarity, 0.0, 1.0))

    except ValueError as exc:
        # Raised by scikit-learn when, after preprocessing, the resulting
        # vocabulary is empty (e.g. only stop words or only stripped
        # punctuation remained in both documents).
        logger.warning("TF-IDF vocabulary was empty for the given texts: %s", exc)
        return 0.0
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to compute keyword similarity: %s", exc)
        return 0.0
