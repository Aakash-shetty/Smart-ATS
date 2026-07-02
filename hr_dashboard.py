"""hr_dashboard.py

Module C: Visualization & HR Interface.

This is the Streamlit front-end for the ATS. It ties together every prior
module:
    - ``database``       : job description / resume / score persistence.
    - ``parser_module``  : PDF text extraction, text cleaning, fast batch
                            embeddings, and link extraction.
    - ``scoring_module`` : the weighted scoring service layer, including
                            async batch resume processing.

Run with:
    streamlit run hr_dashboard.py

All UI-rendering code lives inside ``main()`` (invoked only when this file
is executed directly), so importing this module elsewhere -- e.g. from a
verification script -- never triggers Streamlit UI calls outside of an
actual Streamlit runtime.

Render order within ``main()`` is deliberate: the weighting slider is
rendered BEFORE the upload section and the leaderboard section, so both
of those sections can simply use the local ``semantic_weight`` variable
returned by the slider -- no session-state relay tricks, no chicken-and-egg
ordering problems, and therefore no risk of referencing an undefined
variable.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

import database
import parser_module
import scoring_module

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_SEMANTIC_WEIGHT: float = 0.7
WEIGHT_SLIDER_STEP: float = 0.05
DEFAULT_BATCH_SIZE: int = 10
MIN_BATCH_SIZE: int = 1
MAX_BATCH_SIZE: int = 20


# -----------------------------------------------------------------------------
# Async helper
# -----------------------------------------------------------------------------


def run_async(coroutine: "asyncio.coroutines.Coroutine") -> Any:
    """Run an async coroutine safely from Streamlit's synchronous script context.

    Streamlit executes each script rerun in the main thread with no active
    event loop, so a plain ``asyncio.run(coroutine)`` normally works fine.
    This helper adds a fallback for the rarer case where an event loop is
    already running in the current thread (which would otherwise raise
    ``RuntimeError: asyncio.run() cannot be called from a running event
    loop``), by spinning up and tearing down a fresh loop explicitly.

    Args:
        coroutine: The coroutine object to run to completion.

    Returns:
        Any: Whatever the coroutine returns.
    """
    try:
        return asyncio.run(coroutine)
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()


# -----------------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------------


def _initialize_session_state() -> None:
    """Ensure every key this dashboard relies on exists in ``st.session_state``.

    Streamlit reruns the entire script top-to-bottom on every user
    interaction (slider drag, button click, file upload). Session state is
    the only mechanism that survives across those reruns within a single
    browser session, so every piece of state the dashboard needs to
    remember across reruns is initialized once here, defensively, before
    any widget reads or writes it.

    Note: the semantic weight and batch size slider values do NOT need to
    be stored here -- Streamlit widgets automatically retain their own
    value across reruns as long as the widget is rendered with the same
    label/key on every run.
    """
    if "active_job_description_id" not in st.session_state:
        st.session_state["active_job_description_id"] = None
    if "last_processing_summary" not in st.session_state:
        st.session_state["last_processing_summary"] = None


# -----------------------------------------------------------------------------
# Section 1: Job Description management
# -----------------------------------------------------------------------------


def render_job_description_section() -> Optional[int]:
    """Render the job-description creation/selection panel in the sidebar.

    Gives HR two mutually exclusive ways to choose the "active" job
    description for the rest of the dashboard:
        1. Paste a brand-new job description (persisted immediately).
        2. Select a previously-created job description from a dropdown.

    Returns:
        Optional[int]: The primary key of the currently active job
        description, or ``None`` if HR has neither created nor selected
        one yet.
    """
    st.sidebar.header("1. Job Description")

    jd_mode = st.sidebar.radio(
        label="Job description source",
        options=["Paste New", "Select Existing"],
        index=0,
        help="Paste a brand new job description, or pick one you already created.",
    )

    if jd_mode == "Paste New":
        jd_title = st.sidebar.text_input(
            label="Job Title",
            placeholder="e.g. Senior Backend Engineer",
            help="A short label used to identify this job description later.",
        )
        jd_raw_text = st.sidebar.text_area(
            label="Paste Job Description",
            height=200,
            placeholder="Paste the full job description text here...",
        )

        if st.sidebar.button("Save Job Description", use_container_width=True):
            if not jd_title.strip() or not jd_raw_text.strip():
                st.sidebar.error("Please provide both a title and the job description text.")
            else:
                try:
                    cleaned_jd_text = parser_module.clean_text(jd_raw_text)
                    new_jd_id = database.insert_job_description(
                        title=jd_title.strip(),
                        raw_text=jd_raw_text,
                        cleaned_text=cleaned_jd_text,
                    )
                    st.session_state["active_job_description_id"] = new_jd_id
                    st.sidebar.success(f"Saved '{jd_title.strip()}' (ID {new_jd_id}).")
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to save job description: %s", exc)
                    st.sidebar.error(
                        "A database error occurred while saving the job description. "
                        "Please try again."
                    )

    else:  # jd_mode == "Select Existing"
        try:
            all_job_descriptions: List[Dict[str, Any]] = database.get_all_job_descriptions()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch job descriptions: %s", exc)
            all_job_descriptions = []
            st.sidebar.error("Could not load existing job descriptions from the database.")

        if not all_job_descriptions:
            st.sidebar.info("No job descriptions saved yet. Switch to 'Paste New' to create one.")
        else:
            label_to_id: Dict[str, int] = {
                f"{jd['title']}  (ID {jd['id']}, created {jd['created_at'][:10]})": jd["id"]
                for jd in all_job_descriptions
            }
            selected_label = st.sidebar.selectbox(
                label="Choose a Job Description",
                options=list(label_to_id.keys()),
            )
            st.session_state["active_job_description_id"] = label_to_id[selected_label]

    active_id: Optional[int] = st.session_state.get("active_job_description_id")
    if active_id is not None:
        st.sidebar.caption(f"Active Job Description ID: {active_id}")
    return active_id


# -----------------------------------------------------------------------------
# Section 2: Interactive weighting
# -----------------------------------------------------------------------------


def render_weighting_section() -> float:
    """Render the semantic/keyword weight slider and its live metrics.

    This is rendered before both the upload section and the leaderboard
    section so its return value can simply be passed down as a plain
    function argument, with no session-state relay required.

    Returns:
        float: The current HR-chosen semantic weight, in the range
        ``[0.0, 1.0]``. The keyword weight is always ``1.0`` minus this
        value.
    """
    st.header("1. Scoring Weights")

    semantic_weight: float = st.slider(
        label="Semantic Weight",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_SEMANTIC_WEIGHT,
        step=WEIGHT_SLIDER_STEP,
        help="How much to weigh meaning-based (semantic) similarity vs. exact keyword overlap.",
    )
    keyword_weight: float = 1.0 - semantic_weight

    weight_col_1, weight_col_2 = st.columns(2)
    weight_col_1.metric("Semantic Weight", f"{semantic_weight:.0%}")
    weight_col_2.metric("Keyword Weight", f"{keyword_weight:.0%}")

    return semantic_weight


# -----------------------------------------------------------------------------
# Section 3: Resume upload & async batch processing
# -----------------------------------------------------------------------------


def render_resume_upload_section(
    job_description_id: Optional[int],
    semantic_weight: float,
) -> None:
    """Render the multi-file resume uploader and drive async batch processing.

    HR chooses a batch size (how many resumes are embedded together in a
    single forward pass / processed concurrently), uploads one or more
    PDFs, and clicks "Process Resumes". The whole batch is scored via
    :func:`scoring_module.process_resumes_batch_async`, with a progress bar
    tracking overall completion and per-file success/failure surfaced
    individually afterward.

    Args:
        job_description_id: The primary key of the currently active job
            description. If ``None``, the uploader is disabled with a
            clear explanatory message, since resumes cannot be scored
            without a target job description.
        semantic_weight: The current HR-chosen semantic weight (0.0 to
            1.0), forwarded to the scoring pipeline for every resume in
            this batch.
    """
    st.header("2. Upload Candidate Resumes")

    if job_description_id is None:
        st.info("Please create or select a Job Description in the sidebar first.")
        return

    batch_size: int = st.slider(
        label="Batch size (resumes processed together)",
        min_value=MIN_BATCH_SIZE,
        max_value=MAX_BATCH_SIZE,
        value=DEFAULT_BATCH_SIZE,
        step=1,
        help=(
            "How many resumes are embedded and scored together per batch. "
            "Higher values are faster overall but use more memory/CPU at once."
        ),
    )

    uploaded_files = st.file_uploader(
        label="Upload one or more candidate resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Each file will be parsed, embedded, and scored against the active job description.",
    )

    process_clicked = st.button(
        "Process Resumes",
        type="primary",
        disabled=not uploaded_files,
    )

    if process_clicked and uploaded_files:
        total_files = len(uploaded_files)
        progress_bar = st.progress(0, text=f"Processing 0 / {total_files} resumes...")
        status_container = st.container()

        try:
            # Build the (file_bytes, filename, candidate_name) batch that
            # scoring_module.process_resumes_batch_async expects. Reading
            # .getvalue() up front avoids any risk of an already-consumed
            # read pointer on a Streamlit UploadedFile object.
            file_batch = [
                (uploaded_file.getvalue(), uploaded_file.name, None)
                for uploaded_file in uploaded_files
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read uploaded files: %s", exc)
            st.error("An error occurred while reading the uploaded files. Please try again.")
            return

        try:
            results: List[scoring_module.SubmissionResult] = run_async(
                scoring_module.process_resumes_batch_async(
                    file_batch,
                    job_description_id,
                    semantic_weight=semantic_weight,
                    batch_size=batch_size,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Batch processing failed: %s", exc)
            st.error(f"Batch processing failed: {exc}")
            return

        successes = 0
        failures = 0

        for index, result in enumerate(results, start=1):
            if result.success and result.breakdown is not None:
                successes += 1
                status_container.success(
                    f"\u2705 {result.filename}: scored {result.breakdown.final_score:.2%}"
                )
            else:
                failures += 1
                status_container.warning(f"\u26a0\ufe0f {result.filename}: {result.error_message}")

            progress_bar.progress(
                index / total_files,
                text=f"Processing {index} / {total_files} resumes...",
            )

        st.session_state["last_processing_summary"] = {
            "successes": successes,
            "failures": failures,
            "total": total_files,
        }
        st.rerun()

    summary: Optional[Dict[str, int]] = st.session_state.get("last_processing_summary")
    if summary:
        st.caption(
            f"Last batch: {summary['successes']} succeeded, "
            f"{summary['failures']} failed, out of {summary['total']} file(s)."
        )


# -----------------------------------------------------------------------------
# Section 4: Dynamic leaderboard
# -----------------------------------------------------------------------------


def render_leaderboard_section(
    job_description_id: Optional[int],
    semantic_weight: float,
) -> None:
    """Render the live, re-rankable leaderboard and CSV export button.

    Recomputes every existing candidate's final score under the current
    slider weight via :func:`scoring_module.rescore_leaderboard`, without
    re-parsing any PDFs or re-running the embedding model -- only the
    already-stored semantic/keyword scores are recombined.

    Args:
        job_description_id: The primary key of the currently active job
            description whose leaderboard should be displayed. If
            ``None``, an explanatory empty state is shown instead.
        semantic_weight: The current HR-chosen semantic weight (0.0 to
            1.0) used to recompute every candidate's final score.
    """
    st.header("3. Leaderboard")

    if job_description_id is None:
        st.info("Select or create a Job Description to see a leaderboard.")
        return

    try:
        leaderboard_rows: List[Dict[str, Any]] = database.get_leaderboard(job_description_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch leaderboard: %s", exc)
        st.error("A database error occurred while loading the leaderboard.")
        return

    if not leaderboard_rows:
        st.info("No resumes have been scored against this job description yet. Upload some above.")
        return

    try:
        recomputed_scores: Dict[int, scoring_module.ScoreBreakdown] = (
            scoring_module.rescore_leaderboard(
                job_description_id=job_description_id,
                semantic_weight=semantic_weight,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to rescore leaderboard: %s", exc)
        st.error("An error occurred while recalculating scores.")
        return

    display_rows: List[Dict[str, Any]] = []
    for row in leaderboard_rows:
        resume_id = int(row["resume_id"])
        breakdown = recomputed_scores.get(resume_id)
        if breakdown is None:
            continue

        display_name = row.get("candidate_name") or row["filename"]
        display_rows.append(
            {
                "Candidate": display_name,
                "Filename": row["filename"],
                "Semantic Score": round(breakdown.semantic_score, 4),
                "Keyword Score": round(breakdown.keyword_score, 4),
                "Final Score": round(breakdown.final_score, 4),
            }
        )

    if not display_rows:
        st.info("No resumes have been scored against this job description yet. Upload some above.")
        return

    leaderboard_df = pd.DataFrame(display_rows).sort_values(
        by="Final Score", ascending=False, ignore_index=True
    )
    leaderboard_df.insert(0, "Rank", range(1, len(leaderboard_df) + 1))

    st.dataframe(
        leaderboard_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Semantic Score": st.column_config.ProgressColumn(
                "Semantic Score", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "Keyword Score": st.column_config.ProgressColumn(
                "Keyword Score", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "Final Score": st.column_config.ProgressColumn(
                "Final Score", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )

    csv_bytes = leaderboard_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Report (CSV)",
        data=csv_bytes,
        file_name=f"ats_leaderboard_jd_{job_description_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# Application entry point
# -----------------------------------------------------------------------------


def main() -> None:
    """Configure the page and render every dashboard section in order.

    This function is the single top-level entry point for the Streamlit
    app. It is only invoked when this file is executed directly (e.g. via
    ``streamlit run hr_dashboard.py``), never on a plain ``import
    hr_dashboard`` from another script -- see the ``if __name__ ==
    "__main__"`` guard at the bottom of this file.

    Render order is: job description (sidebar) -> weighting slider ->
    resume upload (uses the weight) -> leaderboard (uses the same weight).
    Each function's return value is passed directly as an argument to the
    next, so no variable is ever read before it has been assigned.
    """
    st.set_page_config(
        page_title="AI-Powered ATS Dashboard",
        page_icon="\U0001F4C4",
        layout="wide",
    )

    _initialize_session_state()

    st.title("\U0001F4C4 AI-Powered Applicant Tracking System")
    st.caption(
        "Parse resumes, score them against a job description using semantic + "
        "keyword similarity, and rank candidates with adjustable weighting."
    )

    active_job_description_id: Optional[int] = render_job_description_section()
    semantic_weight: float = render_weighting_section()

    st.divider()

    render_resume_upload_section(active_job_description_id, semantic_weight)

    st.divider()

    render_leaderboard_section(active_job_description_id, semantic_weight)


if __name__ == "__main__":
    main()
