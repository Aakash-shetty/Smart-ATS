"""hr_dashboard.py

Module C: Visualization & HR Interface.

This is the Streamlit front-end for the ATS. It ties together every prior
module:
    - ``database``       : job description / resume / score persistence.
    - ``parser_module``  : PDF text extraction + text cleaning.
    - ``scoring_module``entative : the weighted scoring service layer.

Run with:
    streamlit run hr_dashboard.py

All UI-rendering code lives inside ``main()`` (invoked only when this file
is executed directly), so importing this module elsewhere -- e.g. from a
verification script -- never triggers Streamlit UI calls outside of an
actual Streamlit runtime.
"""

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
SLIDER_STEP: float = 0.05


# -----------------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------------


def _initialize_session_state() -> None:
    """Ensure every key this dashboard relies on exists in ``st.session_state``.

    Streamlit reruns the entire script top-to-bottom on every user
    interaction (slider drag, button click, file upload). Session state is
    the only mechanism that survives across those reruns within a single
    browser session, so every piece of state the dashboard needs to
    remember (which job description is active, the last processing
    summary) is initialized once here, defensively, before any widget
    reads or writes it.
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

    # Radio toggle between "create new" and "select existing" so the two
    # workflows never fight for the same screen space at once.
    jd_mode = st.sidebar.radio(
        label="Job description source",
        options=["Paste New", "Select Existing"],
        index=0,
        help="Paste a brand new job description, or pick one you already created.",
    )

    if jd_mode == "Paste New":
        # Free-text title so the dropdown (used later, in "Select Existing")
        # is human-readable instead of showing raw database IDs.
        jd_title = st.sidebar.text_input(
            label="Job Title",
            placeholder="e.g. Senior Backend Engineer",
            help="A short label used to identify this job description later.",
        )
        # Large text area for the full job description text HR pastes in.
        jd_raw_text = st.sidebar.text_area(
            label="Paste Job Description",
            height=200,
            placeholder="Paste the full job description text here...",
        )

        if st.sidebar.button("Save Job Description", use_container_width=True):
            if not jd_title.strip() or not jd_raw_text.strip():
                # Guard against empty submissions rather than silently
                # writing a blank row to the database.
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
            # Empty-state handling: nothing to select yet.
            st.sidebar.info("No job descriptions saved yet. Switch to 'Paste New' to create one.")
        else:
            # Build a human-readable label -> id mapping so the selectbox
            # shows titles (and creation dates) rather than raw IDs.
            label_to_id: Dict[str, int] = {
                f"{jd['title']}  (ID {jd['id']}, created {jd['created_at'][:10]})": jd["id"]
                for jd in all_job_descriptions
            }
            selected_label = st.sidebar.selectbox(
                label="Choose a Job Description",
                options=list(label_to_id.keys()),
            )
            st.session_state["active_job_description_id"] = label_to_id[selected_label]

    active_id = st.session_state.get("active_job_description_id")
    if active_id is not None:
        st.sidebar.caption(f"Active Job Description ID: {active_id}")
    return active_id


# -----------------------------------------------------------------------------
# Section 2: Resume upload & processing
# -----------------------------------------------------------------------------


def render_resume_upload_section(job_description_id: Optional[int], semantic_weight: float) -> None:
    """Render the multi-file resume uploader and drive the processing pipeline.

    Each uploaded PDF is run through
    :func:`scoring_module.process_candidate_submission`, with a progress
    bar tracking overall batch progress and per-file success/failure
    surfaced individually so HR knows exactly which resumes did or did not
    make it into the leaderboard.

    Args:
        job_description_id: The primary key of the currently active job
            description. If ``None``, the uploader is disabled with a
            clear explanatory message, since resumes cannot be scored
            without a target job description.
        semantic_weight: The current HR-chosen semantic weight (0.0 to
            1.0), used to compute each new resume's initial score at
            upload time.
    """
    st.header("2. Upload Candidate Resumes")

    if job_description_id is None:
        st.info("Please create or select a Job Description in the sidebar first.")
        return

    # Multi-file uploader restricted to PDFs, matching the parsing pipeline
    # which only knows how to read PDF content.
    uploaded_files = st.file_uploader(
        label="Upload one or more candidate resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Each file will be parsed, embedded, and scored against the active job description.",
    )

    process_clicked = st.button("Process Resumes", type="primary", disabled=not uploaded_files)

    if process_clicked and uploaded_files:
        total_files = len(uploaded_files)
        # Visual progress bar so HR gets feedback during what can be a
        # multi-second operation (PDF parsing + embedding model inference
        # per resume).
        progress_bar = st.progress(0, text=f"Processing 0 / {total_files} resumes...")
        status_container = st.container()

        successes = 0
        failures = 0

        for index, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                # Read the file's raw bytes up front. Passing bytes (rather
                # than the UploadedFile object itself) avoids any risk of
                # the underlying buffer's read-position having already been
                # consumed by a prior Streamlit rerun.
                file_bytes = uploaded_file.getvalue()

                result = scoring_module.process_candidate_submission(
                    uploaded_file=file_bytes,
                    filename=uploaded_file.name,
                    job_description_id=job_description_id,
                    semantic_weight=semantic_weight,
                )

                if result.success:
                    successes += 1
                    status_container.success(
                        f"✅ {uploaded_file.name}: scored "
                        f"{result.breakdown.final_score:.2%}"
                    )
                else:
                    failures += 1
                    status_container.warning(f"⚠️ {uploaded_file.name}: {result.error_message}")

            except Exception as exc:  # noqa: BLE001
                # A final safety net: even an unexpected exception for one
                # file must not stop the rest of the batch from processing.
                failures += 1
                logger.error("Unexpected error processing '%s': %s", uploaded_file.name, exc)
                status_container.error(f"❌ {uploaded_file.name}: an unexpected error occurred.")

            progress_bar.progress(
                index / total_files,
                text=f"Processing {index} / {total_files} resumes...",
            )

        st.session_state["last_processing_summary"] = {
            "successes": successes,
            "failures": failures,
            "total": total_files,
        }
        st.rerun()  # Refresh so the leaderboard below reflects the new data immediately.

    # Show the outcome of the most recent batch, if any, even after the rerun.
    summary = st.session_state.get("last_processing_summary")
    if summary:
        st.caption(
            f"Last batch: {summary['successes']} succeeded, "
            f"{summary['failures']} failed, out of {summary['total']} file(s)."
        )


# -----------------------------------------------------------------------------
# Section 3 & 4: Interactive weighting + dynamic leaderboard
# -----------------------------------------------------------------------------


def render_weighting_and_leaderboard_section(job_description_id: Optional[int]) -> float:
    """Render the weight slider and the live, re-rankable leaderboard.

    The semantic/keyword weight slider drives an immediate recalculation
    of every existing candidate's final score via
    :func:`scoring_module.rescore_leaderboard`, without needing to
    re-parse any PDFs or re-run the embedding model -- only the already
    stored semantic/keyword scores are recombined under the new weight.

    Args:
        job_description_id: The primary key of the currently active job
            description whose leaderboard should be displayed. If
            ``None``, an explanatory empty state is shown instead.

    Returns:
        float: The current semantic weight selected by HR, so callers
        (e.g. the resume upload section) can use the same value when
        scoring newly uploaded resumes.
    """
    st.header("3. Scoring Weights & Leaderboard")

    # The slider is always shown so HR can set their preferred weighting
    # even before any job description/resumes exist yet.
    semantic_weight = st.slider(
        label="Semantic Weight",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_SEMANTIC_WEIGHT,
        step=SLIDER_STEP,
        help="How much to weigh meaning-based (semantic) similarity vs. exact keyword overlap.",
    )
    keyword_weight = 1.0 - semantic_weight

    # Two side-by-side metrics make the weight split immediately legible
    # without HR having to do the subtraction themselves.
    weight_col_1, weight_col_2 = st.columns(2)
    weight_col_1.metric("Semantic Weight", f"{semantic_weight:.0%}")
    weight_col_2.metric("Keyword Weight", f"{keyword_weight:.0%}")

    if job_description_id is None:
        st.info("Select or create a Job Description to see a leaderboard.")
        return semantic_weight

    try:
        leaderboard_rows: List[Dict[str, Any]] = database.get_leaderboard(job_description_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch leaderboard: %s", exc)
        st.error("A database error occurred while loading the leaderboard.")
        return semantic_weight

    if not leaderboard_rows:
        # Empty-state handling: no resumes scored against this JD yet.
        st.info("No resumes have been scored against this job description yet. Upload some above.")
        return semantic_weight

    # Recompute every candidate's score under the current slider weight.
    # This is cheap (pure arithmetic on already-stored numbers) so it can
    # safely run on every single script rerun the slider triggers.
    try:
        recomputed_scores = scoring_module.rescore_leaderboard(
            job_description_id=job_description_id,
            semantic_weight=semantic_weight,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to rescore leaderboard: %s", exc)
        st.error("An error occurred while recalculating scores.")
        return semantic_weight

    # Build a clean, display-ready table: candidate name/filename plus the
    # three score columns, sorted by final_score descending.
    display_rows: List[Dict[str, Any]] = []
    for row in leaderboard_rows:
        resume_id = int(row["resume_id"])
        breakdown = recomputed_scores.get(resume_id)
        if breakdown is None:
            continue  # Defensive: skip rows that failed to recompute.

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

    leaderboard_df = pd.DataFrame(display_rows).sort_values(
        by="Final Score", ascending=False, ignore_index=True
    )
    # 1-indexed rank column reads more naturally to HR than a 0-indexed one.
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

    # CSV export of exactly what's currently displayed, so the downloaded
    # report always matches the on-screen ranking under the chosen weights.
    csv_bytes = leaderboard_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Report (CSV)",
        data=csv_bytes,
        file_name=f"ats_leaderboard_jd_{job_description_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    return semantic_weight


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
    """
    st.set_page_config(
        page_title="AI-Powered ATS Dashboard",
        page_icon="📄",
        layout="wide",
    )

    _initialize_session_state()

    st.title("📄 AI-Powered Applicant Tracking System")
    st.caption(
        "Parse resumes, score them against a job description using semantic + "
        "keyword similarity, and rank candidates with adjustable weighting."
    )

    # Section 1 lives in the sidebar; it returns the currently active job
    # description ID, which every other section depends on.
    active_job_description_id = render_job_description_section()

    # Section 3's slider value is needed by Section 2 (to score newly
    # uploaded resumes), so we render the weighting UI first, then reuse
    # its returned weight when processing uploads directly below it.
    # To keep the on-screen ordering matching the numbered requirements
    # (upload before leaderboard), we read the weight via a placeholder
    # default first, render uploads, then render the authoritative
    # weighting + leaderboard section afterward.
    current_semantic_weight = st.session_state.get("last_used_semantic_weight", DEFAULT_SEMANTIC_WEIGHT)

    render_resume_upload_section(active_job_description_id, current_semantic_weight)

    st.divider()

    final_semantic_weight = render_weighting_and_leaderboard_section(active_job_description_id)
    st.session_state["last_used_semantic_weight"] = final_semantic_weight


if __name__ == "__main__":
    main()