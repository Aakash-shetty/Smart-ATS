def render_resume_upload_section(job_description_id: Optional[int], semantic_weight: float) -> None:
    """Render multi-file uploader with configurable batch size and async processing."""
    st.header("2. Upload Candidate Resumes")

    if job_description_id is None:
        st.info("Please create or select a Job Description in the sidebar first.")
        return

    # HR configures how many resumes to process in parallel
    batch_size = st.slider(
        "Batch size (resumes processed in parallel)",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
        help="Higher = faster overall, but uses more CPU. Recommended: 10-15.",
    )

    uploaded_files = st.file_uploader(
        label="Upload one or more candidate resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Each file will be parsed, scored, and links extracted.",
    )

    process_clicked = st.button("Process Resumes", type="primary", disabled=not uploaded_files)

    if process_clicked and uploaded_files:
        total_files = len(uploaded_files)
        progress_bar = st.progress(0, text=f"Processing 0 / {total_files} resumes...")
        status_container = st.container()

        try:
            # Prepare file batch: (bytes, filename, candidate_name)
            file_batch = [
                (uploaded_file.getvalue(), uploaded_file.name, None)
                for uploaded_file in uploaded_files
            ]

            # Run async batch processing
            results = asyncio.run(
                scoring_module.process_resumes_batch_async(
                    file_batch,
                    job_description_id,
                    semantic_weight=semantic_weight,
                    batch_size=batch_size,
                )
            )

            successes = 0
            failures = 0

            for index, result in enumerate(results, start=1):
                if result.success:
                    successes += 1
                    status_container.success(
                        f"✅ {result.filename}: {result.breakdown.final_score:.2%}"
                    )
                else:
                    failures += 1
                    status_container.warning(
                        f"⚠️ {result.filename}: {result.error_message}"
                    )

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

        except Exception as exc:
            logger.error("Batch processing failed: %s", exc)
            st.error(f"Batch processing error: {exc}")

    summary = st.session_state.get("last_processing_summary")
    if summary:
        st.caption(
            f"Last batch: {summary['successes']} succeeded, "
            f"{summary['failures']} failed, out of {summary['total']} file(s)."
        )