# ATS Hackathon Project Context & Constraints

## 1. Project Goal
Build an AI-powered Applicant Tracking System (ATS) that parses resumes, calculates alignment scores against job descriptions using multiple NLP methods focusing on semantics or context than matching keyword count, and visualizes the results for HR professionals via an interactive dashboard with adjustable weightings.

## 2. Hard Constraints (CRITICAL)
- **Zero Local Testing Environment:** The user is on a restricted client machine and CANNOT run, test, or debug this code before submitting it for evaluation. 
- **Zero-Shot Reliability:** The code MUST be 100% production-ready, structurally perfect, and free of syntax errors. 
- **No Placeholders:** You are strictly forbidden from using `pass`, `...`, or comments like `# TODO: Implement this`. Every function must be fully implemented.
- **Zero Configuration:** Use SQLite for database storage so no external DB setup is required by the evaluator.
- **Don't Reinvent the Wheel:** Use robust, standard libraries (e.g., `FastAPI`, `scikit-learn`, `PyMuPDF`, `sentence-transformers`) instead of building mathematical functions or parsers from scratch also use standard system designs so easy to scale and manage (secure).

## 3. Tech Stack
- **Backend:** Python, FastAPI, Uvicorn, Pydantic.
- **NLP/ML:** `sentence-transformers` (for embeddings), `scikit-learn` (for TF-IDF and Cosine Similarity), `PyMuPDF` (for PDF parsing).
- **Database:** `sqlite3` (built-in Python).
- **Frontend/Visualization:** Streamlit (chosen for modularity, speed of deployment, and built-in interactive weighting widgets for HR).

## 4. Documentation Standards
- Every function and class must have a comprehensive docstring (Google style).
- Inline comments must explain *why* a specific approach was taken, especially for mathematical scoring.