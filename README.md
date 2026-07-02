# AI-Powered Intelligent ATS (Applicant Tracking System)

## The Problem
HR departments are overwhelmed by the volume of incoming applications. Traditional Applicant Tracking Systems rely primarily on simple keyword counting. This leads to two major issues:
1. **Inefficiency:** Manual screening is slow and prone to human fatigue.
2. **Poor Accuracy:** Keyword-based systems are easily "gamed" by candidates who stuff their resumes with buzzwords, while highly qualified candidates who use different terminology are often unfairly filtered out.

## Our Solution
We built an intelligent matchmaking engine for recruitment. Instead of just counting words, our system understands the **contextual meaning** of a resume and compares it against the job description. 

We empower HR professionals to balance **semantic intelligence** with **lexical precision** using a real-time, interactive dashboard.

### The Logic Flow
![System Flowchart](<img width="476" height="346" alt="ats" src="https://github.com/user-attachments/assets/0ec4d906-5d16-4807-809f-c207072bd694" />
)

1. **Upload & Parse:** Upload resumes (PDF) and extract clean, normalized text.
2. **Dual-Engine Scoring:**
   - **Meaning Match (Semantic):** Uses deep learning embeddings to understand job roles and skills, even if the phrasing differs from the job description.
   - **Keyword Match (Lexical):** Provides a classic check to ensure specific mandatory hard skills are present.
3. **Interactive Weighting:** HR can adjust the balance between semantic and keyword matching via a slider to suit specific hiring needs.
4. **Instant Ranking:** A dynamic leaderboard identifies the best-fit candidates instantly.

---

### How It Works: A Toy Example

| Rank | Candidate | Meaning Match | Keyword Match | Final Score (70/30) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `resume_amit.pdf` | 0.88 | 0.62 | **0.80** |
| 2 | `resume_priya.pdf` | 0.79 | 0.75 | **0.78** |
| 3 | `resume_karan.pdf` | 0.55 | 0.90 | **0.65** |

**Why this matters:** Notice candidate 3. They have a high keyword score but low meaning score—likely a candidate who "keyword-stuffed" their resume. Our system exposes this, allowing HR to prioritize candidates who truly demonstrate relevant experience (like candidates 1 and 2).

---

## Tech Stack
* **Backend:** FastAPI (Modularized, high-performance API)
* **Parsing:** PyMuPDF (Clean extraction)
* **Intelligence:** `sentence-transformers` (all-MiniLM-L6-v2) for semantic embeddings & `scikit-learn` for TF-IDF.
* **Storage:** SQLite (Zero-setup, portable database)
* **Interface:** Streamlit (Interactive HR Dashboard)

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Launch the dashboard: `streamlit run hr_dashboard.py`

---
*Built for ATS Hackathon*
