"""database.py

Zero-configuration persistence layer for the ATS hackathon project.

This module wraps Python's built-in ``sqlite3`` library to provide a
self-initializing, file-based database. No external database server,
credentials, or manual migration step is required: the very first time
this module is imported, it guarantees that the SQLite file and every
required table exist.

Schema Overview:
    - job_descriptions: Stores raw job description text submitted by HR.
    - resumes: Stores the raw (original) and cleaned/parsed text of every
      uploaded candidate resume.
    - scores: Stores the individual method scores (semantic, keyword) and
      the resulting weighted final score for every (resume, job_description)
      pair that has been evaluated.

Design Notes:
    - We use ``sqlite3.Row`` as the row factory so query results behave like
      dictionaries (accessible by column name), which keeps the calling
      code (FastAPI routes, Streamlit dashboard) decoupled from raw tuple
      indexing.
    - Foreign keys are enabled explicitly on every connection because
      SQLite disables foreign key enforcement by default for backwards
      compatibility reasons.
    - All write operations are wrapped in try/except blocks and roll back
      the transaction on failure so a single bad insert can never leave the
      database in a half-written state.
"""

import sqlite3
import logging
import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

# -----------------------------------------------------------------------------
# Module-level configuration
# -----------------------------------------------------------------------------

# The database file lives alongside this module so the project is fully
# portable (no absolute paths, no environment variables required).
DB_PATH: Path = Path(__file__).resolve().parent / "ats_database.db"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_connection() -> sqlite3.Connection:
    """Create and return a new SQLite connection configured for this project.

    A new connection is created per call (rather than sharing one global
    connection) because SQLite connections are not guaranteed to be safe
    to share across threads, and both FastAPI (async workers) and Streamlit
    (per-session reruns) may access the database from different threads.

    Returns:
        sqlite3.Connection: An open connection with ``row_factory`` set to
        ``sqlite3.Row`` (dict-like row access) and foreign key enforcement
        turned on.
    """
    connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    # SQLite ships with foreign_keys OFF by default; we turn it on per
    # connection so ON DELETE/UPDATE behavior and referential integrity
    # are actually enforced.
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


@contextmanager
def db_cursor() -> Iterator[sqlite3.Cursor]:
    """Provide a transactional cursor as a context manager.

    Commits automatically on successful exit of the ``with`` block, and
    rolls back automatically if any exception is raised inside the block.
    The underlying connection is always closed, guaranteeing no dangling
    file handles even under error conditions.

    Yields:
        sqlite3.Cursor: A cursor bound to a fresh connection.

    Raises:
        sqlite3.Error: Re-raised after rollback so calling code can decide
        how to handle the failure (the connection itself is always cleaned
        up regardless).
    """
    connection = get_connection()
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        logger.error("Database transaction failed, rolled back: %s", exc)
        raise
    finally:
        cursor.close()
        connection.close()


def initialize_database() -> None:
    """Create every required table if it does not already exist.

    This function is idempotent: calling it multiple times (e.g., once per
    application restart) is always safe because every statement uses
    ``CREATE TABLE IF NOT EXISTS``. This satisfies the "zero configuration"
    project constraint -- the evaluator never has to run a separate setup
    or migration script.
    """
    schema_statements: List[str] = [
        """
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            cleaned_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            cleaned_text TEXT NOT NULL,
            candidate_name TEXT,
            uploaded_at TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER NOT NULL,
            job_description_id INTEGER NOT NULL,
            semantic_score REAL NOT NULL,
            keyword_score REAL NOT NULL,
            semantic_weight REAL NOT NULL,
            keyword_weight REAL NOT NULL,
            final_score REAL NOT NULL,
            computed_at TEXT NOT NULL,
            FOREIGN KEY (resume_id) REFERENCES resumes (id) ON DELETE CASCADE,
            FOREIGN KEY (job_description_id) REFERENCES job_descriptions (id) ON DELETE CASCADE
        );
        """,
        # Index to keep leaderboard queries (scores for a given JD, ordered
        # by final_score) fast even as the resume pool grows.
        """
        CREATE INDEX IF NOT EXISTS idx_scores_jd_final
        ON scores (job_description_id, final_score DESC);
        """,
    ]

    with db_cursor() as cursor:
        for statement in schema_statements:
            cursor.execute(statement)

    logger.info("Database initialized successfully at %s", DB_PATH)


def _utcnow_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string.

    Centralizing timestamp generation avoids inconsistent formats across
    the different insert functions below.

    Returns:
        str: Current UTC time, e.g. ``"2026-07-02T10:15:00.123456+00:00"``.
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# Job Description operations
# -----------------------------------------------------------------------------


def insert_job_description(title: str, raw_text: str, cleaned_text: str) -> int:
    """Persist a new job description.

    Args:
        title: A short, human-readable label for the job posting (e.g.
            "Senior Backend Engineer").
        raw_text: The exact, unmodified job description text as pasted by HR.
        cleaned_text: The normalized/cleaned version of the text used for
            NLP scoring (produced by ``parser_module.clean_text``).

    Returns:
        int: The auto-generated primary key of the newly inserted row.

    Raises:
        sqlite3.Error: If the insert fails for any database-level reason.
    """
    with db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO job_descriptions (title, raw_text, cleaned_text, created_at)
            VALUES (?, ?, ?, ?);
            """,
            (title, raw_text, cleaned_text, _utcnow_iso()),
        )
        return int(cursor.lastrowid)


def get_job_description(job_description_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single job description by its primary key.

    Args:
        job_description_id: The primary key of the job description to fetch.

    Returns:
        Optional[Dict[str, Any]]: A dictionary of the row's columns, or
        ``None`` if no matching job description exists.
    """
    with db_cursor() as cursor:
        cursor.execute(
            "SELECT * FROM job_descriptions WHERE id = ?;",
            (job_description_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None


def get_all_job_descriptions() -> List[Dict[str, Any]]:
    """Fetch every stored job description, most recent first.

    Returns:
        List[Dict[str, Any]]: A list of job description rows as dictionaries.
        Returns an empty list if no job descriptions have been created yet.
    """
    with db_cursor() as cursor:
        cursor.execute("SELECT * FROM job_descriptions ORDER BY created_at DESC;")
        return [dict(row) for row in cursor.fetchall()]


# -----------------------------------------------------------------------------
# Resume operations
# -----------------------------------------------------------------------------


def insert_resume(
    filename: str,
    raw_text: str,
    cleaned_text: str,
    candidate_name: Optional[str] = None,
) -> int:
    """Persist a newly parsed resume.

    Args:
        filename: The original uploaded filename (used for display in the
            HR dashboard leaderboard).
        raw_text: The unmodified text extracted from the resume PDF.
        cleaned_text: The normalized text used for downstream NLP scoring.
        candidate_name: An optional human-readable candidate name, if it was
            successfully detected during parsing. Defaults to ``None`` when
            unknown, in which case the UI should fall back to ``filename``.

    Returns:
        int: The auto-generated primary key of the newly inserted resume row.

    Raises:
        sqlite3.Error: If the insert fails for any database-level reason.
    """
    with db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO resumes (filename, raw_text, cleaned_text, candidate_name, uploaded_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (filename, raw_text, cleaned_text, candidate_name, _utcnow_iso()),
        )
        return int(cursor.lastrowid)


def get_resume(resume_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single resume by its primary key.

    Args:
        resume_id: The primary key of the resume to fetch.

    Returns:
        Optional[Dict[str, Any]]: A dictionary of the row's columns, or
        ``None`` if no matching resume exists.
    """
    with db_cursor() as cursor:
        cursor.execute("SELECT * FROM resumes WHERE id = ?;", (resume_id,))
        row = cursor.fetchone()
        return dict(row) if row is not None else None


def get_all_resumes() -> List[Dict[str, Any]]:
    """Fetch every stored resume, most recently uploaded first.

    Returns:
        List[Dict[str, Any]]: A list of resume rows as dictionaries. Returns
        an empty list if no resumes have been uploaded yet.
    """
    with db_cursor() as cursor:
        cursor.execute("SELECT * FROM resumes ORDER BY uploaded_at DESC;")
        return [dict(row) for row in cursor.fetchall()]


# -----------------------------------------------------------------------------
# Score operations
# -----------------------------------------------------------------------------


def insert_score(
    resume_id: int,
    job_description_id: int,
    semantic_score: float,
    keyword_score: float,
    semantic_weight: float,
    keyword_weight: float,
    final_score: float,
) -> int:
    """Persist a computed score for one (resume, job description) pair.

    Args:
        resume_id: Primary key of the resume that was scored.
        job_description_id: Primary key of the job description it was
            scored against.
        semantic_score: Cosine similarity between sentence-embedding
            vectors, in the range [0.0, 1.0].
        keyword_score: Cosine similarity between TF-IDF vectors, in the
            range [0.0, 1.0].
        semantic_weight: The HR-adjustable weight applied to the semantic
            score when computing ``final_score`` (0.0 to 1.0).
        keyword_weight: The HR-adjustable weight applied to the keyword
            score when computing ``final_score`` (0.0 to 1.0).
        final_score: The resulting weighted combination of the two method
            scores, i.e. ``semantic_score * semantic_weight +
            keyword_score * keyword_weight``.

    Returns:
        int: The auto-generated primary key of the newly inserted score row.

    Raises:
        sqlite3.Error: If the insert fails for any database-level reason.
    """
    with db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO scores (
                resume_id, job_description_id, semantic_score, keyword_score,
                semantic_weight, keyword_weight, final_score, computed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                resume_id,
                job_description_id,
                semantic_score,
                keyword_score,
                semantic_weight,
                keyword_weight,
                final_score,
                _utcnow_iso(),
            ),
        )
        return int(cursor.lastrowid)


def get_leaderboard(job_description_id: int) -> List[Dict[str, Any]]:
    """Fetch every score for a job description, ranked highest-first.

    This performs a join against ``resumes`` so the HR dashboard can render
    a leaderboard with candidate filenames/names without a second query.

    Args:
        job_description_id: The job description to build a leaderboard for.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the
        resume's filename/candidate_name alongside its score breakdown,
        sorted by ``final_score`` descending. Returns an empty list if no
        scores exist yet for this job description.
    """
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT
                scores.id AS score_id,
                scores.resume_id AS resume_id,
                scores.job_description_id AS job_description_id,
                scores.semantic_score AS semantic_score,
                scores.keyword_score AS keyword_score,
                scores.semantic_weight AS semantic_weight,
                scores.keyword_weight AS keyword_weight,
                scores.final_score AS final_score,
                scores.computed_at AS computed_at,
                resumes.filename AS filename,
                resumes.candidate_name AS candidate_name
            FROM scores
            INNER JOIN resumes ON resumes.id = scores.resume_id
            WHERE scores.job_description_id = ?
            ORDER BY scores.final_score DESC;
            """,
            (job_description_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def delete_score(score_id: int) -> bool:
    """Delete a single score record by its primary key.

    Args:
        score_id: The primary key of the score row to delete.

    Returns:
        bool: ``True`` if a row was deleted, ``False`` if no row with that
        ID existed.
    """
    with db_cursor() as cursor:
        cursor.execute("DELETE FROM scores WHERE id = ?;", (score_id,))
        return cursor.rowcount > 0


# -----------------------------------------------------------------------------
# Auto-initialization on import
# -----------------------------------------------------------------------------
# Per the project's "zero configuration" constraint, the schema must exist
# the moment any other module imports this file -- no separate "run this
# setup script first" step should ever be required of the evaluator.
initialize_database()
