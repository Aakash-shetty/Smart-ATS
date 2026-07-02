"""run_verification.py

Final pre-submission sanity check.

Imports all four ATS modules in dependency order and prints a single clear
success message if every import succeeds. Run this as the last step before
zipping up the project for the evaluator:

    python run_verification.py

If any module has a missing dependency, a syntax error, or a broken
import, this script will surface the exact traceback and exit with a
non-zero status code instead of silently failing later inside Streamlit.
"""

import sys


def main() -> int:
    """Attempt to import every ATS module and report the result.

    Returns:
        int: ``0`` if all modules imported successfully, ``1`` otherwise.
        Intended to be used directly as the process exit code so this
        script can also be wired into a CI step if desired.
    """
    try:
        import database  # noqa: F401
        import parser_module  # noqa: F401
        import scoring_module  # noqa: F401
        import hr_dashboard  # noqa: F401
    except ImportError as exc:
        print(f"❌ IMPORT FAILED: {exc}")
        print(
            "Double-check that all four files (database.py, parser_module.py, "
            "scoring_module.py, hr_dashboard.py) are in the same directory, and "
            "that every package in requirements.txt has been installed."
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"❌ UNEXPECTED ERROR DURING IMPORT: {exc}")
        return 1

    print("✅ SUCCESS: All four modules (database, parser_module, scoring_module, "
          "hr_dashboard) imported without error. The project is ready to run with:")
    print("    streamlit run hr_dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())