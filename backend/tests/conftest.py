"""Test setup: a small throw-away SQLite database seeded from the shipped dataset.

The tests never touch backend/data/argocast.db. Set ARGOCAST_TEST_USE_MAIN_DB=1 to
run them against the main database instead.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if not os.environ.get("ARGOCAST_TEST_USE_MAIN_DB"):
    _db = Path(tempfile.mkdtemp(prefix="argocast_test_")) / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{_db.as_posix()}"
    subprocess.run(
        [sys.executable, str(BACKEND_DIR / "scripts" / "seed_database.py"), "--floats", "6"],
        check=True, env=os.environ.copy(), cwd=BACKEND_DIR, stdout=subprocess.DEVNULL,
    )
