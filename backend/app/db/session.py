import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from app.models.schema import Base

# Database file is in backend/data/ relative to the backend directory
# session.py is at backend/app/db/session.py, so 3 dirnames to get to backend/
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_PATH = os.path.join(BACKEND_DIR, "data", "argocast.db")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"
elif DATABASE_URL.startswith("sqlite:///"):
    db_file_path = DATABASE_URL[len("sqlite:///"):]
    # If the relative path does not exist from current working directory, resolve against root or backend
    if not os.path.isabs(db_file_path) and not os.path.exists(db_file_path):
        workspace_root = os.path.dirname(BACKEND_DIR)
        candidate_root = os.path.normpath(os.path.join(workspace_root, db_file_path))
        clean_rel = db_file_path.lstrip("./").lstrip(".\\")
        if clean_rel.startswith("backend/") or clean_rel.startswith("backend\\"):
            clean_rel = clean_rel[8:]
        candidate_backend = os.path.normpath(os.path.join(BACKEND_DIR, clean_rel))

        if os.path.exists(candidate_root):
            DATABASE_URL = f"sqlite:///{candidate_root}"
        elif os.path.exists(candidate_backend):
            DATABASE_URL = f"sqlite:///{candidate_backend}"
        elif os.path.exists(DEFAULT_DB_PATH):
            DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )
else:
    engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()