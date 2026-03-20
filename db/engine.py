"""SQLite database engine setup."""

from pathlib import Path
from sqlmodel import SQLModel, Session, create_engine

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "mirofish.db"


def get_engine():
    DATA_DIR.mkdir(exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)


def init_db():
    """Create all tables."""
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    return engine


def get_session(engine=None):
    """Get a new database session."""
    if engine is None:
        engine = get_engine()
    return Session(engine)
