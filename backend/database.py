import os

from pathlib import Path

from sqlalchemy import (
    create_engine,
)

from sqlalchemy.orm import (
    DeclarativeBase,
    sessionmaker,
)

from dotenv import (
    load_dotenv,
)


# ================================================================
# 1. PROJECT PATHS
# ================================================================

BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PROJECT_ROOT = (
    BACKEND_DIR
    .parent
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# 2. ENVIRONMENT VARIABLES
# ================================================================

load_dotenv(
    PROJECT_ROOT
    / ".env"
)


# ================================================================
# 3. DEFAULT LOCAL DATABASE
#
# We deliberately begin with SQLite so we can prove the database
# architecture without requiring PostgreSQL to be installed yet.
#
# Later:
#
# DATABASE_URL=postgresql+psycopg://...
#
# will switch the same SQLAlchemy code to PostgreSQL.
# ================================================================

DEFAULT_DB_FILE = (
    DATA_DIR
    / "operations.db"
)

DEFAULT_DATABASE_URL = (
    "sqlite:///"
    +
    DEFAULT_DB_FILE
    .as_posix()
)


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    DEFAULT_DATABASE_URL,
)


# ================================================================
# 4. SQLALCHEMY BASE
#
# All database models inherit from this class.
# ================================================================

class Base(
    DeclarativeBase
):
    pass


# ================================================================
# 5. ENGINE SETTINGS
# ================================================================

engine_kwargs = {
    "pool_pre_ping":
        True,
}


# SQLite needs this option for local FastAPI development.
if DATABASE_URL.startswith(
    "sqlite"
):

    engine_kwargs[
        "connect_args"
    ] = {
        "check_same_thread":
            False,
    }


engine = create_engine(
    DATABASE_URL,
    **engine_kwargs,
)


# ================================================================
# 6. DATABASE SESSION
#
# A session is the unit through which Python reads/writes records.
# ================================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# ================================================================
# 7. FASTAPI DATABASE DEPENDENCY
#
# Later our API routes can use:
#
#     db = Depends(get_db)
#
# The session always closes safely after the request.
# ================================================================

def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()