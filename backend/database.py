import os

from pathlib import Path

from sqlalchemy import (
    URL,
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
# 2. LOAD LOCAL ENVIRONMENT VARIABLES
#
# Local development can continue using:
#
# DATABASE_URL=postgresql+psycopg://...
#
# ECS can instead provide:
#
# DB_HOST
# DB_PORT
# DB_NAME
# DB_USER
# DB_PASSWORD
# ================================================================

load_dotenv(
    PROJECT_ROOT
    / ".env"
)


# ================================================================
# 3. DEFAULT LOCAL SQLITE DATABASE
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


# ================================================================
# 4. BUILD DATABASE CONNECTION
#
# Priority:
#
# 1. DATABASE_URL
# 2. AWS/ECS DB_* variables
# 3. SQLite local fallback
#
# SQLAlchemy URL.create() safely handles special characters
# inside the database password.
# ================================================================

def build_database_url():

    explicit_database_url = os.getenv(
        "DATABASE_URL"
    )

    if explicit_database_url:

        return explicit_database_url


    db_host = os.getenv(
        "DB_HOST"
    )

    db_port = os.getenv(
        "DB_PORT",
        "5432",
    )

    db_name = os.getenv(
        "DB_NAME"
    )

    db_user = os.getenv(
        "DB_USER"
    )

    db_password = os.getenv(
        "DB_PASSWORD"
    )


    aws_values = {
        "DB_HOST": db_host,
        "DB_NAME": db_name,
        "DB_USER": db_user,
        "DB_PASSWORD": db_password,
    }


    supplied_aws_values = [
        name
        for name, value
        in aws_values.items()
        if value
    ]


    # ------------------------------------------------------------
    # If AWS-style variables have started being supplied,
    # require all required values.
    #
    # This prevents a production container from silently falling
    # back to SQLite because one ECS variable was forgotten.
    # ------------------------------------------------------------

    if supplied_aws_values:

        missing_values = [
            name
            for name, value
            in aws_values.items()
            if not value
        ]

        if missing_values:

            missing_text = ", ".join(
                missing_values
            )

            raise RuntimeError(
                "Incomplete database configuration. "
                f"Missing: {missing_text}"
            )


        try:

            port_number = int(
                db_port
            )

        except ValueError as exc:

            raise RuntimeError(
                "DB_PORT must be a valid integer."
            ) from exc


        return URL.create(
            drivername=(
                "postgresql+psycopg"
            ),
            username=db_user,
            password=db_password,
            host=db_host,
            port=port_number,
            database=db_name,
        )


    return DEFAULT_DATABASE_URL


DATABASE_URL = build_database_url()


# ================================================================
# 5. SQLALCHEMY BASE
# ================================================================

class Base(
    DeclarativeBase
):
    pass


# ================================================================
# 6. ENGINE SETTINGS
# ================================================================

engine_kwargs = {
    "pool_pre_ping":
        True,
}


# SQLite requires this setting for local FastAPI development.
if str(
    DATABASE_URL
).startswith(
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
# 7. DATABASE SESSION
# ================================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# ================================================================
# 8. FASTAPI DATABASE DEPENDENCY
# ================================================================

def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()