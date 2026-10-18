import os

import psycopg

from psycopg import (
    sql,
)

from sqlalchemy import (
    URL,
)


# ================================================================
# 1. REQUIRED ENVIRONMENT VARIABLES
# ================================================================

REQUIRED_VARIABLES = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_ADMIN_USER",
    "DB_ADMIN_PASSWORD",
    "DB_APP_USER",
    "DB_APP_PASSWORD",
)


def get_required_environment():

    values = {}

    missing = []

    for name in REQUIRED_VARIABLES:

        value = os.getenv(
            name
        )

        if not value:

            missing.append(
                name
            )

        else:

            values[
                name
            ] = value


    if missing:

        missing_text = ", ".join(
            missing
        )

        raise RuntimeError(
            "Missing required environment variables: "
            f"{missing_text}"
        )


    return values


# ================================================================
# 2. BUILD ADMIN DATABASE URL
# ================================================================

def configure_admin_database_url(
    settings
):

    try:

        db_port = int(
            settings[
                "DB_PORT"
            ]
        )

    except ValueError as exc:

        raise RuntimeError(
            "DB_PORT must be a valid integer."
        ) from exc


    admin_url = URL.create(
        drivername=(
            "postgresql+psycopg"
        ),

        username=(
            settings[
                "DB_ADMIN_USER"
            ]
        ),

        password=(
            settings[
                "DB_ADMIN_PASSWORD"
            ]
        ),

        host=(
            settings[
                "DB_HOST"
            ]
        ),

        port=db_port,

        database=(
            settings[
                "DB_NAME"
            ]
        ),
    )


    os.environ[
        "DATABASE_URL"
    ] = admin_url.render_as_string(
        hide_password=False
    )


# ================================================================
# 3. CREATE OR UPDATE APPLICATION DATABASE ROLE
#
# PostgreSQL CREATE ROLE / ALTER ROLE does not accept a normal
# bind parameter for the PASSWORD clause.
#
# sql.Literal() safely quotes the password for PostgreSQL while
# sql.Identifier() safely quotes the role/database names.
# ================================================================

def configure_application_role(
    settings
):

    db_host = settings[
        "DB_HOST"
    ]

    db_port = int(
        settings[
            "DB_PORT"
        ]
    )

    db_name = settings[
        "DB_NAME"
    ]

    admin_user = settings[
        "DB_ADMIN_USER"
    ]

    admin_password = settings[
        "DB_ADMIN_PASSWORD"
    ]

    app_user = settings[
        "DB_APP_USER"
    ]

    app_password = settings[
        "DB_APP_PASSWORD"
    ]


    with psycopg.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=admin_user,
        password=admin_password,
        autocommit=True,
    ) as connection:

        with connection.cursor() as cursor:

            # ----------------------------------------------------
            # Check whether the application role already exists.
            # ----------------------------------------------------

            cursor.execute(
                """
                SELECT 1
                FROM pg_roles
                WHERE rolname = %s
                """,
                (
                    app_user,
                ),
            )


            role_exists = (
                cursor.fetchone()
                is not None
            )


            # ----------------------------------------------------
            # CREATE or UPDATE application role.
            #
            # We use sql.Literal() here instead of %s because
            # PostgreSQL does not support a bind parameter in the
            # CREATE ROLE / ALTER ROLE PASSWORD clause.
            # ----------------------------------------------------

            if role_exists:

                print(
                    "Application database role already exists. "
                    "Updating credentials..."
                )


                cursor.execute(
                    sql.SQL(
                        """
                        ALTER ROLE {}
                        WITH LOGIN PASSWORD {}
                        """
                    ).format(
                        sql.Identifier(
                            app_user
                        ),
                        sql.Literal(
                            app_password
                        ),
                    )
                )


            else:

                print(
                    "Creating application database role..."
                )


                cursor.execute(
                    sql.SQL(
                        """
                        CREATE ROLE {}
                        WITH LOGIN PASSWORD {}
                        """
                    ).format(
                        sql.Identifier(
                            app_user
                        ),
                        sql.Literal(
                            app_password
                        ),
                    )
                )


            # ----------------------------------------------------
            # Allow application user to connect to the database.
            # ----------------------------------------------------

            cursor.execute(
                sql.SQL(
                    """
                    GRANT CONNECT
                    ON DATABASE {}
                    TO {}
                    """
                ).format(
                    sql.Identifier(
                        db_name
                    ),
                    sql.Identifier(
                        app_user
                    ),
                )
            )


            # ----------------------------------------------------
            # Allow access to the public schema.
            # ----------------------------------------------------

            cursor.execute(
                sql.SQL(
                    """
                    GRANT USAGE
                    ON SCHEMA public
                    TO {}
                    """
                ).format(
                    sql.Identifier(
                        app_user
                    )
                )
            )


            # ----------------------------------------------------
            # Grant normal application table permissions.
            # ----------------------------------------------------

            cursor.execute(
                sql.SQL(
                    """
                    GRANT
                        SELECT,
                        INSERT,
                        UPDATE,
                        DELETE
                    ON ALL TABLES
                    IN SCHEMA public
                    TO {}
                    """
                ).format(
                    sql.Identifier(
                        app_user
                    )
                )
            )


            # ----------------------------------------------------
            # Grant sequence permissions.
            # ----------------------------------------------------

            cursor.execute(
                sql.SQL(
                    """
                    GRANT
                        USAGE,
                        SELECT,
                        UPDATE
                    ON ALL SEQUENCES
                    IN SCHEMA public
                    TO {}
                    """
                ).format(
                    sql.Identifier(
                        app_user
                    )
                )
            )


            # ----------------------------------------------------
            # Future tables created by the administrator should
            # automatically grant normal application permissions.
            # ----------------------------------------------------

            cursor.execute(
                sql.SQL(
                    """
                    ALTER DEFAULT PRIVILEGES
                    IN SCHEMA public
                    GRANT
                        SELECT,
                        INSERT,
                        UPDATE,
                        DELETE
                    ON TABLES
                    TO {}
                    """
                ).format(
                    sql.Identifier(
                        app_user
                    )
                )
            )


            cursor.execute(
                sql.SQL(
                    """
                    ALTER DEFAULT PRIVILEGES
                    IN SCHEMA public
                    GRANT
                        USAGE,
                        SELECT,
                        UPDATE
                    ON SEQUENCES
                    TO {}
                    """
                ).format(
                    sql.Identifier(
                        app_user
                    )
                )
            )


    print(
        "Application database role configured successfully."
    )


# ================================================================
# 4. RUN EXISTING DATABASE SEEDER
# ================================================================

def seed_application_database():

    # Import only after DATABASE_URL has been configured.
    from backend.seed_database import (
        main as seed_database,
    )


    print(
        "\nCreating and seeding application tables..."
    )


    seed_database()


# ================================================================
# 5. MAIN BOOTSTRAP FLOW
# ================================================================

def main():

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "AWS RDS DATABASE BOOTSTRAP"
    )

    print(
        "=" * 60
    )


    settings = (
        get_required_environment()
    )


    print(
        "Target database:",
        settings[
            "DB_NAME"
        ],
    )

    print(
        "Target host:",
        settings[
            "DB_HOST"
        ],
    )

    print(
        "Application role:",
        settings[
            "DB_APP_USER"
        ],
    )


    # ------------------------------------------------------------
    # Step 1:
    # Configure SQLAlchemy to connect as the RDS administrator.
    # ------------------------------------------------------------

    configure_admin_database_url(
        settings
    )


    # ------------------------------------------------------------
    # Step 2:
    # Create application tables and seed initial data.
    # ------------------------------------------------------------

    seed_application_database()


    # ------------------------------------------------------------
    # Step 3:
    # Create/configure the lower-privilege application user.
    # ------------------------------------------------------------

    configure_application_role(
        settings
    )


    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "RDS BOOTSTRAP COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()