"""Pure validation for the isolated development environment contract."""

from django.core.exceptions import ImproperlyConfigured


def validate_development_database_override(database_name, production_database):
    database_name = (database_name or "").strip()
    production_database = (production_database or "").strip()
    allowed_prefixes = (
        "pharmacy_development_refresh_",
        "pharmacy_development_previous_",
        "pharmacy_development_failed_",
    )
    suffix = ""
    for prefix in allowed_prefixes:
        if database_name.startswith(prefix):
            suffix = database_name[len(prefix) :]
            break
    if not suffix or len(suffix) != 8 or any(
        character not in "0123456789abcdef" for character in suffix
    ):
        raise ImproperlyConfigured(
            "PHARMACY_DEVELOPMENT_DB_OVERRIDE is not a managed refresh database."
        )
    if database_name.casefold() == production_database.casefold():
        raise ImproperlyConfigured(
            "Development database override must not name production."
        )
    return database_name


def validate_development_environment(
    environment_name,
    development_database,
    production_database,
    resolved_production_database=None,
    test_database=None,
    development_database_user=None,
    production_database_user=None,
):
    environment_name = (environment_name or "").strip()
    development_database = (development_database or "").strip()
    production_database = (production_database or "").strip()
    resolved_production_database = (
        resolved_production_database or production_database
    ).strip()
    test_database = (test_database or "").strip()
    development_database_user = (development_database_user or "").strip()
    production_database_user = (production_database_user or "").strip()

    if environment_name != "development":
        raise ImproperlyConfigured(
            "Development requires PHARMACY_ENVIRONMENT=development in "
            ".env.development."
        )
    if not development_database:
        raise ImproperlyConfigured(
            "Development requires an explicit DB_NAME in .env.development."
        )
    if not production_database:
        raise ImproperlyConfigured(
            "Development requires PRODUCTION_DB_NAME so the live database can "
            "be excluded explicitly."
        )
    if production_database.casefold() != resolved_production_database.casefold():
        raise ImproperlyConfigured(
            "PRODUCTION_DB_NAME in .env.development does not match the live "
            "DB_NAME resolved from .env; refusing to start development."
        )
    if development_database.casefold() == resolved_production_database.casefold():
        raise ImproperlyConfigured(
            "Development DB_NAME must differ from PRODUCTION_DB_NAME; refusing "
            "to start against the production database."
        )
    if development_database.casefold() != "pharmacy_development":
        raise ImproperlyConfigured(
            "Development DB_NAME must be exactly pharmacy_development."
        )
    if test_database.casefold() != "test_pharmacy_development":
        raise ImproperlyConfigured(
            "DEVELOPMENT_TEST_DB_NAME must be exactly "
            "test_pharmacy_development."
        )
    if test_database.casefold() in {
        development_database.casefold(),
        resolved_production_database.casefold(),
        "postgres",
        "template0",
        "template1",
    }:
        raise ImproperlyConfigured(
            "The development test database must not name a live, development, "
            "or protected PostgreSQL database."
        )
    if development_database_user.casefold() != "pharmacy_development":
        raise ImproperlyConfigured(
            "Development DB_USER must be the isolated pharmacy_development role."
        )
    if (
        production_database_user
        and development_database_user.casefold()
        == production_database_user.casefold()
    ):
        raise ImproperlyConfigured(
            "Development and production must not use the same PostgreSQL role."
        )

    return development_database, production_database
