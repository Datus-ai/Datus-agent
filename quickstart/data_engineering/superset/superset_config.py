"""Superset configuration: use PostgreSQL as metadata database."""

import os

SQLALCHEMY_DATABASE_URI = (
    "postgresql+psycopg2://"
    f"{os.environ.get('SUPERSET_DB_USER', 'superset')}:"
    f"{os.environ['SUPERSET_DB_PASSWORD']}@postgres:5432/"
    f"{os.environ.get('SUPERSET_DB_NAME', 'superset_examples')}"
)
