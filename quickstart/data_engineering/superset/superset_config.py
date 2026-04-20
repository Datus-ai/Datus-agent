"""
Superset configuration: use PostgreSQL as metadata database.
"""

SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://superset:superset@postgres:5432/superset_examples"
