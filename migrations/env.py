from alembic import context
from sqlalchemy import create_engine

from cleaner.config import Settings

config = context.config
connection = config.attributes.get("connection")


def migrate(connection):
    context.configure(connection=connection, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()


if connection is not None:
    migrate(connection)
else:
    engine = create_engine("sqlite:///" + str(Settings.env().data / "library.sqlite3"))
    with engine.connect() as connection:
        migrate(connection)
