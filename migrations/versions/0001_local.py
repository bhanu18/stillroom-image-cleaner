"""Initial local schema. Existing prototype data is never upgraded in place."""

from alembic import op

from cleaner.db import SCHEMA

revision = "0001_local"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    for statement in SCHEMA:
        op.execute(statement)


def downgrade():
    raise RuntimeError("Destructive downgrade is unsupported; restore a consistent backup instead")
