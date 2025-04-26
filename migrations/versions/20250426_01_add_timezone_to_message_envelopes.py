"""add timezone to message_envelopes

Revision ID: 20250426_01
Revises: None (dropping and recreating table per user instruction)
Create Date: 2025-04-26
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20250426_01"
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.drop_table("message_envelopes")
    op.create_table(
        "message_envelopes",
        sa.Column("envelope_id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("channel", sa.String(), nullable=True),
        sa.Column("instruction", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="received"),
        sa.Column("timezone", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("message_envelopes")
