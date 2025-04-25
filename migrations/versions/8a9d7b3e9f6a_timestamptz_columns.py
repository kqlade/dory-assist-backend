"""add timezone-aware timestamps

Revision ID: 8a9d7b3e9f6a
Revises: 99bae3f947d2
Create Date: 2025-04-26 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '8a9d7b3e9f6a'
down_revision: Union[str, None] = '99bae3f947d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Make timestamp columns timezone-aware (TIMESTAMPTZ)."""
    # message_envelopes.created_at → TIMESTAMPTZ
    op.alter_column(
        "message_envelopes",
        "created_at",
        type_=sa.TIMESTAMP(timezone=True),
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )

    # reminders.reminder_time → TIMESTAMPTZ (nullable)
    op.alter_column(
        "reminders",
        "reminder_time",
        type_=sa.TIMESTAMP(timezone=True),
        postgresql_using="reminder_time AT TIME ZONE 'UTC'",
        existing_nullable=True,
    )

    # reminders.created_at → TIMESTAMPTZ
    op.alter_column(
        "reminders",
        "created_at",
        type_=sa.TIMESTAMP(timezone=True),
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )

    # reminders.updated_at → TIMESTAMPTZ
    op.alter_column(
        "reminders",
        "updated_at",
        type_=sa.TIMESTAMP(timezone=True),
        postgresql_using="updated_at AT TIME ZONE 'UTC'",
    )


def downgrade() -> None:
    """Revert timestamp columns back to naive (TIMESTAMP w/o TZ)."""
    # reminders.updated_at → naive
    op.alter_column(
        "reminders",
        "updated_at",
        type_=sa.DateTime(),
    )

    # reminders.created_at → naive
    op.alter_column(
        "reminders",
        "created_at",
        type_=sa.DateTime(),
    )

    # reminders.reminder_time → naive
    op.alter_column(
        "reminders",
        "reminder_time",
        type_=sa.DateTime(),
        existing_nullable=True,
    )

    # message_envelopes.created_at → naive
    op.alter_column(
        "message_envelopes",
        "created_at",
        type_=sa.DateTime(),
    ) 