"""
Alembic environment file – async-ready (SQLAlchemy ≥2.0)

Reads DATABASE_URL from the environment (or alembic.ini fallback),
imports Base from *db/db.py*, and supports both offline (DDL script
generation) and online (direct DB) modes.
"""

from __future__ import annotations

import os
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

# ---------------------------------------------------------------------
# 1. Logging
# ---------------------------------------------------------------------
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------
# 2. Model metadata
# ---------------------------------------------------------------------
from db.db import Base          # ← corrected path
target_metadata = Base.metadata

# ---------------------------------------------------------------------
# 3. Database URL helper
# ---------------------------------------------------------------------
def _database_url() -> str:
    """Determine the connection string Alembic should use."""
    try:
        from config import settings
        url = settings.DATABASE_URL or getattr(settings, "DATABASE_PUBLIC_URL", None)
    except Exception:
        url = (
            os.getenv("DATABASE_URL")
            or os.getenv("DATABASE_PUBLIC_URL")
            or config.get_main_option("sqlalchemy.url")
        )
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set and sqlalchemy.url missing from alembic.ini"
        )

    # add async driver only if not present already
    if "+asyncpg" not in url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url

# ---------------------------------------------------------------------
# 4. Offline migrations (generate SQL only)
# ---------------------------------------------------------------------
def run_migrations_offline() -> None:
    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

# ---------------------------------------------------------------------
# 5. Online migrations (run against DB) – async
# ---------------------------------------------------------------------
def _make_async_engine() -> AsyncEngine:
    return create_async_engine(_database_url(), poolclass=pool.NullPool, future=True)

async def run_migrations_online() -> None:
    engine = _make_async_engine()

    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: context.configure(
                connection=sync_conn,
                target_metadata=target_metadata,
            )
        )
        await conn.run_sync(lambda _: context.run_migrations())

    await engine.dispose()

# ---------------------------------------------------------------------
# 6. Entrypoint
# ---------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())