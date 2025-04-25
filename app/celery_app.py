"""Celery application instance shared across the backend.

Start a worker with:
    celery -A app.celery_app worker -Q save,reminder -l info --concurrency=2
"""

import os
from celery import Celery

BROKER_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("dory_backend", broker=BROKER_URL, backend=BROKER_URL)

# Global task settings
celery_app.conf.task_acks_late = True
celery_app.conf.task_reject_on_worker_lost = True
celery_app.conf.task_default_retry_delay = 30  # seconds

celery_app.conf.task_routes = {
    "app.workers.entity_resolver.handle_save": {"queue": "save"},
    "app.workers.reminder.handle": {"queue": "reminder"},
}
