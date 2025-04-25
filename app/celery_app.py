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

# Beat schedule: dispatch due time-based reminders every minute
celery_app.conf.beat_schedule = {
    "dispatch-due-reminders": {
        "task": "app.workers.reminder.dispatch_due",
        "schedule": 60.0,
    }
}

# --- Ensure tasks are registered ---
import app.workers.entity_resolver
import app.workers.reminder
import app.workers.parser

