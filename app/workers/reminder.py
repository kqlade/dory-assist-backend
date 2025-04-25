"""Placeholder reminder task."""

from app.celery_app import celery_app


@celery_app.task(name="app.workers.reminder.handle", bind=True, max_retries=3)
def handle(self, envelope: dict, reminder: dict):  # noqa: ANN001
    print("[ReminderWorker] Scheduling reminder", envelope["envelope_id"], reminder)
