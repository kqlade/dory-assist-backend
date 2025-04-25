"""Placeholder reminder task."""

from app.celery_app import celery_app


@celery_app.task(name="app.workers.reminder.handle", bind=True, max_retries=3)
def handle(self, envelope: dict, reply: dict):  # noqa: ANN001
    print("[ReminderWorker] TODO implement", envelope["envelope_id"], reply.get("intent"))
