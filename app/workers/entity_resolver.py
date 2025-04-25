"""Entity resolver celery task."""

import json
from typing import List, Dict

from app.celery_app import celery_app


@celery_app.task(name="app.workers.entity_resolver.handle_save", bind=True, max_retries=3)
def handle_save(self, envelope: Dict, entities: List[Dict]):  # noqa: ANN401
    print("[EntityResolver] Handling envelope", envelope["envelope_id"])
    print("Entities:", json.dumps(entities, indent=2))
    # TODO: resolution logic (Google Places, Neo4j merge, etc.)
