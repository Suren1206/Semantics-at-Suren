from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

import requests


ZAPIER_WEBHOOK_URL = os.getenv("ZAPIER_WEBHOOK_URL", "").strip()


def _post_to_zapier(payload: Dict[str, Any], timeout: int = 8) -> bool:
    """
    Sends payload to Zapier webhook.
    Returns True if successful.
    """
    if not ZAPIER_WEBHOOK_URL:
        return False

    try:
        response = requests.post(
            ZAPIER_WEBHOOK_URL,
            json=payload,
            timeout=timeout,
        )
        return 200 <= response.status_code < 300
    except requests.RequestException:
        return False


def send_mutation_request_notification(
    *,
    mutation_id: str,
    request_type: str,
    created_by: str,
    query_text: str,
    original_text: str,
    proposed_text: str,
    header: Optional[str] = None,
    version_id: Optional[str] = None,
    app_name: str = "raGROW",
) -> bool:
    """
    Sends notification when Maker creates a mutation request.
    """
    payload = {
        "event_type": "maker_mutation_created",
        "app_name": app_name,
        "mutation_id": mutation_id,
        "request_type": request_type,
        "created_by": created_by,
        "query_text": query_text,
        "original_text": original_text,
        "proposed_text": proposed_text,
        "header": header or "",
        "version_id": version_id or "",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    return _post_to_zapier(payload)