"""Optional Langfuse tracing without capturing user document content."""
import logging
import os
import sys
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def is_enabled():
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@contextmanager
def trace_span(name, *, metadata=None):
    """Create a best-effort Langfuse span; telemetry failures never fail RAG."""
    if not is_enabled():
        yield None
        return

    try:
        from langfuse import get_client

        manager = get_client().start_as_current_observation(
            as_type="span",
            name=name,
            metadata=metadata or {},
        )
        span = manager.__enter__()
    except Exception as exc:
        logger.warning("Langfuse tracing unavailable type=%s", type(exc).__name__)
        yield None
        return

    try:
        yield span
    except BaseException as exc:
        try:
            span.update(level="ERROR", status_message=type(exc).__name__)
        except Exception as telemetry_exc:
            logger.warning("Langfuse span update failed type=%s", type(telemetry_exc).__name__)
        try:
            manager.__exit__(*sys.exc_info())
        except Exception as telemetry_exc:
            logger.warning("Langfuse span close failed type=%s", type(telemetry_exc).__name__)
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception as exc:
            logger.warning("Langfuse span close failed type=%s", type(exc).__name__)
