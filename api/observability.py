
"""Request logging and Prometheus instrumentation."""

import json
import logging
import time
import uuid

from fastapi import Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "heart_api_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

logger = logging.getLogger("heart_api")


def configure_logging() -> None:
    """Emit application logs to stdout for container collection."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


async def observability_middleware(
    request: Request,
    call_next,
):
    """Measure requests and log metadata without medical inputs."""
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response

    finally:
        duration = time.perf_counter() - start

        # Route templates avoid unbounded metric label cardinality.
        route = request.scope.get("route")
        endpoint = route.path if route is not None else "unmatched"

        REQUEST_COUNT.labels(
            request.method,
            endpoint,
            str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            request.method,
            endpoint,
        ).observe(duration)

        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "endpoint": endpoint,
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 2),
                }
            )
        )


def metrics_response() -> Response:
    """Expose metrics in Prometheus format."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
