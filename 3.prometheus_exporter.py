import os
import time
import threading
import psutil
import requests
from typing import Any
from urllib.parse import urlsplit
from flask import Flask, Response, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

MODEL_URL = os.getenv("MODEL_URL", "http://127.0.0.1:5001/invocations")
EXPORTER_HOST = os.getenv("EXPORTER_HOST", "0.0.0.0")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", os.getenv("METRICS_PORT", "8000")))
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "15"))
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", str(500 * 1024)))

MODEL_STATUS_POLL_INTERVAL = float(os.getenv("MODEL_STATUS_POLL_INTERVAL", "5"))
MODEL_HEALTH_ENDPOINT = os.getenv("MODEL_HEALTH_ENDPOINT", "/ping")
MODEL_HEALTH_TIMEOUT_SECONDS = float(os.getenv("MODEL_HEALTH_TIMEOUT_SECONDS", "3"))

app = Flask(__name__)

REQUESTS_TOTAL = Counter(
    "ml_prediction_requests_total",
    "Total request inference yang diterima exporter.",
)

REQUEST_SUCCESS_TOTAL = Counter(
    "ml_prediction_success_total",
    "Total request inference yang berhasil diproses model.",
)

REQUEST_ERROR_TOTAL = Counter(
    "ml_prediction_error_total",
    "Total request inference yang gagal berdasarkan tipe error.",
    ["error_type"],
)

REQUEST_DURATION_SECONDS = Histogram(
    "ml_request_duration_seconds",
    "Durasi total request dari client ke exporter sampai response dikembalikan.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

MODEL_REQUEST_DURATION_SECONDS = Histogram(
    "ml_model_request_duration_seconds",
    "Durasi request dari exporter ke MLflow model serving.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

REQUEST_SIZE_BYTES = Histogram(
    "ml_request_size_bytes",
    "Ukuran body request inference dalam bytes.",
    buckets=(100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000),
)

REQUEST_SIZE_BUCKET_TOTAL = Counter(
    "ml_request_size_bucket_total",
    "Total request berdasarkan range ukuran body non-cumulative.",
    ["size_bucket"],
)

REQUEST_DURATION_BUCKET_TOTAL = Counter(
    "ml_request_duration_bucket_total",
    "Total request berdasarkan range durasi non-cumulative.",
    ["duration_bucket"],
)

REQUEST_BODY_SIZE_BYTES = Gauge(
    "ml_request_body_size_bytes",
    "Ukuran body request terakhir dalam bytes, termasuk request yang gagal validasi.",
)

REQUEST_BODY_SIZE_MIN_BYTES = Gauge(
    "ml_request_body_size_min_bytes",
    "Ukuran body request terkecil sejak exporter berjalan, termasuk request yang gagal validasi.",
)

REQUEST_BODY_SIZE_MAX_BYTES = Gauge(
    "ml_request_body_size_max_bytes",
    "Ukuran body request terbesar sejak exporter berjalan, termasuk request yang gagal validasi.",
)

REQUEST_BODY_SIZE_TOTAL_BYTES = Counter(
    "ml_request_body_size_total_bytes",
    "Total akumulasi ukuran body request dalam bytes, termasuk request yang gagal validasi.",
)

REQUEST_BODY_SIZE_OBSERVED_TOTAL = Counter(
    "ml_request_body_size_observed_total",
    "Total request yang ukuran body-nya berhasil dicatat, termasuk request yang gagal validasi.",
)

IN_FLIGHT_REQUESTS = Gauge(
    "ml_in_flight_requests",
    "Jumlah request inference yang sedang diproses exporter.",
)

MODEL_SERVING_UP = Gauge(
    "ml_model_serving_up",
    "Status MLflow model serving. 1 berarti model reachable, 0 berarti tidak reachable.",
)

CPU_USAGE_PERCENT = Gauge(
    "ml_cpu_usage_percent",
    "Persentase penggunaan CPU system/container environment.",
)

MEMORY_USAGE_PERCENT = Gauge(
    "ml_memory_usage_percent",
    "Persentase penggunaan RAM system/container environment.",
)

DISK_USAGE_PERCENT = Gauge(
    "ml_disk_usage_percent",
    "Persentase penggunaan disk pada filesystem root.",
)

_model_status_lock = threading.Lock()
_model_is_up = False

_body_size_min: int | None = None
_body_size_max: int | None = None
_body_size_lock = threading.Lock()


def get_model_base_url() -> str:
    parsed = urlsplit(MODEL_URL)
    return f"{parsed.scheme}://{parsed.netloc}"


def get_model_health_url() -> str:
    endpoint = MODEL_HEALTH_ENDPOINT

    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"

    return f"{get_model_base_url()}{endpoint}"


def get_size_bucket(size_bytes: int) -> str:
    if size_bytes <= 1_000:
        return "0-1KB"
    if size_bytes <= 5_000:
        return "1KB-5KB"
    if size_bytes <= 10_000:
        return "5KB-10KB"
    if size_bytes <= 50_000:
        return "10KB-50KB"
    if size_bytes <= 100_000:
        return "50KB-100KB"
    if size_bytes <= 250_000:
        return "100KB-250KB"
    if size_bytes <= 500_000:
        return "250KB-500KB"

    return ">500KB"


def get_duration_bucket(duration_seconds: float) -> str:
    if duration_seconds <= 0.05:
        return "0-50ms"
    if duration_seconds <= 0.10:
        return "50ms-100ms"
    if duration_seconds <= 0.25:
        return "100ms-250ms"
    if duration_seconds <= 0.50:
        return "250ms-500ms"
    if duration_seconds <= 1.00:
        return "500ms-1s"
    if duration_seconds <= 2.50:
        return "1s-2.5s"
    if duration_seconds <= 5.00:
        return "2.5s-5s"
    if duration_seconds <= 10.00:
        return "5s-10s"

    return ">10s"


def record_body_size(size_bytes: int) -> None:
    global _body_size_min, _body_size_max

    REQUEST_BODY_SIZE_BYTES.set(size_bytes)
    REQUEST_BODY_SIZE_TOTAL_BYTES.inc(size_bytes)
    REQUEST_BODY_SIZE_OBSERVED_TOTAL.inc()

    REQUEST_SIZE_BYTES.observe(size_bytes)
    REQUEST_SIZE_BUCKET_TOTAL.labels(size_bucket=get_size_bucket(size_bytes)).inc()

    with _body_size_lock:
        if _body_size_min is None or size_bytes < _body_size_min:
            _body_size_min = size_bytes
            REQUEST_BODY_SIZE_MIN_BYTES.set(size_bytes)

        if _body_size_max is None or size_bytes > _body_size_max:
            _body_size_max = size_bytes
            REQUEST_BODY_SIZE_MAX_BYTES.set(size_bytes)


def set_model_serving_status(is_up: bool) -> None:
    global _model_is_up

    with _model_status_lock:
        _model_is_up = is_up
        MODEL_SERVING_UP.set(1 if is_up else 0)


def is_model_marked_up() -> bool:
    with _model_status_lock:
        return _model_is_up


def poll_model_once() -> bool:
    try:
        response = requests.get(
            get_model_health_url(),
            timeout=MODEL_HEALTH_TIMEOUT_SECONDS,
        )

        # Endpoint /ping MLflow model serving dianggap sehat hanya jika 2xx.
        return 200 <= response.status_code < 300

    except requests.exceptions.RequestException:
        return False


def monitor_model_serving_status() -> None:
    while True:
        is_up = poll_model_once()
        set_model_serving_status(is_up)
        time.sleep(MODEL_STATUS_POLL_INTERVAL)


def get_error_type_from_exception(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"

    if isinstance(exc, requests.exceptions.ConnectionError):
        return "connection"

    return type(exc).__name__.lower()


def update_system_metrics() -> None:
    while True:
        CPU_USAGE_PERCENT.set(psutil.cpu_percent(interval=None))
        MEMORY_USAGE_PERCENT.set(psutil.virtual_memory().percent)
        DISK_USAGE_PERCENT.set(psutil.disk_usage("/").percent)
        time.sleep(5)


@app.get("/")
def index() -> Any:
    return jsonify(
        {
            "service": "heart-disease-prometheus-exporter",
            "model_url": MODEL_URL,
            "model_base_url": get_model_base_url(),
            "model_health_url": get_model_health_url(),
            "model_serving_up": is_model_marked_up(),
            "max_request_bytes": MAX_REQUEST_BYTES,
            "model_status_poll_interval": MODEL_STATUS_POLL_INTERVAL,
            "model_health_timeout_seconds": MODEL_HEALTH_TIMEOUT_SECONDS,
            "endpoints": {
                "predict": "/predict",
                "metrics": "/metrics",
                "health": "/health",
            },
        }
    )


@app.get("/health")
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "model_url": MODEL_URL,
            "model_base_url": get_model_base_url(),
            "model_health_url": get_model_health_url(),
            "model_serving_up": is_model_marked_up(),
            "max_request_bytes": MAX_REQUEST_BYTES,
            "model_status_poll_interval": MODEL_STATUS_POLL_INTERVAL,
            "model_health_timeout_seconds": MODEL_HEALTH_TIMEOUT_SECONDS,
        }
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict() -> Response:
    REQUESTS_TOTAL.inc()
    IN_FLIGHT_REQUESTS.inc()

    request_start = time.perf_counter()

    try:
        content_length = request.content_length

        if content_length is not None:
            request_size = int(content_length)
        else:
            raw_body_preview = request.get_data(cache=True)
            request_size = len(raw_body_preview)

        # Body size dicatat sebelum validasi.
        # Jadi request gagal 400/413 tetap masuk metrik body size.
        record_body_size(request_size)

        if request_size > MAX_REQUEST_BYTES:
            REQUEST_ERROR_TOTAL.labels(error_type="request_too_large").inc()

            return jsonify(
                {
                    "error": "request_too_large",
                    "message": f"Request body melebihi batas {MAX_REQUEST_BYTES} bytes.",
                    "max_request_bytes": MAX_REQUEST_BYTES,
                    "request_bytes": request_size,
                }
            ), 413

        raw_body = request.get_data(cache=False)

        if len(raw_body) == 0:
            REQUEST_ERROR_TOTAL.labels(error_type="empty_body").inc()

            return jsonify(
                {
                    "error": "empty_body",
                    "message": "Body request kosong.",
                    "request_bytes": request_size,
                }
            ), 400

        content_type = request.headers.get("Content-Type", "application/json")

        forward_headers = {
            "Content-Type": content_type,
            "Accept": "application/json",
        }

        model_start = time.perf_counter()

        model_response = requests.post(
            MODEL_URL,
            data=raw_body,
            headers=forward_headers,
            timeout=TIMEOUT_SECONDS,
        )

        model_elapsed = time.perf_counter() - model_start
        MODEL_REQUEST_DURATION_SECONDS.observe(model_elapsed)

        response_body = model_response.content
        response_content_type = model_response.headers.get(
            "Content-Type",
            "application/json",
        )

        # Kalau MLflow memberi response HTTP apa pun, model server reachable.
        # 400/500 dari MLflow bukan connection down.
        set_model_serving_status(True)

        if 200 <= model_response.status_code < 300:
            REQUEST_SUCCESS_TOTAL.inc()
        else:
            REQUEST_ERROR_TOTAL.labels(
                error_type=f"http_{model_response.status_code}"
            ).inc()

        return Response(
            response=response_body,
            status=model_response.status_code,
            content_type=response_content_type,
        )

    except requests.exceptions.Timeout as exc:
        set_model_serving_status(False)
        REQUEST_ERROR_TOTAL.labels(error_type="timeout").inc()

        return jsonify(
            {
                "error": "timeout",
                "message": str(exc),
            }
        ), 502

    except requests.exceptions.ConnectionError as exc:
        set_model_serving_status(False)
        REQUEST_ERROR_TOTAL.labels(error_type="connection").inc()

        return jsonify(
            {
                "error": "connection",
                "message": str(exc),
            }
        ), 502

    except requests.exceptions.RequestException as exc:
        set_model_serving_status(False)

        error_type = get_error_type_from_exception(exc)
        REQUEST_ERROR_TOTAL.labels(error_type=error_type).inc()

        return jsonify(
            {
                "error": error_type,
                "message": str(exc),
            }
        ), 502

    except Exception as exc:
        # Error internal exporter tidak otomatis berarti model down.
        error_type = type(exc).__name__.lower()
        REQUEST_ERROR_TOTAL.labels(error_type=error_type).inc()

        return jsonify(
            {
                "error": error_type,
                "message": str(exc),
            }
        ), 500

    finally:
        request_elapsed = time.perf_counter() - request_start
        REQUEST_DURATION_SECONDS.observe(request_elapsed)
        REQUEST_DURATION_BUCKET_TOTAL.labels(
            duration_bucket=get_duration_bucket(request_elapsed)
        ).inc()
        IN_FLIGHT_REQUESTS.dec()


def main() -> None:
    threading.Thread(target=update_system_metrics, daemon=True).start()
    threading.Thread(target=monitor_model_serving_status, daemon=True).start()

    print(
        f"Exporter Flask berjalan di http://{EXPORTER_HOST}:{EXPORTER_PORT}", flush=True
    )
    print(f"Target MLflow model: {MODEL_URL}", flush=True)
    print(f"Model base URL: {get_model_base_url()}", flush=True)
    print(f"Model health URL: {get_model_health_url()}", flush=True)
    print(
        f"Model status poll interval: {MODEL_STATUS_POLL_INTERVAL} seconds", flush=True
    )
    print(f"Model health timeout: {MODEL_HEALTH_TIMEOUT_SECONDS} seconds", flush=True)
    print(f"Max request size: {MAX_REQUEST_BYTES} bytes", flush=True)

    app.run(
        host=EXPORTER_HOST,
        port=EXPORTER_PORT,
        threaded=True,
    )


if __name__ == "__main__":
    main()
