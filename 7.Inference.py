import argparse
import json
import sys
import time
import requests
from pathlib import Path
from typing import Any, Dict, List


def expand_dataframe_split(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    split = payload.get("dataframe_split", {})
    columns = split.get("columns", [])
    rows = split.get("data", [])

    if not columns or not rows:
        return []

    return [
        {
            "dataframe_split": {
                "columns": columns,
                "data": [row],
            }
        }
        for row in rows
    ]


def expand_dataframe_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = payload.get("dataframe_records", [])

    if not records:
        return []

    return [{"dataframe_records": [record]} for record in records]


def expand_instances(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    instances = payload.get("instances", [])

    if not instances:
        return []

    return [{"instances": [instance]} for instance in instances]


def expand_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ubah payload batch menjadi banyak payload satu data.
    Dipakai untuk mode batch agar 1 request = 1 data.
    """
    if "dataframe_split" in payload:
        return expand_dataframe_split(payload)

    if "dataframe_records" in payload:
        return expand_dataframe_records(payload)

    if "instances" in payload:
        return expand_instances(payload)

    return [payload]


def load_batch_payloads(path: str) -> List[Dict[str, Any]]:
    """
    Membaca file input dan memecah seluruh data menjadi payload satu per satu.
    Exporter tidak membaca file. File hanya dibaca oleh inference.py.
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path.resolve()}")

    data = json.loads(input_path.read_text(encoding="utf-8"))

    payloads: List[Dict[str, Any]] = []

    if isinstance(data, dict) and "payloads" in data:
        for item in data["payloads"]:
            if isinstance(item, dict):
                payloads.extend(expand_payload(item))

    elif isinstance(data, dict):
        payloads.extend(expand_payload(data))

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                payloads.extend(expand_payload(item))

    else:
        raise ValueError(
            "Format input harus dict, list[dict], atau {'payloads': [...]}."
        )

    if not payloads:
        raise ValueError("Tidak ada payload valid yang bisa dikirim.")

    return payloads


def apply_limit(payloads: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """
    limit -1 = semua data.
    limit >= 0 = ambil data sebanyak limit.
    """
    if limit >= 0:
        return payloads[:limit]

    return payloads


def build_bulk_payload(batch_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Mode bulk:
    - input awal sudah dipecah menjadi payload satu data
    - lalu digabung lagi menjadi 1 payload besar sesuai limit
    - hasilnya dikirim dalam 1 request
    """
    if not batch_payloads:
        raise ValueError("Tidak ada payload untuk mode bulk.")

    first_payload = batch_payloads[0]

    if "dataframe_split" in first_payload:
        columns = first_payload["dataframe_split"]["columns"]
        rows = []

        for payload in batch_payloads:
            if "dataframe_split" not in payload:
                raise ValueError("Mode bulk gagal: format payload tercampur.")

            current_columns = payload["dataframe_split"]["columns"]

            if current_columns != columns:
                raise ValueError("Mode bulk gagal: columns dataframe_split tidak sama.")

            rows.extend(payload["dataframe_split"]["data"])

        return {
            "dataframe_split": {
                "columns": columns,
                "data": rows,
            }
        }

    if "dataframe_records" in first_payload:
        records = []

        for payload in batch_payloads:
            if "dataframe_records" not in payload:
                raise ValueError("Mode bulk gagal: format payload tercampur.")

            records.extend(payload["dataframe_records"])

        return {
            "dataframe_records": records,
        }

    if "instances" in first_payload:
        instances = []

        for payload in batch_payloads:
            if "instances" not in payload:
                raise ValueError("Mode bulk gagal: format payload tercampur.")

            instances.extend(payload["instances"])

        return {
            "instances": instances,
        }

    if len(batch_payloads) == 1:
        return batch_payloads[0]

    raise ValueError(
        "Mode bulk hanya mendukung dataframe_split, dataframe_records, instances, "
        "atau payload tunggal."
    )


def send_payload(
    url: str, payload: Dict[str, Any], timeout: float
) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    return requests.post(
        url,
        data=json.dumps(payload),
        headers=headers,
        timeout=timeout,
    )


def print_response_body(response: requests.Response) -> None:
    try:
        parsed = response.json()
        print(json.dumps(parsed, ensure_ascii=False))
    except ValueError:
        print(response.text)


def run_batch_mode(
    url: str,
    payloads: List[Dict[str, Any]],
    repeat: int,
    delay: float,
    timeout: float,
) -> None:
    """
    Mode batch:
    - default
    - 1 data = 1 request
    - jika ada request gagal 400/413/500, script tetap lanjut
    """
    total_round = repeat + 1
    request_number = 0
    success_count = 0
    failed_count = 0

    for round_index in range(1, total_round + 1):
        for payload_index, payload in enumerate(payloads, start=1):
            request_number += 1
            started = time.perf_counter()

            try:
                response = send_payload(
                    url=url,
                    payload=payload,
                    timeout=timeout,
                )

                elapsed = time.perf_counter() - started
                is_success = 200 <= response.status_code < 300

                if is_success:
                    success_count += 1
                    result = "SUCCESS"
                else:
                    failed_count += 1
                    result = "FAILED"

                print(
                    f"Request #{request_number} | "
                    f"mode=batch | "
                    f"round={round_index}/{total_round} | "
                    f"data={payload_index}/{len(payloads)} | "
                    f"status={response.status_code} | "
                    f"result={result} | "
                    f"duration={elapsed:.4f}s"
                )

                print_response_body(response)
                print("-" * 80)

            except requests.exceptions.RequestException as exc:
                elapsed = time.perf_counter() - started
                failed_count += 1

                print(
                    f"Request #{request_number} | "
                    f"mode=batch | "
                    f"round={round_index}/{total_round} | "
                    f"data={payload_index}/{len(payloads)} | "
                    f"status=REQUEST_ERROR | "
                    f"result=FAILED | "
                    f"duration={elapsed:.4f}s"
                )

                print(f"{type(exc).__name__}: {exc}")
                print("-" * 80)

            if delay > 0:
                time.sleep(delay)

    print("Summary")
    print("-" * 80)
    print(f"Total request          : {request_number}")
    print(f"Total request success  : {success_count}")
    print(f"Total request failed   : {failed_count}")


def run_bulk_mode(
    url: str,
    payload: Dict[str, Any],
    repeat: int,
    delay: float,
    timeout: float,
    total_data: int,
) -> None:
    """
    Mode bulk:
    - semua data sesuai limit dikirim sekaligus dalam 1 request
    - jika request gagal 400/413/500, script tetap lanjut untuk repeat berikutnya
    """
    total_round = repeat + 1
    success_count = 0
    failed_count = 0

    for round_index in range(1, total_round + 1):
        started = time.perf_counter()

        try:
            response = send_payload(
                url=url,
                payload=payload,
                timeout=timeout,
            )

            elapsed = time.perf_counter() - started
            is_success = 200 <= response.status_code < 300

            if is_success:
                success_count += 1
                result = "SUCCESS"
            else:
                failed_count += 1
                result = "FAILED"

            print(
                f"Request #{round_index} | "
                f"mode=bulk | "
                f"round={round_index}/{total_round} | "
                f"data={total_data} | "
                f"status={response.status_code} | "
                f"result={result} | "
                f"duration={elapsed:.4f}s"
            )

            print_response_body(response)
            print("-" * 80)

        except requests.exceptions.RequestException as exc:
            elapsed = time.perf_counter() - started
            failed_count += 1

            print(
                f"Request #{round_index} | "
                f"mode=bulk | "
                f"round={round_index}/{total_round} | "
                f"data={total_data} | "
                f"status=REQUEST_ERROR | "
                f"result=FAILED | "
                f"duration={elapsed:.4f}s"
            )

            print(f"{type(exc).__name__}: {exc}")
            print("-" * 80)

        if delay > 0:
            time.sleep(delay)

    print("Summary")
    print("-" * 80)
    print(f"Total request          : {total_round}")
    print(f"Total request success  : {success_count}")
    print(f"Total request failed   : {failed_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kirim request inference ke exporter /predict atau MLflow /invocations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh penggunaan:\n"
            "  python 7.Inference.py --url http://127.0.0.1:8000/predict\n"
            "  python 7.Inference.py --url http://127.0.0.1:5001/invocations --mode bulk --limit 10\n"
            "  python 7.Inference.py --url http://127.0.0.1:8000/predict --repeat 5 --delay 1\n"
        ),
    )

    parser.add_argument(
        "--url",
        help=(
            "Endpoint inference. "
            "Default exporter: http://127.0.0.1:8000/predict. "
            "Untuk MLflow langsung: http://127.0.0.1:5001/invocations."
        ),
    )

    parser.add_argument(
        "--input-file",
        default="sample_data/sample_input.json",
        help="Path file JSON input inference. Default: `./sample_data/sample_input.json`",
    )

    parser.add_argument(
        "--mode",
        choices=["batch", "bulk"],
        default="batch",
        help=(
            "batch = kirim satu per satu data. "
            "bulk = kirim sekaligus dalam 1 request sesuai limit."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Jumlah data yang dikirim. Default -1 berarti baca seluruh data.",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Jumlah pengulangan tambahan. Default 0 berarti hanya kirim sekali.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay antar request dalam detik.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Timeout request dalam detik.",
    )

    args = parser.parse_args()

    if args.url is None or not args.url.strip():
        parser.print_help(sys.stderr)
        parser.exit(2, "\nERROR: --url wajib diisi. Lihat contoh penggunaan di atas.\n")

    batch_payloads = load_batch_payloads(args.input_file)
    selected_payloads = apply_limit(batch_payloads, args.limit)

    if not selected_payloads:
        raise ValueError("Tidak ada payload yang dikirim setelah limit diterapkan.")

    print(f"URL         : {args.url}")
    print(f"Input file  : {Path(args.input_file).resolve()}")
    print(f"Mode        : {args.mode}")
    print(f"Total data  : {len(selected_payloads)}")
    print(f"Limit       : {args.limit}")
    print(f"Repeat      : {args.repeat}")
    print(f"Total round : {args.repeat + 1}")
    print(f"Delay       : {args.delay}s")
    print("-" * 80)

    if args.mode == "batch":
        run_batch_mode(
            url=args.url,
            payloads=selected_payloads,
            repeat=args.repeat,
            delay=args.delay,
            timeout=args.timeout,
        )

    elif args.mode == "bulk":
        bulk_payload = build_bulk_payload(selected_payloads)

        run_bulk_mode(
            url=args.url,
            payload=bulk_payload,
            repeat=args.repeat,
            delay=args.delay,
            timeout=args.timeout,
            total_data=len(selected_payloads),
        )


if __name__ == "__main__":
    main()
