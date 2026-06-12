
# Generate metrik MLFlow model serving
podman stop msml-mlflow-model
echo "waiting for 60s"
sleep 60
podman start msml-mlflow-model

run_inference_large() {
    local n_loop=${1:-1}
    local mode=${2:-bulk}
    for ((i=1; i<=n_loop; i++)); do
        python 7.Inference.py --input-file ./sample_data/sample_input_large.json --mode "$mode" &
    done
}
run_inference_failed_large() {
    local n_loop=${1:-1}
    local mode=${2:-bulk}
    for ((i=1; i<=n_loop; i++)); do
        python 7.Inference.py --input-file ./sample_data/sample_input_failed_large.json --mode "$mode" &
    done
}
run_inference() {
    local n_loop=${1:-1}
    local mode=${2:-bulk}
    for ((i=1; i<=n_loop; i++)); do
        python 7.Inference.py --input-file ./sample_data/sample_input.json --mode "$mode" &
    done
}
run_inference_failed() {
    local n_loop=${1:-1}
    local mode=${2:-bulk}
    for ((i=1; i<=n_loop; i++)); do
        python 7.Inference.py --input-file ./sample_data/sample_input_failed.json --mode "$mode" &
    done
}

# Generate all metrik
run_inference 1 batch
run_inference_failed 1 batch
run_inference 1 bulk
run_inference_failed 1 bulk

run_inference_large 1 batch
run_inference_failed_large 1 batch
run_inference_large 1000 bulk
run_inference_failed_large 1000 bulk


# Trigger Alert Request Throughput
for i in {1..3000}; do
  curl -s -o /dev/null -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    --data '' &
done
wait


# Trigger Alert system CPU usage 80%
bash -lc '
workers=$(nproc)
pids=""

cleanup() {
  kill $pids 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "Starting CPU burn with $workers workers for 90 seconds..."

for i in $(seq 1 "$workers"); do
  yes > /dev/null &
  pids="$pids $!"
done

echo "waiting for 60s"
sleep 60
cleanup

echo "Done."
'


# Trigger Alert System RAM usage 80%
python3 - <<'PY'
import time

TARGET_PERCENT = 82
HOLD_SECONDS = 60
CHUNK_MB = 256

def meminfo():
    data = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            data[key] = int(value.strip().split()[0])  # kB
    return data

def mem_percent():
    data = meminfo()
    total = data["MemTotal"]
    available = data["MemAvailable"]
    return (total - available) / total * 100

total_mb = meminfo()["MemTotal"] // 1024

# Batas maksimum dibuat dinamis.
# Untuk trigger alert, boleh sampai 90% dari total RAM.
MAX_ALLOC_MB = int(total_mb * 0.90)

chunks = []
allocated_mb = 0

print(f"Total RAM       : {total_mb} MB")
print(f"Initial RAM use : {mem_percent():.2f}%")
print(f"Target RAM use  : {TARGET_PERCENT}%")
print(f"Max allocation  : {MAX_ALLOC_MB} MB")
print("Allocating RAM...")

try:
    while mem_percent() < TARGET_PERCENT and allocated_mb < MAX_ALLOC_MB:
        block = bytearray(CHUNK_MB * 1024 * 1024)

        # Paksa page benar-benar disentuh agar tidak cuma virtual allocation.
        for i in range(0, len(block), 1024 * 1024):
            block[i] = 1

        chunks.append(block)
        allocated_mb += CHUNK_MB

        print(f"Allocated: {allocated_mb:5d} MB | RAM usage: {mem_percent():.2f}%")
        time.sleep(0.3)

    print(f"Holding memory for {HOLD_SECONDS} seconds...")
    time.sleep(HOLD_SECONDS)

finally:
    print("Releasing memory...")
    chunks.clear()
    time.sleep(3)
    print(f"Final RAM usage: {mem_percent():.2f}%")
PY
