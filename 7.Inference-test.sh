podman run --rm -p 5001:8080 --name mlflow-model docker.io/akhsaul/heart-disease-svc:latest &

python 7.Inference.py --url "http://127.0.0.1:5001/invocations" \
    --input-file ./sample_data/sample_input_large.json \
    --mode batch
