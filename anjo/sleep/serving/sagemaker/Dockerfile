# Glimmer's SageMaker image: vLLM's own release image, with the shim in front on port 8080.
# Built from the context `python -m anjo.sleep.serving.sagemaker.deploy image --context <dir>` writes
# (this file and the shim's few modules, nothing else from the repo). The weights are not in the
# image: SageMaker copies the staged S3 prefix to /opt/ml/model when a copy starts.
ARG VLLM_VERSION=0.28.0
FROM vllm/vllm-openai:v${VLLM_VERSION}

# boto3 reads the adapters from S3; the release image does not carry it.
RUN python3 -m pip install --no-cache-dir boto3==1.43.111

COPY anjo /opt/anjo-sleep/anjo
ENV PYTHONPATH=/opt/anjo-sleep \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8080
# SageMaker runs the image with the argument "serve"; the shim takes it.
ENTRYPOINT ["python3", "-m", "anjo.sleep.serving.sagemaker.shim"]
