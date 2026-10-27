"""The operator's command line for Glimmer on SageMaker. Every subcommand but ``image`` calls AWS and
some start paid GPU time, so only Kevin runs them, with the operator's own credentials, never a test.

    python -m anjo.sleep.serving.sagemaker.deploy stage-weights --scratch <dir>   # HF -> S3, once
    python -m anjo.sleep.serving.sagemaker.deploy image --context <dir>           # prints build and push
    python -m anjo.sleep.serving.sagemaker.deploy up --image-uri <uri>            # model .. scaling, then asleep
    python -m anjo.sleep.serving.sagemaker.deploy upload-adapter --folder <dir> --name p-<hash>-n<night>
    python -m anjo.sleep.serving.sagemaker.deploy stop                            # copy count 0 now
    python -m anjo.sleep.serving.sagemaker.deploy status

It reads ``ANJO_SLEEP_GLIMMER_REVISION`` (the pin), ``ANJO_SLEEP_S3_BUCKET`` and, if set,
``ANJO_SLEEP_TOOL_CALL_PARSER`` and ``ANJO_SLEEP_VLLM_VERSION``. It names the two IAM roles and
never creates one (deploy/aws holds them for the console). The account id comes from STS (or
``--account-id``) and is never printed. Every name starts with ``anjo-sleep-``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from anjo.sleep.serving import config
from anjo.sleep.serving.sagemaker import settings
from anjo.sleep.serving.sagemaker.invoke import CONTROL_ATTEMPTS, CONTROL_READ_TIMEOUT_S, make_client
from anjo.sleep.serving.sagemaker.serving import SageMakerServing

REPO_ROOT: Final = Path(__file__).resolve().parents[4]
CONTAINER_FILES: Final = (  # all the image carries from the repo: the shim and what it imports
    "anjo/sleep/__init__.py", "anjo/sleep/serving/__init__.py", "anjo/sleep/serving/config.py",
    "anjo/sleep/serving/sagemaker/__init__.py", "anjo/sleep/serving/sagemaker/settings.py",
    "anjo/sleep/serving/sagemaker/shim.py",
)
WAIT_TOTAL_S: Final = 3600.0
WAIT_POLL_S: Final = 30.0
UPLOAD_READ_TIMEOUT_S: Final = 300.0
SKIPPED_WEIGHTS: Final = re.compile(r"(^|/)original/|\.pth$|^\.gitattributes$")  # not what vLLM loads
_IMAGE = re.compile(r"[0-9]{12}\.dkr\.ecr\.us-east-1\.amazonaws\.com/" + re.escape(settings.REPOSITORY)
                    + r":([A-Za-z0-9][A-Za-z0-9_.-]{0,40})")


class DeployError(RuntimeError):
    """A step that can't go on. The message names the step, never a credential or the account."""


@dataclass
class Aws:
    """The operator's clients, made on first use (tests pass stand-ins)."""

    made: dict[str, Any] = field(default_factory=dict)

    def get(self, service: str) -> Any:
        if service not in self.made:
            read_s = UPLOAD_READ_TIMEOUT_S if service == "s3" else CONTROL_READ_TIMEOUT_S
            self.made[service] = make_client(service, read_s=read_s, attempts=CONTROL_ATTEMPTS)
        return self.made[service]


def _missing(error: ClientError) -> bool:
    details = error.response.get("Error", {})
    return details.get("Code") in ("ValidationException", "ResourceNotFound", "404", "NoSuchKey") \
        or error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404


def _exists(describe: Callable[[], Any]) -> bool:
    try:
        describe()
    except ClientError as error:
        if _missing(error):
            return False
        raise
    return True


def weights_prefix(revision: str) -> str:
    return f"{settings.WEIGHTS_PREFIX}{config.MODEL}/{revision}/"


# ── weights ─────────────────────────────────────────────────────────────────


def stage_weights(aws: Aws, serve: config.ServeConfig, bucket: str, scratch: Path) -> dict[str, int]:
    """Copy the pinned revision from Hugging Face to the bucket one file at a time (so the scratch
    disk holds one shard), skipping what is there at the same size: a rerun resumes."""
    from huggingface_hub import HfApi, hf_hub_download

    revision, s3 = serve.pinned_revision(), aws.get("s3")
    info = HfApi().model_info(config.MODEL, revision=revision, files_metadata=True)
    counts = {"copied": 0, "present": 0, "skipped": 0}
    for sibling in sorted(info.siblings or [], key=lambda s: str(s.rfilename)):
        name, size = str(sibling.rfilename), sibling.size
        if SKIPPED_WEIGHTS.search(name):
            counts["skipped"] += 1
            continue
        key = weights_prefix(revision) + name
        try:
            if s3.head_object(Bucket=bucket, Key=key).get("ContentLength") == size:
                counts["present"] += 1
                continue
        except ClientError as error:
            if not _missing(error):
                raise
        local = Path(hf_hub_download(config.MODEL, name, revision=revision, local_dir=scratch))
        s3.upload_file(str(local), bucket, key)
        local.unlink()
        counts["copied"] += 1
    return counts


# ── the image ───────────────────────────────────────────────────────────────


def image_tag(serve: config.ServeConfig, root: Path = REPO_ROOT) -> str:
    """vLLM's release, the pin, and a digest of the shim's files: new code is a new tag."""
    digest = hashlib.sha256()
    for name in CONTAINER_FILES + ("anjo/sleep/serving/sagemaker/Dockerfile",):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return f"vllm{serve.vllm_version}-{serve.pinned_revision()[:12]}-{digest.hexdigest()[:8]}"


def write_context(context: Path, root: Path = REPO_ROOT) -> None:
    """The build context: the Dockerfile and ``CONTAINER_FILES``, and nothing else (no .env, no data)."""
    if context.exists() and any(context.iterdir()):
        raise DeployError(f"{context} is not empty")
    for name in CONTAINER_FILES:
        target = context / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    shutil.copyfile(root / "anjo/sleep/serving/sagemaker/Dockerfile", context / "Dockerfile")


def image_commands(context: Path, serve: config.ServeConfig, tag: str) -> list[str]:
    registry = f"$AWS_ACCOUNT_ID.dkr.ecr.{settings.REGION}.amazonaws.com"
    uri = f"{registry}/{settings.REPOSITORY}:{tag}"
    return [
        f"aws ecr create-repository --repository-name {settings.REPOSITORY} --region {settings.REGION}  # once",
        f"aws ecr get-login-password --region {settings.REGION} | docker login --username AWS --password-stdin {registry}",
        f"docker buildx build --platform linux/amd64 --build-arg VLLM_VERSION={serve.vllm_version} -t {uri} --push {context}",
        f"python -m anjo.sleep.serving.sagemaker.deploy up --image-uri {uri}",
    ]


# ── the endpoint ────────────────────────────────────────────────────────────


def model_name(image_uri: str) -> str:
    found = _IMAGE.fullmatch(image_uri)
    if found is None:
        raise DeployError(f"the image is <account>.dkr.ecr.{settings.REGION}.amazonaws.com/{settings.REPOSITORY}:<tag>")
    return (settings.MODEL_PREFIX + re.sub(r"[^A-Za-z0-9-]", "-", found.group(1)))[:63].rstrip("-")


def _model_env(serve: config.ServeConfig, bucket: str) -> dict[str, str]:
    env = {config.REVISION_ENV: serve.pinned_revision(), config.VLLM_VERSION_ENV: serve.vllm_version,
           settings.BUCKET_ENV: bucket}
    if serve.tool_parser is not None:
        env[config.TOOL_PARSER_ENV] = serve.tool_parser
    return env


def _wait(describe: Callable[[], Mapping[str, Any]], status_key: str, *, clock: Callable[[], float],
          sleep: Callable[[float], None]) -> float:
    started = clock()
    while True:
        status = describe().get(status_key)
        if status == "InService":
            return clock() - started
        if status == "Failed" or clock() - started > WAIT_TOTAL_S:
            raise DeployError(f"{status_key} is {status} after {clock() - started:.0f} s")
        sleep(WAIT_POLL_S)


def up(aws: Aws, serve: config.ServeConfig, bucket: str, image_uri: str, account_id: str | None = None, *,
       clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Model, endpoint config, endpoint, inference component and the idle net, each made if missing
    (a new image gives a new model, which the component is moved to). It waits for the first copy,
    records how long that took, and sets the copy count to 0 before it returns."""
    sm = aws.get("sagemaker")
    account = account_id or str(aws.get("sts").get_caller_identity()["Account"])
    role = f"arn:aws:iam::{account}:role/{settings.EXECUTION_ROLE}"
    model = model_name(image_uri)
    if not _exists(lambda: sm.describe_model(ModelName=model)):
        sm.create_model(ModelName=model, ExecutionRoleArn=role, PrimaryContainer={
            "Image": image_uri, "Environment": _model_env(serve, bucket),
            "ModelDataSource": {"S3DataSource": {
                "S3Uri": f"s3://{bucket}/{weights_prefix(serve.pinned_revision())}",
                "S3DataType": "S3Prefix", "CompressionType": "None"}},
        })
    if not _exists(lambda: sm.describe_endpoint_config(EndpointConfigName=settings.ENDPOINT_CONFIG)):
        sm.create_endpoint_config(EndpointConfigName=settings.ENDPOINT_CONFIG, ExecutionRoleArn=role, ProductionVariants=[{
            "VariantName": settings.VARIANT, "InstanceType": settings.INSTANCE_TYPE, "InitialInstanceCount": 1,
            "ManagedInstanceScaling": {"Status": "ENABLED", "MinInstanceCount": 0, "MaxInstanceCount": 1},
            "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
        }])
    if not _exists(lambda: sm.describe_endpoint(EndpointName=settings.ENDPOINT)):
        sm.create_endpoint(EndpointName=settings.ENDPOINT, EndpointConfigName=settings.ENDPOINT_CONFIG)
    endpoint_s = _wait(lambda: sm.describe_endpoint(EndpointName=settings.ENDPOINT), "EndpointStatus",
                       clock=clock, sleep=sleep)
    spec = {"ModelName": model,
            "StartupParameters": {"ModelDataDownloadTimeoutInSeconds": settings.MODEL_DOWNLOAD_TIMEOUT_S,
                                  "ContainerStartupHealthCheckTimeoutInSeconds": settings.STARTUP_HEALTH_TIMEOUT_S},
            "ComputeResourceRequirements": {"NumberOfAcceleratorDevicesRequired": 1,
                                            "MinMemoryRequiredInMb": settings.MIN_MEMORY_MB}}
    describe_ic = lambda: sm.describe_inference_component(InferenceComponentName=settings.COMPONENT)  # noqa: E731
    if _exists(describe_ic):
        sm.update_inference_component(InferenceComponentName=settings.COMPONENT, Specification=spec,
                                      RuntimeConfig={"CopyCount": 1})
    else:
        sm.create_inference_component(InferenceComponentName=settings.COMPONENT, EndpointName=settings.ENDPOINT,
                                      VariantName=settings.VARIANT, Specification=spec, RuntimeConfig={"CopyCount": 1})
    try:
        copy_s = _wait(describe_ic, "InferenceComponentStatus", clock=clock, sleep=sleep)
        idle_net(aws)
    finally:  # paid while a copy is up: whatever happened, it goes back to sleep
        SageMakerServing(bucket=bucket, control=sm, clock=clock, sleep=sleep).sleep_now()
    return {"model": model, "endpoint": settings.ENDPOINT, "component": settings.COMPONENT,
            "endpoint_seconds": round(endpoint_s), "first_copy_seconds": round(copy_s), "copies": 0}


def idle_net(aws: Aws) -> None:
    """The AWS-side net: when a copy has been up ``IDLE_MINUTES`` straight with no invocation, the
    alarm's step policy sets the copy count to exactly 0. It reads 1 only while a copy runs (its
    memory metric exists) and nobody called it that minute, so a sleeping component never breaches
    and a wake is not undone."""
    aas, cloudwatch = aws.get("application-autoscaling"), aws.get("cloudwatch")
    target = {"ServiceNamespace": "sagemaker", "ResourceId": settings.scalable_resource(),
              "ScalableDimension": settings.SCALING_DIMENSION}
    aas.register_scalable_target(**target, MinCapacity=0, MaxCapacity=1)
    policy = aas.put_scaling_policy(PolicyName=settings.IDLE_POLICY, PolicyType="StepScaling", **target,
                                    StepScalingPolicyConfiguration={
                                        "AdjustmentType": "ExactCapacity", "Cooldown": 300,
                                        "StepAdjustments": [{"MetricIntervalLowerBound": 0.0, "ScalingAdjustment": 0}]})
    component = [{"Name": "InferenceComponentName", "Value": settings.COMPONENT}]

    def stat(key: str, namespace: str, metric: str, how: str) -> dict[str, Any]:
        return {"Id": key, "ReturnData": False, "MetricStat": {
            "Metric": {"Namespace": namespace, "MetricName": metric, "Dimensions": component},
            "Period": 60, "Stat": how}}

    cloudwatch.put_metric_alarm(
        AlarmName=settings.IDLE_ALARM, AlarmDescription="Glimmer's copy is up and nobody called it: back to 0",
        Metrics=[stat("calls", "AWS/SageMaker", "Invocations", "Sum"),
                 stat("up", "/aws/sagemaker/InferenceComponents", "MemoryUtilizationNormalized", "Maximum"),
                 {"Id": "idle", "ReturnData": True, "Label": "up and idle",
                  "Expression": "IF(FILL(up, 0) > 0 AND FILL(calls, 0) == 0, 1, 0)"}],
        EvaluationPeriods=settings.IDLE_MINUTES, DatapointsToAlarm=settings.IDLE_MINUTES, Threshold=1.0,
        ComparisonOperator="GreaterThanOrEqualToThreshold", TreatMissingData="notBreaching",
        AlarmActions=[policy["PolicyARN"]])


def status(aws: Aws) -> dict[str, Any]:
    sm = aws.get("sagemaker")
    endpoint = sm.describe_endpoint(EndpointName=settings.ENDPOINT)
    variants = endpoint.get("ProductionVariants") or [{}]
    component = sm.describe_inference_component(InferenceComponentName=settings.COMPONENT)
    runtime = component.get("RuntimeConfig", {})
    alarms = aws.get("cloudwatch").describe_alarms(AlarmNames=[settings.IDLE_ALARM]).get("MetricAlarms") or [{}]
    return {"endpoint": endpoint.get("EndpointStatus"), "instances": variants[0].get("CurrentInstanceCount"),
            "component": component.get("InferenceComponentStatus"),
            "desired_copies": runtime.get("DesiredCopyCount"), "current_copies": runtime.get("CurrentCopyCount"),
            "idle_alarm": alarms[0].get("StateValue"), "usd_per_hour_while_up": settings.USD_PER_HOUR}


# ── the command line ────────────────────────────────────────────────────────


def _bucket(env: Mapping[str, str]) -> str:
    name = env.get(settings.BUCKET_ENV, "").strip()
    if not name:
        raise DeployError(f"set {settings.BUCKET_ENV}")
    return settings.check_bucket(name)


def main(argv: list[str] | None = None, *, aws: Aws | None = None) -> int:
    parser = argparse.ArgumentParser(description="Glimmer on SageMaker: the operator's steps (paid).")
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("stage-weights").add_argument("--scratch", type=Path, required=True)
    steps.add_parser("image").add_argument("--context", type=Path, required=True)
    started = steps.add_parser("up")
    started.add_argument("--image-uri", required=True)
    started.add_argument("--account-id", help="instead of asking STS")
    adapter = steps.add_parser("upload-adapter")
    adapter.add_argument("--folder", type=Path, required=True)
    adapter.add_argument("--name", required=True)
    steps.add_parser("stop")
    steps.add_parser("status")
    args = parser.parse_args(argv)
    aws = aws or Aws()
    try:
        serve = config.ServeConfig.from_env()
        if args.step == "image":
            write_context(args.context)
            print("\n".join(image_commands(args.context, serve, image_tag(serve))))
            return 0
        if args.step == "status":
            done: Mapping[str, Any] = status(aws)
        elif args.step == "stop":
            SageMakerServing(bucket=None, control=aws.get("sagemaker")).sleep_now()
            done = status(aws)
        elif args.step == "upload-adapter":
            SageMakerServing(bucket=_bucket(os.environ), s3=aws.get("s3")).upload_adapter(args.folder, args.name)
            done = {"uploaded": args.name}
        elif args.step == "stage-weights":
            done = stage_weights(aws, serve, _bucket(os.environ), args.scratch)
        else:
            done = up(aws, serve, _bucket(os.environ), args.image_uri, args.account_id)
    except (DeployError, ValueError) as error:
        raise SystemExit(str(error)) from None
    print(json.dumps(done, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
