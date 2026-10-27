# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The operator's SageMaker steps against stand-ins checked by botocore's own service models (every
name and type in every call), the image's build context, and the dev host's token file. Offline:
nothing is signed or sent."""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep.serving import config
from anjo.sleep.serving.sagemaker import deploy, settings, token_refresh
from anjo.sleep.tests.aws_fakes import FakeAws, client_error

REVISION = "0123456789abcdef0123456789abcdef01234567"
BUCKET = "anjo-sleep-sagemaker-test"
ACCOUNT = "0" * 12  # built at runtime: no account-shaped literal in the repo
IMAGE = f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/{settings.REPOSITORY}:vllm0.28.0-0123456789ab-1a2b3c4d"


class Plane:
    """SageMaker's control plane for our names: what exists, and every call in order."""

    def __init__(self) -> None:
        self.exists: set[str] = set()
        self.copies = 0
        self.order: list[str] = []

    def _describe(self, kind: str, name: str, status_key: str) -> dict[str, Any]:
        if name not in self.exists:
            raise client_error("ValidationException", 400, f"Describe{kind}")
        return {status_key: "InService"}

    def handlers(self) -> dict[str, Any]:
        def made(kind: str, key: str) -> Any:
            def create(**params: Any) -> dict[str, Any]:
                self.order.append(f"create_{kind}")
                self.exists.add(params[key])
                if kind == "inference_component":
                    self.copies = params["RuntimeConfig"]["CopyCount"]
                return {}
            return create

        def describe_component(**params: Any) -> dict[str, Any]:
            found = self._describe("InferenceComponent", params["InferenceComponentName"], "InferenceComponentStatus")
            return {**found, "RuntimeConfig": {"DesiredCopyCount": self.copies, "CurrentCopyCount": self.copies}}

        def set_copies(**params: Any) -> dict[str, Any]:
            self.order.append(f"copies:{params['DesiredRuntimeConfig']['CopyCount']}")
            self.copies = params["DesiredRuntimeConfig"]["CopyCount"]
            return {}

        return {
            "describe_model": lambda **p: self._describe("Model", p["ModelName"], "ModelName"),
            "describe_endpoint_config": lambda **p: self._describe("EndpointConfig", p["EndpointConfigName"], "x"),
            "describe_endpoint": lambda **p: self._describe("Endpoint", p["EndpointName"], "EndpointStatus"),
            "describe_inference_component": describe_component,
            "create_model": made("model", "ModelName"),
            "create_endpoint_config": made("endpoint_config", "EndpointConfigName"),
            "create_endpoint": made("endpoint", "EndpointName"),
            "create_inference_component": made("inference_component", "InferenceComponentName"),
            "update_inference_component_runtime_config": set_copies,
        }


def _aws(plane: Plane) -> tuple[deploy.Aws, dict[str, FakeAws]]:
    fakes = {
        "sagemaker": FakeAws("sagemaker", **plane.handlers()),
        "sts": FakeAws("sts", get_caller_identity=lambda: {"Account": ACCOUNT}),
        "application-autoscaling": FakeAws(
            "application-autoscaling", register_scalable_target=lambda **p: {},
            put_scaling_policy=lambda **p: {"PolicyARN": "policy-arn"}),
        "cloudwatch": FakeAws("cloudwatch", put_metric_alarm=lambda **p: {}),
    }
    return deploy.Aws(made=dict(fakes)), fakes


def _serve() -> config.ServeConfig:
    return config.ServeConfig.from_env({config.REVISION_ENV: REVISION})


def test_up_builds_everything_once_under_our_names_and_ends_asleep() -> None:
    plane = Plane()
    aws, fakes = _aws(plane)
    done = deploy.up(aws, _serve(), BUCKET, IMAGE, clock=lambda: 0.0, sleep=lambda s: None)
    assert plane.order == ["create_model", "create_endpoint_config", "create_endpoint",
                           "create_inference_component", "copies:0"]  # the first copy, then asleep
    assert done["copies"] == 0 and plane.copies == 0
    assert all(name.startswith("anjo-sleep-") for name in plane.exists)

    sm = fakes["sagemaker"]
    (model,) = sm.named("create_model")
    container = model["PrimaryContainer"]
    assert model["ExecutionRoleArn"].endswith(f":role/{settings.EXECUTION_ROLE}")
    assert container["ModelDataSource"]["S3DataSource"]["S3Uri"] == \
        f"s3://{BUCKET}/weights/{config.MODEL}/{REVISION}/"  # the staged copy, not Hugging Face
    assert container["Environment"][config.REVISION_ENV] == REVISION
    (variant,) = sm.named("create_endpoint_config")[0]["ProductionVariants"]
    assert (variant["InstanceType"], variant["ManagedInstanceScaling"]) == (
        "ml.g6e.xlarge", {"Status": "ENABLED", "MinInstanceCount": 0, "MaxInstanceCount": 1})
    (component,) = sm.named("create_inference_component")
    assert component["Specification"]["ComputeResourceRequirements"]["NumberOfAcceleratorDevicesRequired"] == 1

    (target,) = fakes["application-autoscaling"].named("register_scalable_target")
    assert (target["MinCapacity"], target["MaxCapacity"]) == (0, 1)
    (policy,) = fakes["application-autoscaling"].named("put_scaling_policy")
    assert policy["StepScalingPolicyConfiguration"]["AdjustmentType"] == "ExactCapacity"
    assert policy["StepScalingPolicyConfiguration"]["StepAdjustments"][0]["ScalingAdjustment"] == 0
    (alarm,) = fakes["cloudwatch"].named("put_metric_alarm")
    assert alarm["EvaluationPeriods"] == alarm["DatapointsToAlarm"] == settings.IDLE_MINUTES
    assert alarm["AlarmActions"] == ["policy-arn"] and alarm["TreatMissingData"] == "notBreaching"

    # A second run with a new image makes a new model and moves the component to it, nothing else.
    plane.order.clear()
    sm.calls.clear()

    def update(**params: Any) -> dict[str, Any]:
        plane.order.append("update_component")
        plane.copies = params["RuntimeConfig"]["CopyCount"]
        return {}

    sm._handlers["update_inference_component"] = update
    deploy.up(aws, _serve(), BUCKET, IMAGE.replace("1a2b3c4d", "5e6f7a8b"), clock=lambda: 0.0, sleep=lambda s: None)
    assert plane.order == ["create_model", "update_component", "copies:0"]


def test_up_puts_glimmer_to_sleep_even_when_the_copy_fails() -> None:
    plane = Plane()
    aws, fakes = _aws(plane)
    handlers = plane.handlers()

    def failing(**params: Any) -> dict[str, Any]:
        found = handlers["describe_inference_component"](**params)
        return {**found, "InferenceComponentStatus": "Failed"}

    fakes["sagemaker"]._handlers["describe_inference_component"] = failing
    with pytest.raises(deploy.DeployError, match="Failed"):
        deploy.up(aws, _serve(), BUCKET, IMAGE, clock=lambda: 0.0, sleep=lambda s: None)
    assert plane.order[-1] == "copies:0"


def test_up_refuses_an_image_outside_our_repository() -> None:
    with pytest.raises(deploy.DeployError):
        deploy.model_name(IMAGE.replace(settings.REPOSITORY, "someone-else"))
    with pytest.raises(deploy.DeployError):
        deploy.model_name(IMAGE.replace("us-east-1", "eu-west-1"))
    assert len(deploy.model_name(IMAGE)) <= 63 and "." not in deploy.model_name(IMAGE)


def test_the_image_context_holds_only_the_shim_and_imports_alone(tmp_path: Path) -> None:
    context = tmp_path / "context"
    deploy.write_context(context)
    held = sorted(str(p.relative_to(context)) for p in context.rglob("*") if p.is_file())
    assert held == sorted([*deploy.CONTAINER_FILES, "Dockerfile"])  # no .env, no data, no other code
    with pytest.raises(deploy.DeployError, match="not empty"):
        deploy.write_context(context)
    # The container's Python sees only the context: the shim must import from it alone.
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(context)}
    probe = "import anjo.sleep.serving.sagemaker.shim as s; print(s.SHIM_PORT)"
    out = subprocess.run([sys.executable, "-c", probe], env=env, cwd=tmp_path, capture_output=True, text=True,
                         timeout=60, check=False)
    assert out.returncode == 0 and out.stdout.strip() == "8080", out.stderr[-2000:]

    serve = _serve()
    tag = deploy.image_tag(serve)
    assert tag.startswith("vllm0.28.0-0123456789ab-") and tag == deploy.image_tag(serve)  # deterministic
    commands = "\n".join(deploy.image_commands(context, serve, tag))
    assert "--platform linux/amd64" in commands and "$AWS_ACCOUNT_ID" in commands and ACCOUNT not in commands


def test_the_token_file_is_written_whole_and_private(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    token = ".".join(("header", "payload", "signature"))  # a JWT's shape, nothing real
    seen: list[httpx.Request] = []

    def metadata(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=token)

    out = tmp_path / "aws" / "token"
    token_refresh.write(token_refresh.fetch(transport=httpx.MockTransport(metadata)), out)
    assert out.read_text() == token and stat.S_IMODE(out.stat().st_mode) == 0o600
    (request,) = seen
    assert request.headers["Metadata-Flavor"] == "Google"
    assert request.url.params["audience"] == token_refresh.AUDIENCE == "anjo-sleep-devhost-aws"
    assert [p.name for p in out.parent.iterdir()] == ["token"]  # no half-written file left

    for reply in (httpx.Response(404), httpx.Response(200, text="<html>not a token</html>")):
        with pytest.raises(token_refresh.TokenError) as raised:
            token_refresh.fetch(transport=httpx.MockTransport(lambda r, reply=reply: reply))
        assert token not in str(raised.value)
    assert token not in capsys.readouterr().out
