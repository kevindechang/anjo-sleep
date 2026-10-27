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

    def __init__(self, *exists: str) -> None:
        self.exists: set[str] = set(exists)
        self.status: dict[str, str] = {}  # a name's status, InService unless set
        self.copies = 0
        self.order: list[str] = []

    def _describe(self, kind: str, name: str, status_key: str) -> dict[str, Any]:
        if name not in self.exists:
            raise client_error("ValidationException", 400, f"Describe{kind}")
        return {status_key: self.status.get(name, "InService")}

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

        def gone(kind: str, key: str) -> Any:
            def delete(**params: Any) -> dict[str, Any]:
                self.order.append(f"delete_{kind}")
                self.exists.discard(params[key])
                return {}
            return delete

        return {
            "delete_model": gone("model", "ModelName"),
            "delete_endpoint_config": gone("endpoint_config", "EndpointConfigName"),
            "delete_endpoint": gone("endpoint", "EndpointName"),
            "delete_inference_component": gone("inference_component", "InferenceComponentName"),
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
        "cloudwatch": FakeAws("cloudwatch", put_metric_alarm=lambda **p: {},
                              describe_alarms=lambda **p: {"MetricAlarms": [{"StateValue": "OK"}]}),
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


class Boom(RuntimeError):
    pass


def _up(aws: deploy.Aws, image: str = IMAGE) -> None:
    deploy.up(aws, _serve(), BUCKET, image, clock=lambda: 0.0, sleep=lambda s: None)


def test_up_failure_after_endpoint_cleans_up() -> None:
    # The copy came up, then the idle net failed: everything this run made goes, the component first.
    plane = Plane()
    aws, fakes = _aws(plane)

    def refused(**params: Any) -> dict[str, Any]:
        raise Boom("put_metric_alarm refused")

    fakes["cloudwatch"]._handlers["put_metric_alarm"] = refused
    with pytest.raises(Boom):
        _up(aws)
    assert plane.order[-4:] == ["delete_inference_component", "delete_endpoint", "delete_endpoint_config",
                                "delete_model"]
    assert plane.exists == set()  # no endpoint left billing, with or without a component

    # The endpoint never came up: it goes, with its config and the model; no component was made.
    plane = Plane()
    aws, fakes = _aws(plane)
    plane.status[settings.ENDPOINT] = "Failed"
    with pytest.raises(deploy.DeployError, match="Failed"):
        _up(aws)
    assert "create_inference_component" not in plane.order and plane.exists == set()

    # The copy failed: the component this run made goes, then the endpoint it would have left empty.
    plane = Plane()
    aws, fakes = _aws(plane)
    plane.status[settings.COMPONENT] = "Failed"
    with pytest.raises(deploy.DeployError, match="Failed"):
        _up(aws)
    assert plane.exists == set()

    # A component from an earlier run stays, at 0 copies; nothing of the earlier run is deleted.
    plane = Plane(settings.ENDPOINT_CONFIG, settings.ENDPOINT, settings.COMPONENT)
    plane.copies = 1
    aws, fakes = _aws(plane)
    fakes["sagemaker"]._handlers["update_inference_component"] = refused
    with pytest.raises(Boom):
        _up(aws)
    assert plane.order[-1] == "copies:0" and plane.copies == 0
    assert {settings.ENDPOINT_CONFIG, settings.ENDPOINT, settings.COMPONENT} <= plane.exists
    assert not any(step.startswith("delete_") and step != "delete_model" for step in plane.order)


def test_stop_delete_endpoint_without_component(capsys: pytest.CaptureFixture[str]) -> None:
    # What a failed run of the old deploy could leave: an endpoint and its config, no component.
    plane = Plane(settings.ENDPOINT_CONFIG, settings.ENDPOINT)
    aws, _ = _aws(plane)
    assert deploy.main(["stop"], aws=aws) == 0  # nothing to set: no crash on a missing component
    assert '"component": null' in capsys.readouterr().out
    assert deploy.main(["stop", "--delete-endpoint"], aws=aws) == 0
    assert plane.order == ["delete_endpoint", "delete_endpoint_config"] and plane.exists == set()

    with_component = Plane(settings.ENDPOINT_CONFIG, settings.ENDPOINT, settings.COMPONENT)
    aws, _ = _aws(with_component)
    assert deploy.main(["stop", "--delete-endpoint"], aws=aws) == 0
    assert with_component.order == ["delete_inference_component", "delete_endpoint", "delete_endpoint_config"]

    # Any refusal from AWS ends the command with a message, never a traceback.
    denied = Plane(settings.ENDPOINT_CONFIG, settings.ENDPOINT)
    aws, fakes = _aws(denied)
    fakes["sagemaker"]._handlers["delete_endpoint"] = lambda **p: (_ for _ in ()).throw(
        client_error("AccessDeniedException", 403, "DeleteEndpoint"))
    with pytest.raises(SystemExit, match="AccessDenied"):
        deploy.main(["stop", "--delete-endpoint"], aws=aws)

    # A failed `up` whose cleanup is refused too says both, and what to run next.
    refused_twice = Plane()
    aws, fakes = _aws(refused_twice)
    refused_twice.status[settings.ENDPOINT] = "Failed"
    fakes["sagemaker"]._handlers["delete_endpoint"] = lambda **p: (_ for _ in ()).throw(
        client_error("AccessDeniedException", 403, "DeleteEndpoint"))
    env = {config.REVISION_ENV: REVISION, settings.BUCKET_ENV: BUCKET}
    with pytest.MonkeyPatch.context() as patch:
        for name, value in env.items():
            patch.setenv(name, value)
        with pytest.raises(SystemExit, match=r"Failed.*cleanup failed too.*AccessDenied.*stop --delete-endpoint"):
            deploy.main(["up", "--image-uri", IMAGE], aws=aws)


def test_the_idle_net_spares_a_loading_copy() -> None:
    # Nothing tells the alarm a copy is still loading, so its window outlasts the longest wake: a
    # copy can't be zeroed before the wake gives up (and the wake then sleeps it anyway).
    aws, fakes = _aws(Plane())
    deploy.idle_net(aws)
    (alarm,) = fakes["cloudwatch"].named("put_metric_alarm")
    period = alarm["Metrics"][0]["MetricStat"]["Period"]
    window_s = alarm["DatapointsToAlarm"] * period
    assert alarm["DatapointsToAlarm"] == alarm["EvaluationPeriods"]  # every minute in the window idle
    assert window_s >= settings.WAKE_TOTAL_TIMEOUT_S + 600 and settings.IDLE_MINUTES == 30
    assert settings.IDLE_MINUTES / 60 * settings.USD_PER_HOUR <= 1.31  # the worst idle cost per miss


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
