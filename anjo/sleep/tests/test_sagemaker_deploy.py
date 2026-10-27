# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The operator's SageMaker steps against stand-ins checked by botocore's own service models (every
name and type in every call), the image's build context, and the dev host's token file. Offline:
nothing is signed or sent."""
from __future__ import annotations

import json
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
                if params[key] not in self.exists:  # SageMaker's reply for a name that was never made
                    raise client_error("ValidationException", 400, f"Delete{kind}",
                                       Message=f'Could not find {kind} "{params[key]}".')
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


UNSUPPORTED = ("Unsupported manifest media type application/vnd.oci.image.index.v1+json for image "
               "x.dkr.ecr.us-east-1.amazonaws.com/anjo-sleep-glimmer:tag")


def test_a_refused_create_model_is_not_cleaned_up_as_if_it_were_made(capsys: pytest.CaptureFixture[str]) -> None:
    plane = Plane()
    aws, fakes = _aws(plane)

    def refused(**params: Any) -> dict[str, Any]:
        raise client_error("ValidationException", 400, "CreateModel", Message=UNSUPPORTED)

    fakes["sagemaker"]._handlers["create_model"] = refused
    with pytest.raises(Exception, match="Unsupported manifest media type") as raised:
        _up(aws)
    seen = " ".join([str(raised.value), *getattr(raised.value, "__notes__", ())]) + "".join(capsys.readouterr())
    assert "cleanup failed" not in seen
    assert plane.exists == set()


def test_up_failure_at_create_endpoint_still_deletes_the_model_and_config_it_made() -> None:
    plane = Plane()
    aws, fakes = _aws(plane)

    def refused(**params: Any) -> dict[str, Any]:
        raise Boom("create_endpoint refused")

    fakes["sagemaker"]._handlers["create_endpoint"] = refused
    with pytest.raises(Boom) as raised:
        _up(aws)
    assert "delete_model" in plane.order and "delete_endpoint_config" in plane.order
    assert "delete_endpoint" not in plane.order and plane.exists == set()
    assert not getattr(raised.value, "__notes__", ())


def test_an_interrupted_create_endpoint_that_aws_accepted_is_deleted() -> None:
    plane = Plane()
    aws, fakes = _aws(plane)

    def accepted_then_interrupted(**params: Any) -> dict[str, Any]:
        plane.exists.add(params["EndpointName"])  # AWS took it; the call never returned
        raise KeyboardInterrupt

    fakes["sagemaker"]._handlers["create_endpoint"] = accepted_then_interrupted
    with pytest.raises(KeyboardInterrupt):
        _up(aws)
    assert "delete_endpoint" in plane.order and plane.exists == set()


def test_a_delete_that_fails_for_real_still_reports_cleanup_failed() -> None:
    plane = Plane()
    aws, fakes = _aws(plane)

    def refused(**params: Any) -> dict[str, Any]:
        raise Boom("create_endpoint refused")

    def denied(**params: Any) -> dict[str, Any]:
        raise client_error("AccessDeniedException", 400, "DeleteModel", Message="not allowed")

    fakes["sagemaker"]._handlers["create_endpoint"] = refused
    fakes["sagemaker"]._handlers["delete_model"] = denied
    with pytest.raises(Boom) as raised:
        _up(aws)
    notes = " ".join(getattr(raised.value, "__notes__", ()))
    assert "cleanup failed" in notes and "deploy stop --delete-endpoint" in notes


def test_a_failed_wait_shows_sagemakers_failure_reason() -> None:
    plane = Plane()
    aws, fakes = _aws(plane)

    def failed(**params: Any) -> dict[str, Any]:
        if params["InferenceComponentName"] not in plane.exists:
            raise client_error("ValidationException", 400, "DescribeInferenceComponent")
        return {"InferenceComponentStatus": "Failed", "FailureReason": "xyz"}

    fakes["sagemaker"]._handlers["describe_inference_component"] = failed
    with pytest.raises(deploy.DeployError, match="xyz"):
        _up(aws)


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
    assert settings.IDLE_MINUTES / 60 * settings.INSTANCE_TYPES[settings.DEFAULT_INSTANCE_TYPE] <= 1.31  # the worst idle cost per miss


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


def test_the_printed_build_pushes_one_docker_v2_manifest_without_attestations(tmp_path: Path) -> None:
    serve = _serve()
    commands = deploy.image_commands(tmp_path, serve, "tag")
    build = next(line for line in commands if "buildx build" in line)
    uri = f"$AWS_ACCOUNT_ID.dkr.ecr.{settings.REGION}.amazonaws.com/{settings.REPOSITORY}:tag"
    assert "--provenance=false" in build and "--sbom=false" in build and "oci-mediatypes=false" in build
    assert f"name={uri}" in build and "push=true" in build
    assert "--platform linux/amd64" in build and "VLLM_VERSION=" in build


def _endpoint_env(patch: pytest.MonkeyPatch, **extra: str) -> None:
    patch.delenv(settings.INSTANCE_TYPE_ENV, raising=False)
    for name, value in {config.REVISION_ENV: REVISION, settings.BUCKET_ENV: BUCKET, **extra}.items():
        patch.setenv(name, value)


def _status(aws: deploy.Aws, capsys: pytest.CaptureFixture[str], *args: str) -> dict[str, Any]:
    capsys.readouterr()
    assert deploy.main(["status", *args], aws=aws) == 0
    done: dict[str, Any] = json.loads(capsys.readouterr().out)
    return done


def test_the_default_type_is_g6e_xlarge_at_2_61(capsys: pytest.CaptureFixture[str]) -> None:
    assert settings.DEFAULT_INSTANCE_TYPE == "ml.g6e.xlarge"
    assert settings.INSTANCE_TYPES == {"ml.g6e.xlarge": 2.61, "ml.g6e.2xlarge": 2.81}
    with pytest.raises(TypeError):
        settings.INSTANCE_TYPES["ml.g6e.12xlarge"] = 1.0
    with pytest.MonkeyPatch.context() as patch:
        _endpoint_env(patch)
        aws, _ = _aws(Plane())
        done = _status(aws, capsys)
    assert (done["instance_type"], done["usd_per_hour_while_up"]) == ("ml.g6e.xlarge", 2.61)


def test_the_cli_option_picks_the_type_for_the_config_and_status(capsys: pytest.CaptureFixture[str]) -> None:
    plane = Plane()
    aws, fakes = _aws(plane)
    with pytest.MonkeyPatch.context() as patch:
        _endpoint_env(patch)
        assert deploy.main(["up", "--image-uri", IMAGE, "--instance-type", "ml.g6e.2xlarge"], aws=aws) == 0
        (variant,) = fakes["sagemaker"].named("create_endpoint_config")[0]["ProductionVariants"]
        assert variant["InstanceType"] == "ml.g6e.2xlarge"
        done = _status(aws, capsys, "--instance-type", "ml.g6e.2xlarge")
    assert (done["instance_type"], done["usd_per_hour_while_up"]) == ("ml.g6e.2xlarge", 2.81)


def test_an_unknown_type_is_refused_before_any_aws_call() -> None:
    plane = Plane()
    aws, fakes = _aws(plane)
    aws.get = lambda service: pytest.fail(f"AWS was reached: {service}")  # type: ignore[method-assign]
    with pytest.MonkeyPatch.context() as patch:
        _endpoint_env(patch)
        for step in (["up", "--image-uri", IMAGE], ["status"], ["stop"]):
            with pytest.raises(SystemExit, match=r"ml\.p5\.48xlarge.*ml\.g6e\.xlarge.*ml\.g6e\.2xlarge"):
                deploy.main([*step, "--instance-type", "ml.p5.48xlarge"], aws=aws)
        _endpoint_env(patch, **{settings.INSTANCE_TYPE_ENV: "ml.g5.xlarge"})
        with pytest.raises(SystemExit, match=r"ml\.g5\.xlarge.*ml\.g6e\.xlarge"):
            deploy.main(["status"], aws=aws)
    assert plane.order == [] and all(not f.calls for f in fakes.values())


def test_the_env_var_picks_the_type_and_the_option_overrides_it(capsys: pytest.CaptureFixture[str]) -> None:
    aws, _ = _aws(Plane())
    with pytest.MonkeyPatch.context() as patch:
        _endpoint_env(patch, **{settings.INSTANCE_TYPE_ENV: "ml.g6e.2xlarge"})
        assert _status(aws, capsys)["usd_per_hour_while_up"] == 2.81
        done = _status(aws, capsys, "--instance-type", "ml.g6e.xlarge")
        assert (done["instance_type"], done["usd_per_hour_while_up"]) == ("ml.g6e.xlarge", 2.61)


def test_status_reports_the_type_an_existing_endpoint_really_has(capsys: pytest.CaptureFixture[str]) -> None:
    plane = Plane(settings.ENDPOINT_CONFIG, settings.ENDPOINT)
    aws, fakes = _aws(plane)
    fakes["sagemaker"]._handlers["describe_endpoint_config"] = lambda **p: {
        "ProductionVariants": [{"InstanceType": "ml.g6e.2xlarge"}]}
    with pytest.MonkeyPatch.context() as patch:
        _endpoint_env(patch)
        done = _status(aws, capsys)  # asked for the default, but the endpoint runs the larger
    assert (done["instance_type"], done["usd_per_hour_while_up"]) == ("ml.g6e.2xlarge", 2.81)
