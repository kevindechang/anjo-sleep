"""What Glimmer on SageMaker is called and how it is bounded. Standard library only: the container
reads it too.

Every AWS name starts with ``anjo-sleep-``. The bucket's name contains ``sagemaker``, since the
operator's managed policy (AmazonSageMakerFullAccess) reaches only buckets named that way; its
suffix is the operator's choice and never in the repo (``BUCKET_ENV``).
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

REGION: Final = "us-east-1"
# The approved instance types and their hourly SageMaker hosting price while a copy is up (the
# budget ledger's modal_usd pool holds it). Each has the same single L40S (48 GB) and local NVMe, so
# the serving numbers never change: only the vCPU and RAM differ (4 and 32 GiB; 8 and 64 GiB).
INSTANCE_TYPES: Final[Mapping[str, float]] = MappingProxyType({"ml.g6e.xlarge": 2.61, "ml.g6e.2xlarge": 2.81})
DEFAULT_INSTANCE_TYPE: Final = "ml.g6e.xlarge"
INSTANCE_TYPE_ENV: Final = "ANJO_SLEEP_INSTANCE_TYPE"

ENDPOINT: Final = "anjo-sleep-glimmer"
ENDPOINT_CONFIG: Final = "anjo-sleep-glimmer-config"
VARIANT: Final = "AllTraffic"
COMPONENT: Final = "anjo-sleep-glimmer-ic"
MODEL_PREFIX: Final = "anjo-sleep-glimmer-"  # + the image tag: a SageMaker model never changes
REPOSITORY: Final = "anjo-sleep-glimmer"  # the ECR repository the image is pushed to

# The two roles Kevin creates in the console from deploy/aws/*.json; the code only names them.
EXECUTION_ROLE: Final = "anjo-sleep-sagemaker-exec"  # SageMaker pulls the image, reads weights and adapters
RUNTIME_ROLE: Final = "anjo-sleep-devhost"  # the dev host: invoke, wake, sleep, upload adapters

BUCKET_ENV: Final = "ANJO_SLEEP_S3_BUCKET"
BUCKET_PREFIX: Final = "anjo-sleep-sagemaker-"
BUCKET_PLACEHOLDER: Final = BUCKET_PREFIX + "<BUCKET_SUFFIX>"
ADAPTERS_PREFIX: Final = "adapters/"
WEIGHTS_PREFIX: Final = "weights/"

# A cold wake provisions the instance, pulls the image, copies the weights and loads them, so it
# gets twice Modal's total. If the idle alarm takes the copy back mid-wake, the wake puts it back,
# at most this many times.
WAKE_TOTAL_TIMEOUT_S: Final = 1200.0
WAKE_MAX_REASSERTS: Final = 3
# Putting Glimmer to sleep waits out an update in progress (a wake still provisioning), up to this.
SLEEP_WAIT_S: Final = 600.0
SLEEP_POLL_S: Final = 15.0
# The container's own limits: copying 60 GB of weights, then vLLM's load and FP8 quantization.
MODEL_DOWNLOAD_TIMEOUT_S: Final = 1800
STARTUP_HEALTH_TIMEOUT_S: Final = 1800
MIN_MEMORY_MB: Final = 16384

# The idle net: an alarm that sets the copy count to 0 once a copy has been up for
# IDLE_MINUTES straight with no invocation. The keep-warm ping (every 240 s) holds it off.
# Nothing in CloudWatch tells a loading copy from a serving one, so the window outlasts the longest
# wake (WAKE_TOTAL_TIMEOUT_S) with ten minutes to spare: the net never zeroes a copy still loading.
# A missed sleep costs at most IDLE_MINUTES of the instance (about $1.30).
IDLE_ALARM: Final = "anjo-sleep-idle"
IDLE_POLICY: Final = "anjo-sleep-idle-to-zero"
IDLE_MINUTES: Final = 30
SCALING_DIMENSION: Final = "sagemaker:inference-component:DesiredCopyCount"


def check_bucket(name: str) -> str:
    """The adapters bucket, named so the operator's policy reaches it. Raises ``ValueError``."""
    if not name.startswith(BUCKET_PREFIX) or len(name) > 63 or len(name) <= len(BUCKET_PREFIX):
        raise ValueError(f"the bucket is named {BUCKET_PLACEHOLDER}")
    if not all(c.islower() or c.isdigit() or c == "-" for c in name) or name.endswith("-"):
        raise ValueError("a bucket name is lowercase letters, digits and dashes")
    return name


def check_instance_type(name: str) -> str:
    """The instance type if it is approved. Raises ``ValueError`` naming the ones that are."""
    if name not in INSTANCE_TYPES:
        raise ValueError(f"the instance type {name!r} is not approved; use one of: {', '.join(INSTANCE_TYPES)}")
    return name


def scalable_resource() -> str:
    return f"inference-component/{COMPONENT}"
