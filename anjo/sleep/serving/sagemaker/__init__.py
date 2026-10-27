"""Glimmer on AWS SageMaker: one L40S (``ml.g6e.xlarge``) in us-east-1, scaled to zero between uses.

``settings`` is plain data (standard library only), and ``shim`` imports only it, ``config`` and
boto3, so the container can load them without the rest of Anjo. ``serving`` is the dev host's side
(wake, sleep, the night's client, the adapter upload), ``invoke`` the signed calls under it,
``deploy`` the operator's command line, and ``token_refresh`` the dev host's keyless AWS identity.
"""
