"""Configuration namespace isolation for the shared backbone."""

from contextlib import contextmanager
from contextvars import ContextVar

from amgpn.config import require

_BACKBONE_NAMESPACE = ContextVar("amgpn_backbone_namespace", default="GPN_V4_adaMulti_clean.py")


@contextmanager
def backbone_scope(namespace):
    """Keep each historical variant's settings separate, including nested construction."""
    token = _BACKBONE_NAMESPACE.set(namespace)
    try:
        yield
    finally:
        _BACKBONE_NAMESPACE.reset(token)


def require_backbone(suffix):
    return require(f"{_BACKBONE_NAMESPACE.get()}.{suffix}")


def resolve_backbone(suffix, explicit=None):
    return explicit if explicit is not None else require_backbone(suffix)
