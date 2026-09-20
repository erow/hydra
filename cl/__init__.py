"""Two-view contrastive learners: SimCLR, SupCon, SimLAP (no momentum encoder)."""

from cl.model import ContrastiveModel, build_model

__all__ = ["ContrastiveModel", "build_model"]
