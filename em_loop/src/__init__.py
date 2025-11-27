"""EM-Refinement Loop source code."""

from .config import EMRefinementConfig
from .models import (
    FusionModel,
    LPAModel,
    StudentSurrogate,
    copy_model_weights,
    update_teacher_ema,
)
from .em_refinement import (
    EMRefinementLoop,
    PseudoLabel,
    AnswerKey,
    PseudoLabelDataset,
)

__all__ = [
    "EMRefinementConfig",
    "FusionModel",
    "LPAModel",
    "StudentSurrogate",
    "copy_model_weights",
    "update_teacher_ema",
    "EMRefinementLoop",
    "PseudoLabel",
    "AnswerKey",
    "PseudoLabelDataset",
]

