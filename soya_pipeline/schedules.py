"""Learning-rate schedule (pure Python, so it is unit-testable without PyTorch)."""
import math


def lr_lambdas(steps_per_epoch: int, epochs: int, warmup_epochs: int):
    """-> (backbone_factor(step), head_factor(step)); multipliers of the base learning rates."""
    total, warm = epochs * steps_per_epoch, warmup_epochs * steps_per_epoch
    head_ramp = max(steps_per_epoch // 2, 1)

    def head(step):
        return min(1.0, (step + 1) / head_ramp) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))

    def backbone(step):
        if step < warm:
            return 0.0
        ramp = min(1.0, (step - warm + 1) / steps_per_epoch)
        return ramp * 0.5 * (1 + math.cos(math.pi * min(step - warm, total - warm) / max(total - warm, 1)))

    return backbone, head
