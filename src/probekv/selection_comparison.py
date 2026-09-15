"""Request-local Residual-K arithmetic; no Source or repair-mask caching."""
import math


def prepare_current_k(current):
    """One FP32 conversion and norm, owned by the caller's workspace lease."""
    current_fp32 = current.float()
    norm = current_fp32.square().sum((1, 2)).sqrt().clamp_min(1e-12)
    return current_fp32, norm


def residual_scores(source_tensor, current_fp32, norm, trim_ratio):
    """Preserve the original reduction, stable ordering and integer trimming."""
    drift = (source_tensor.float() - current_fp32).square().sum((2, 3)).sqrt()
    drift /= norm
    order = drift.argsort(dim=1, descending=True, stable=True)
    token_count = current_fp32.shape[0]
    trim = min(token_count - 1, math.ceil(trim_ratio * token_count))
    return drift.gather(1, order)[:, trim:].mean(1)
