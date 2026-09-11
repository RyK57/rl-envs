"""Policy-gradient losses for prime-rl's `[trainer.loss] type = "custom"` hook.

Each function takes prime-rl's `LossInputs` for one sequence and returns the loss summed over the
tokens in `loss_mask` (the trainer divides by the global token count of the rl component) plus a
few metrics that the trainer averages across sequences. `advantages` is per token; on a
single-turn task it is one value repeated over the action tokens. `trainer_logprobs` carry the
gradient, `inference_logprobs` are the sampler's and do not. Torch comes from the prime-rl venv
this package is installed into.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LossOutputs:
    loss: Tensor
    metrics: dict[str, Tensor]


def _mean(values: Tensor, mask: Tensor) -> Tensor:
    return values[mask].mean() if bool(mask.any()) else values.sum() * 0.0


def _weighted(per_token: Tensor, inputs) -> Tensor:
    if inputs.loss_weights is not None:
        per_token = per_token * inputs.loss_weights
    return per_token[inputs.loss_mask].sum()


def _ratio(inputs) -> Tensor:
    return torch.exp(inputs.trainer_logprobs - inputs.inference_logprobs)


def ppo_clip(inputs, clip_low: float = 0.2, clip_high: float = 0.2) -> LossOutputs:
    """PPO's clipped surrogate per token: `-min(r A, clip(r, 1 - low, 1 + high) A)`."""
    ratio = _ratio(inputs)
    clipped = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    surrogate = torch.min(ratio * inputs.advantages, clipped * inputs.advantages)
    mask = inputs.loss_mask
    metrics = {"clip_frac": _mean((ratio != clipped).float(), mask), "ratio": _mean(ratio, mask)}
    return LossOutputs(loss=_weighted(-surrogate, inputs), metrics=metrics)


def gspo(inputs, clip_low: float = 3e-4, clip_high: float = 4e-4) -> LossOutputs:
    """GSPO: one importance ratio per sequence (the geometric mean of the token ratios), clipped,
    times the sequence advantage, counted once per member token so the trainer's token
    normalisation averages over sequences."""
    mask = inputs.loss_mask
    if not bool(mask.any()):
        return LossOutputs(loss=inputs.trainer_logprobs.sum() * 0.0, metrics={})
    log_ratio = (inputs.trainer_logprobs - inputs.inference_logprobs)[mask].mean()
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    advantage = inputs.advantages[mask].mean()
    surrogate = torch.min(ratio * advantage, clipped * advantage)
    weight = inputs.loss_weights[mask].sum() if inputs.loss_weights is not None else mask.sum()
    metrics = {"clip_frac": (ratio != clipped).float(), "ratio": ratio.detach()}
    return LossOutputs(loss=-surrogate * weight, metrics=metrics)


def cispo(inputs, clip_low: float = 1.0, clip_high: float = 5.0) -> LossOutputs:
    """CISPO: the importance ratio is clipped and detached, then weights the REINFORCE term
    `A log pi`, so every token keeps a gradient and large ratios are only capped."""
    ratio = _ratio(inputs)
    weight = torch.clamp(ratio, 1 - clip_low, 1 + clip_high).detach()
    per_token = -(weight * inputs.advantages * inputs.trainer_logprobs)
    mask = inputs.loss_mask
    metrics = {"clip_frac": _mean((ratio != weight).float(), mask), "ratio": _mean(ratio, mask)}
    return LossOutputs(loss=_weighted(per_token, inputs), metrics=metrics)


def policy_gradient(inputs, importance_ratio: bool = True) -> LossOutputs:
    """The unclipped policy gradient: `-r A` with the importance ratio (exact off-policy
    correction, unbounded variance) or `-A log pi` without it (plain REINFORCE)."""
    ratio = _ratio(inputs)
    if importance_ratio:
        per_token = -(ratio * inputs.advantages)
    else:
        per_token = -(inputs.advantages * inputs.trainer_logprobs)
    metrics = {"ratio": _mean(ratio, inputs.loss_mask)}
    return LossOutputs(loss=_weighted(per_token, inputs), metrics=metrics)
