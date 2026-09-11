"""Checks for the custom prime-rl losses: values on known inputs, clipping, masking, gradients."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rl_losses import cispo, gspo, policy_gradient, ppo_clip  # noqa: E402


def inputs(trainer, inference, advantages, mask=None, weights=None):
    trainer = torch.tensor(trainer, requires_grad=True)
    return SimpleNamespace(
        trainer_logprobs=trainer,
        inference_logprobs=torch.tensor(inference),
        ref_logprobs=None,
        advantages=torch.tensor(advantages),
        loss_mask=torch.tensor(mask if mask is not None else [True] * len(advantages)),
        loss_weights=torch.tensor(weights) if weights is not None else None,
    )


def test_ppo_clip_matches_the_unclipped_surrogate_on_policy():
    batch = inputs([-1.0, -2.0, -0.5], [-1.0, -2.0, -0.5], [1.0, 1.0, -1.0])
    out = ppo_clip(batch)
    assert out.loss.item() == pytest.approx(-(1.0 + 1.0 - 1.0))
    assert out.metrics["clip_frac"].item() == 0.0
    out.loss.backward()
    grads = batch.trainer_logprobs.grad
    assert grads[0] < 0 and grads[2] > 0, "positive advantage raises the logprob, negative lowers it"


def test_ppo_clip_zeroes_the_gradient_of_clipped_tokens():
    batch = inputs([0.0, 0.0], [-1.0, -1.0], [1.0, -1.0])  # ratio e on both tokens
    out = ppo_clip(batch, clip_low=0.2, clip_high=0.2)
    assert out.metrics["clip_frac"].item() == 1.0
    out.loss.backward()
    assert batch.trainer_logprobs.grad[0].item() == 0.0, "a positive-advantage token above 1+eps is clipped"
    assert batch.trainer_logprobs.grad[1].item() != 0.0, "a negative-advantage token above the clip still learns"


def test_masks_and_weights_select_tokens():
    batch = inputs(
        [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], mask=[True, False, True], weights=[2.0, 1.0, 1.0]
    )
    assert ppo_clip(batch).loss.item() == pytest.approx(-3.0)
    assert policy_gradient(batch).loss.item() == pytest.approx(-3.0)
    assert cispo(batch).loss.item() == pytest.approx(3.0), "-(1 * A * logprob) with logprob -1"


def test_gspo_uses_one_sequence_ratio():
    batch = inputs([-1.0, -1.0, -3.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], mask=[True, True, False])
    out = gspo(batch)
    assert out.metrics["ratio"].item() == pytest.approx(1.0) and out.metrics["clip_frac"].item() == 0.0
    assert out.loss.item() == pytest.approx(-2.0), "surrogate 1 * 1 counted once per member token"
    spiky = inputs([0.0, -1.0], [-1.0, -1.0], [1.0, 1.0])
    out = gspo(spiky, clip_low=3e-4, clip_high=4e-4)
    assert out.metrics["clip_frac"].item() == 1.0
    out.loss.backward()
    assert spiky.trainer_logprobs.grad.abs().sum().item() == 0.0, "a clipped sequence with positive advantage stops"


def test_policy_gradient_variants():
    batch = inputs([-0.5, -0.5], [-1.0, -1.0], [1.0, 1.0])
    with_ratio = policy_gradient(batch, importance_ratio=True)
    assert with_ratio.loss.item() == pytest.approx(-2 * torch.exp(torch.tensor(0.5)).item())
    plain = policy_gradient(inputs([-0.5, -0.5], [-1.0, -1.0], [1.0, 1.0]), importance_ratio=False)
    assert plain.loss.item() == pytest.approx(1.0)
