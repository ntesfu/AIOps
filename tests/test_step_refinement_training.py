from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiops.data.stategraph_cache import StateGraphCacheRecord
from aiops.models.stategraph_psr import (
    StateGraphLossConfig,
    StateGraphPSRConfig,
    build_stategraph_loss,
    build_stategraph_psr,
)
from aiops.training.train_stategraph_psr import (
    _configure_step_only_training,
    _initialization_model_configs_compatible,
    _psr_step_class_weights,
)

torch = pytest.importorskip("torch")


def _config(**overrides) -> StateGraphPSRConfig:
    values = {
        "motion_dim": 4,
        "appearance_dim": 3,
        "sensor_dim": 2,
        "num_steps": 4,
        "num_completion_components": 3,
        "num_components": 3,
        "hidden_dim": 8,
        "num_temporal_blocks": 1,
        "attention_every": 0,
        "num_heads": 2,
        "dropout": 0.0,
    }
    values.update(overrides)
    return StateGraphPSRConfig(**values)


def _inputs(length: int = 8):
    return (
        torch.randn(1, length, 4),
        torch.randn(1, length, 3),
        torch.randn(1, length, 2),
        torch.ones(1, length, dtype=torch.bool),
        torch.ones(1, length, 3, dtype=torch.bool),
    )


def _targets(length: int = 8):
    valid = torch.ones(1, length, dtype=torch.bool)
    completion = torch.zeros(1, length, 3)
    completion[0, 2, 0] = 1.0
    completion[0, 5, 1] = 1.0
    return {
        "valid_mask": valid,
        "step": torch.arange(length).remainder(4).unsqueeze(0),
        "completion": completion,
        "component_outcome": torch.full(
            (1, length, 3), -100, dtype=torch.long
        ),
        "state": torch.ones(1, length, 3, dtype=torch.long),
        "state_mask": torch.ones(1, length, 3, dtype=torch.bool),
        "boundary": torch.zeros(1, length),
        "next_step": torch.arange(length).remainder(4).unsqueeze(0),
    }


def test_legacy_checkpoint_extends_with_zero_initialized_refinement() -> None:
    torch.manual_seed(5)
    legacy_config = _config()
    extended_config = _config(
        num_psr_step_refinement_blocks=2,
        psr_step_refinement_right_context=1,
    )
    assert _initialization_model_configs_compatible(
        legacy_config.to_dict(), extended_config.to_dict()
    )
    legacy = build_stategraph_psr(legacy_config).eval()
    extended = build_stategraph_psr(extended_config).eval()
    missing = extended.load_state_dict(legacy.state_dict(), strict=False)
    assert missing.missing_keys
    assert all(
        name.startswith("psr_step_refinement.") for name in missing.missing_keys
    )
    assert not missing.unexpected_keys
    with torch.no_grad():
        output = extended(*_inputs())
    torch.testing.assert_close(
        output["psr_step_logits"], output["psr_step_raw_logits"]
    )
    assert output["psr_step_refined_logits"] is not None


def test_step_refinement_has_only_configured_future_context() -> None:
    torch.manual_seed(7)
    model = build_stategraph_psr(
        _config(
            num_psr_step_refinement_blocks=2,
            psr_step_refinement_right_context=1,
        )
    ).eval()
    torch.nn.init.normal_(model.psr_step_refinement.output[-1].weight)
    inputs = _inputs()
    changed = tuple(value.clone() for value in inputs)
    changed[0][:, 5:] += 100.0
    changed[1][:, 5:] -= 100.0
    with torch.no_grad():
        original_output = model(*inputs)["psr_step_logits"]
        changed_output = model(*changed)["psr_step_logits"]
    # Row 3 may inspect row 4, but never row 5 or later.
    torch.testing.assert_close(
        original_output[:, :4], changed_output[:, :4], rtol=0.0, atol=0.0
    )


def test_step_only_freeze_selects_exact_parameter_prefixes_and_backpropagates() -> None:
    model = build_stategraph_psr(
        _config(num_psr_step_refinement_blocks=2)
    )
    trainable = _configure_step_only_training(model)
    assert trainable
    assert all(
        name.startswith(("psr_step_head.", "psr_step_refinement."))
        for name in trainable
    )
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == set(trainable)

    output = model(*_inputs())
    losses = build_stategraph_loss(StateGraphLossConfig())(
        output, _targets()
    )
    losses["total"].backward()
    assert model.psr_step_head.weight.grad is not None
    assert model.psr_step_refinement.output[-1].weight.grad is not None
    assert model.motion_stem.net[1].weight.grad is None


def test_psr_step_weights_are_used_by_focal_loss() -> None:
    torch.manual_seed(13)
    model = build_stategraph_psr(_config()).eval()
    output = model(*_inputs())
    criterion = build_stategraph_loss(StateGraphLossConfig())
    targets = _targets()
    unweighted = criterion(output, targets)["psr_step"]
    weights = torch.tensor([0.25, 0.5, 2.0, 4.0])
    weighted = criterion(
        output, targets, psr_step_class_weights=weights
    )["psr_step"]
    assert not torch.isclose(unweighted, weighted)


def test_loss_prefers_explicit_recording_global_step_target() -> None:
    model = build_stategraph_psr(_config()).eval()
    output = model(*_inputs())
    logits = torch.full((1, 8, 4), -8.0)
    logits[..., 3] = 8.0
    output["psr_step_logits"] = logits
    output["psr_step_raw_logits"] = logits
    targets = _targets()
    targets["completion"].zero_()
    fallback_loss = build_stategraph_loss(StateGraphLossConfig())(
        output, targets
    )["psr_step"]
    targets["psr_step_target"] = torch.full(
        (1, 8), 3, dtype=torch.long
    )
    explicit_loss = build_stategraph_loss(StateGraphLossConfig())(
        output, targets
    )["psr_step"]
    assert explicit_loss < fallback_loss


def test_train_only_run_up_weights_favor_rare_classes_with_bounds(
    tmp_path: Path,
) -> None:
    completion = np.zeros((24, 3), dtype=np.float32)
    completion[18, 0] = 1.0
    completion[20, 1] = 1.0
    completion[21, 2] = 1.0
    path = tmp_path / "train.npz"
    np.savez_compressed(path, completion=completion)
    record = StateGraphCacheRecord(
        recording_id="train",
        split="train",
        path=path,
        num_steps=4,
        motion_dim=1,
        appearance_dim=1,
        sensor_dim=1,
        num_components=3,
        num_completion_components=3,
    )
    ignored_validation = StateGraphCacheRecord(
        recording_id="validation",
        split="val",
        path=tmp_path / "must-not-be-opened.npz",
        num_steps=4,
        motion_dim=1,
        appearance_dim=1,
        sensor_dim=1,
        num_components=3,
        num_completion_components=3,
    )
    weights = _psr_step_class_weights(
        [record, ignored_validation],
        4,
        method="inverse_sqrt",
        cap=3.0,
    )
    assert weights is not None
    assert weights.mean() == pytest.approx(1.0, abs=1e-6)
    assert weights.max() <= 3.0
    assert weights.min() >= 1.0 / 3.0
    # Step 1 owns the long run-up; later steps are rarer and weigh more.
    assert weights[3] > weights[1]
