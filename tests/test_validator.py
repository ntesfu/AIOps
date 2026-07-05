from aiops.procedure import Procedure, ProcedureStep
from aiops.types import StepPrediction
from aiops.validation import ProcedureValidator


def make_procedure() -> Procedure:
    return Procedure(
        name="demo",
        steps=(
            ProcedureStep("pick", "Pick"),
            ProcedureStep("attach", "Attach"),
            ProcedureStep("inspect", "Inspect"),
        ),
        min_confidence=0.7,
    )


def test_validator_accepts_ordered_sequence() -> None:
    predictions = [
        StepPrediction("pick", "Pick", 0.0, 1.0, 0.9),
        StepPrediction("attach", "Attach", 1.0, 2.0, 0.9),
        StepPrediction("inspect", "Inspect", 2.0, 3.0, 0.9),
    ]

    assert ProcedureValidator(make_procedure()).validate(predictions) == []


def test_validator_flags_sequence_problems() -> None:
    predictions = [
        StepPrediction("pick", "Pick", 0.0, 1.0, 0.9),
        StepPrediction("inspect", "Inspect", 1.0, 2.0, 0.6),
        StepPrediction("attach", "Attach", 2.0, 3.0, 0.9),
        StepPrediction("attach", "Attach", 3.0, 4.0, 0.9),
    ]

    events = ProcedureValidator(make_procedure()).validate(predictions)
    kinds = [event.kind for event in events]

    assert "skipped_step" in kinds
    assert "low_confidence" in kinds
    assert "out_of_order" in kinds
    assert "repeated_step" in kinds
