from aiops.models import UniformProcedureBaseline
from aiops.procedure import Procedure, ProcedureStep


def test_baseline_segments_cover_duration() -> None:
    procedure = Procedure(
        name="demo",
        steps=(ProcedureStep("a", "A"), ProcedureStep("b", "B")),
    )

    predictions = UniformProcedureBaseline(procedure).predict(duration_s=10.0)

    assert len(predictions) == 2
    assert predictions[0].start_s == 0.0
    assert predictions[-1].end_s == 10.0
    assert [prediction.step_id for prediction in predictions] == ["a", "b"]

