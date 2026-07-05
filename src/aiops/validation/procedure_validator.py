from __future__ import annotations

from aiops.procedure import Procedure
from aiops.types import StepPrediction, ValidationEvent


class ProcedureValidator:
    def __init__(self, procedure: Procedure) -> None:
        self.procedure = procedure
        self.expected_order = procedure.step_ids
        self.expected_index = {step_id: index for index, step_id in enumerate(self.expected_order)}

    def validate(self, predictions: list[StepPrediction]) -> list[ValidationEvent]:
        events: list[ValidationEvent] = []
        seen: set[str] = set()
        highest_index = -1

        for prediction in sorted(predictions, key=lambda item: item.start_s):
            step_index = self.expected_index.get(prediction.step_id)
            if step_index is None:
                events.append(
                    ValidationEvent(
                        kind="unknown_step",
                        step_id=prediction.step_id,
                        time_s=prediction.start_s,
                        message=f"Predicted step '{prediction.step_id}' is not in the procedure.",
                    )
                )
                continue

            if prediction.confidence < self.procedure.min_confidence:
                events.append(
                    ValidationEvent(
                        kind="low_confidence",
                        step_id=prediction.step_id,
                        time_s=prediction.start_s,
                        message=f"Step '{prediction.step_id}' confidence is below threshold.",
                    )
                )

            if prediction.step_id in seen:
                events.append(
                    ValidationEvent(
                        kind="repeated_step",
                        step_id=prediction.step_id,
                        time_s=prediction.start_s,
                        message=f"Step '{prediction.step_id}' was repeated.",
                    )
                )

            if step_index < highest_index:
                events.append(
                    ValidationEvent(
                        kind="out_of_order",
                        step_id=prediction.step_id,
                        time_s=prediction.start_s,
                        message=f"Step '{prediction.step_id}' appeared after a later step.",
                    )
                )
            else:
                missing_between = self.expected_order[highest_index + 1 : step_index]
                for missing_step in missing_between:
                    events.append(
                        ValidationEvent(
                            kind="skipped_step",
                            step_id=missing_step,
                            time_s=prediction.start_s,
                            message=f"Expected step '{missing_step}' before '{prediction.step_id}'.",
                        )
                    )
                highest_index = step_index

            seen.add(prediction.step_id)

        for missing_step in self.expected_order[highest_index + 1 :]:
            events.append(
                ValidationEvent(
                    kind="missing_step",
                    step_id=missing_step,
                    message=f"Expected step '{missing_step}' was not observed.",
                )
            )

        return events

