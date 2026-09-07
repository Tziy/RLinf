"""Shared validation helpers for FDVLA real-time execution conditions."""


def resolve_execution_horizon(
    predicted_horizon: int,
    configured_horizon: int,
    *,
    field_name: str,
) -> int:
    predicted_horizon = int(predicted_horizon)
    configured_horizon = int(configured_horizon)
    execution_horizon = (
        predicted_horizon if configured_horizon < 0 else configured_horizon
    )
    if not 1 <= execution_horizon <= predicted_horizon:
        raise ValueError(
            f"{field_name} must be within the predicted action horizon: "
            f"execution_horizon={execution_horizon}, "
            f"predicted_horizon={predicted_horizon}"
        )
    return execution_horizon


def resolve_eval_execution_horizon(
    predicted_horizon: int, configured_horizon: int
) -> int:
    return resolve_execution_horizon(
        predicted_horizon,
        configured_horizon,
        field_name="eval_execution_horizon",
    )


def resolve_train_execution_horizon(
    predicted_horizon: int, configured_horizon: int
) -> int:
    return resolve_execution_horizon(
        predicted_horizon,
        configured_horizon,
        field_name="train_execution_horizon",
    )


def fixed_age_publish_frame(fixed_age: int, execution_horizon: int) -> int:
    """Return the within-chunk publication phase for an exact fixed packet age."""
    fixed_age = int(fixed_age)
    execution_horizon = int(execution_horizon)
    if fixed_age < 0:
        raise ValueError(f"fixed_age must be non-negative, got {fixed_age}")
    if execution_horizon < 1:
        raise ValueError(f"execution_horizon must be positive, got {execution_horizon}")
    remainder = fixed_age % execution_horizon
    return execution_horizon if remainder == 0 else execution_horizon - remainder
