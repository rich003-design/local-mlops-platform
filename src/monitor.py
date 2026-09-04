"""
Inspect local prediction logs.
"""

import json
from collections import Counter
from pathlib import Path

from src.config import MONITORING_LOG


def monitor_predictions() -> None:
    log_path = Path(
        MONITORING_LOG
    )

    if not log_path.exists():
        print(
            "No prediction logs found."
        )
        return

    labels = []

    with log_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            record = json.loads(
                line
            )

            label = (
                record["response"][
                    "label"
                ]
            )

            labels.append(
                label
            )

    counts = Counter(
        labels
    )

    print(
        f"Predictions processed: "
        f"{len(labels)}"
    )

    print(
        "Prediction distribution:"
    )

    for label, count in (
        counts.items()
    ):
        print(
            f"- {label}: {count}"
        )


if __name__ == "__main__":
    monitor_predictions()
