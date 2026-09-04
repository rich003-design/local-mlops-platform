"""
Send a sample prediction request to the deployed API.
"""

import requests

from sklearn.datasets import (
    load_breast_cancer,
)


API_URL = (
    "http://127.0.0.1:8000/predict"
)


def build_payload():
    dataset = load_breast_cancer(
        as_frame=True
    )

    sample = (
        dataset.data.iloc[0]
    )

    payload = {}

    for column, value in (
        sample.items()
    ):
        api_name = (
            column.replace(
                " ",
                "_",
            )
        )

        payload[api_name] = float(
            value
        )

    return payload


def test_prediction():
    payload = build_payload()

    response = requests.post(
        API_URL,
        json=payload,
        timeout=10,
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )

    print(
        response.json()
    )


if __name__ == "__main__":
    test_prediction()
