"""Fixtures for external TEE tests."""

import pytest
import requests

TEE_URL = "https://inference-verification-tee.tinfoil.sh"


def pytest_addoption(parser):
    parser.addoption(
        "--url",
        default=TEE_URL,
        help="Base URL of the TEE deployment to test against",
    )


@pytest.fixture(scope="session")
def base_url(request):
    return request.config.getoption("--url").rstrip("/")


@pytest.fixture(scope="session")
def verify_response(base_url):
    """Call /verify once and cache the result for the session."""
    resp = requests.post(
        f"{base_url}/verify",
        json={"n_prompts": 1, "max_tokens": 10},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()
