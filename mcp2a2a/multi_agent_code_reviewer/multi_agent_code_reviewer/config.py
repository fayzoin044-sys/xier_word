"""Network configuration for the independent A2A services."""

import os

BIND_HOST = os.getenv("CODE_REVIEWER_A2A_BIND_HOST", "127.0.0.1")
PUBLIC_HOST = os.getenv("CODE_REVIEWER_A2A_PUBLIC_HOST", "127.0.0.1")

SECURITY_A2A_PORT = int(os.getenv("SECURITY_A2A_PORT", "8301"))
QUALITY_A2A_PORT = int(os.getenv("QUALITY_A2A_PORT", "8302"))
TEST_A2A_PORT = int(os.getenv("TEST_A2A_PORT", "8303"))
FIX_A2A_PORT = int(os.getenv("FIX_A2A_PORT", "8304"))


def a2a_url(port: int) -> str:
    return f"http://{PUBLIC_HOST}:{port}"


SECURITY_A2A_URL = a2a_url(SECURITY_A2A_PORT)
QUALITY_A2A_URL = a2a_url(QUALITY_A2A_PORT)
TEST_A2A_URL = a2a_url(TEST_A2A_PORT)
FIX_A2A_URL = a2a_url(FIX_A2A_PORT)
