"""Network helper containing one intentional shell-injection risk."""

import subprocess


def ping_host(host: str) -> int:
    result = subprocess.run(
        f"ping -n 1 {host}",
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode
