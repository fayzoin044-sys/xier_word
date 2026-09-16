"""Intentionally vulnerable demo. Do not copy these patterns into real code."""

import sqlite3
import subprocess


# Intentionally hard-coded credential for static-analysis demonstration only.
DATABASE_PASSWORD = "SuperSecretDatabasePassword123!"


def lookup_host(host: str) -> None:
    """Vulnerable: untrusted input reaches a shell command."""
    subprocess.run(f"nslookup {host}", shell=True, check=False)


def find_user(username: str) -> list[tuple]:
    """Vulnerable: untrusted input is concatenated into a SQL statement."""
    connection = sqlite3.connect("demo.db")
    cursor = connection.cursor()
    query = "SELECT id, username FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchall()


if __name__ == "__main__":
    lookup_host(input("Host to look up: "))
    print(find_user(input("Username: ")))

