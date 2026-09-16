"""User lookup containing one intentional SQL-injection risk."""

import sqlite3


def find_user(connection: sqlite3.Connection, username: str):
    cursor = connection.cursor()
    query = "SELECT name FROM users WHERE name = '" + username + "'"
    return cursor.execute(query).fetchone()
