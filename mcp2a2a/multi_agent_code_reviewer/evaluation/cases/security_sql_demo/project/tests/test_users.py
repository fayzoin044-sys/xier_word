import sqlite3

from users import find_user


def test_find_user() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE users (name TEXT)")
    connection.execute("INSERT INTO users VALUES (?)", ("alice",))
    assert find_user(connection, "alice") == ("alice",)
