"""Name formatter containing one intentional unused variable."""


def full_name(first: str, last: str) -> str:
    result = f"{first} {last}"
    debug_message = f"formatted={result}"
    return result
