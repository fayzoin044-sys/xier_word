"""Small text helpers with no intentional defects."""


def normalize_spaces(value: str) -> str:
    return " ".join(value.split())


def greeting(name: str) -> str:
    return f"你好，{name}"
