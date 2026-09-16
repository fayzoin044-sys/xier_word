from text_utils import greeting, normalize_spaces


def test_normalize_spaces() -> None:
    assert normalize_spaces("hello   world") == "hello world"


def test_chinese_greeting() -> None:
    assert greeting("小明") == "你好，小明"
