from labels import label


def test_label() -> None:
    assert label("ready") == "READY"
