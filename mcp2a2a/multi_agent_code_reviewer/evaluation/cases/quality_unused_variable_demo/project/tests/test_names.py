from names import full_name


def test_full_name() -> None:
    assert full_name("Ada", "Lovelace") == "Ada Lovelace"
