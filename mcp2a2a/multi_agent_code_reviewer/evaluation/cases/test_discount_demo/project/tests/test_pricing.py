from pricing import discounted_price


def test_discounted_price() -> None:
    assert discounted_price(100, 20) == 80
