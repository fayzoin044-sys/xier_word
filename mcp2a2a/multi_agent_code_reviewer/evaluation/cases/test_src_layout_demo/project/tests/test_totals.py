from shop.totals import order_total


def test_order_total() -> None:
    assert order_total([10, 20, 5]) == 35
