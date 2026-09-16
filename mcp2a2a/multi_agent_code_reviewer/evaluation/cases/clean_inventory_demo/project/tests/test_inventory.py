import pytest
from inventory import in_stock, remaining


def test_inventory_helpers() -> None:
    assert in_stock(3) is True
    assert remaining(5, 2) == 3


def test_rejects_excess_purchase() -> None:
    with pytest.raises(ValueError):
        remaining(2, 3)
