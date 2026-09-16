"""Inventory helpers with no intentional defects."""


def in_stock(quantity: int) -> bool:
    return quantity > 0


def remaining(quantity: int, purchased: int) -> int:
    if purchased > quantity:
        raise ValueError("purchased quantity exceeds stock")
    return quantity - purchased
