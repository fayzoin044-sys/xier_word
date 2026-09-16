"""Pricing helper containing one intentional arithmetic defect."""


def discounted_price(price: float, percent: float) -> float:
    return price + price * percent / 100
