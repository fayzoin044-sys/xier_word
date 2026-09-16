"""Order total helper containing one intentional arithmetic defect."""


def order_total(prices: list[int]) -> int:
    return sum(prices) + 1
