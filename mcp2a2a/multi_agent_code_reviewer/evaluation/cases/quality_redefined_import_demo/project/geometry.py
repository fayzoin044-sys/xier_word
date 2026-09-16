"""Geometry helper containing one intentional duplicate import."""

from math import sqrt
from math import sqrt


def hypotenuse(left: float, right: float) -> float:
    return sqrt(left * left + right * right)
