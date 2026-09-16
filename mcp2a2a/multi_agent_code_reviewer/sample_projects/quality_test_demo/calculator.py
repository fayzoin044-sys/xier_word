"""Small demo with one intentional quality issue and one logic defect."""

import os


def add(left: int, right: int) -> int:
    return left + right


def subtract(left: int, right: int) -> int:
    # Intentional defect so Test Agent receives a real failing test.
    return left + right
