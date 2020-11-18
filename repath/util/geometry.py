from typing import NamedTuple


class Point(NamedTuple):
    x: int
    y: int


class Address(NamedTuple):
    row: int
    col: int


class Size(NamedTuple):
    width: int
    height: int