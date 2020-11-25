from typing import NamedTuple


class Point(NamedTuple):
    x: int
    y: int


class PointF(NamedTuple):
    x: float
    y: float


class Address(NamedTuple):
    row: int
    col: int


class Size(NamedTuple):
    width: int
    height: int


class Shape(NamedTuple):
    num_rows: int
    num_cols: int
