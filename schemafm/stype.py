from enum import Enum


class Stype(str, Enum):
    numerical = 'numerical'
    categorical = 'categorical'
