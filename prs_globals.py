from enum import Enum

class PatternType(Enum):
    Base = 1
    Composite = 2

class Shape(Enum):
    Acyclic = 1
    Cyclic = 2
