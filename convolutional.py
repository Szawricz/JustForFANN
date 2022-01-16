from perceptron import Perceptron
from utils import make_simple_structure


class Convolutional:
    def __init__(self, shape_wide: int, pading=False):
        self.perceptrons = list()
        for number in range(shape_wide):
            if number == 0:
                Perceptron(make_simple_structure(4, 1, 1))*
            self.perceptrons.append(
                Perceptron(make_simple_structure(4, 1, 1)),
            )
--------+