from numpy import sign

from perceptron import Perceptron
from population import Population
from utils import generate_sign

pop = Population(
    10000,
    Perceptron,
    dict(
        structure=[2, 3, 1],
        transmition_function=sign,
        value_generator=generate_sign,
        сalibration_functions=[round],
    ),
)
dataset = (
    (
        (0, 0),
        (0,),
    ),
    (
        (0, 1),
        (1,),
    ),
    (
        (1, 0),
        (1,),
    ),
    (
        (1, 1),
        (0,),
    ),
)
print(pop.tich(dataset, mutability=0.02).get_outputs([0, 1]))
