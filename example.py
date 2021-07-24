from perceptron import Perceptron
from utils import make_simple_structure

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

net = Perceptron(
    make_simple_structure(2, 1, 1),
).tich_by_genetic(dataset=dataset, size=10000)

print(net.get_outputs([0, 0]))
