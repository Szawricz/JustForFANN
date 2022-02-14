from perceptron import Perceptron
from utils import make_simple_structure

# XOR problem example:
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

if __name__ == '__main__':
    net = Perceptron(make_simple_structure(2, 1, 3, 1))\
        .tich_by_genetic(dataset, size=100)\
            .weights_number
    print(f'>> {(net)}')
