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
    net = Perceptron.load_from_file('/home/user/Desktop/My_folder/xor0.15.ann')
    # net = Perceptron([2, 3, 1])
    net = net.tich_by_genetic(dataset, size=100000, error=0.19)
    net.save_to_file('/home/user/Desktop/My_folder/xor0.15.ann')
