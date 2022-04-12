from perceptron import Perceptron


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
    net = Perceptron([2, 3, 1])
    # net = Perceptron.load_from_file(
    #     '/home/user/Desktop/My_folder/xor0.15.ann',
    # )
    net = net.tich_by_genetic(dataset, size=100000, error=0.19)
