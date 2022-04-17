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
    net = Perceptron.load_from_pickle(
        '/home/user/Desktop/My_folder/xor0.15.ann',
    )
    net = net.tich_by_genetic(
        dataset,
        size=10**5,
        error=0.19,
        ann_path='/home/user/Desktop/My_folder/xor0.15.ann',
        save_population=True,
    )
