from random import uniform

from perceptron import Perceptron
from utils import make_simple_structure, Scaler


def celsius_to_fahrenheit(celsius: float) -> float:
    return 1.8 * celsius + 32


def generate_dataset(items_number: int) -> list:
    dataset = list()
    for _number in range(items_number):
        celsius = uniform(-273, 1000)
        fahrenheit = celsius_to_fahrenheit(celsius)
        waited_output_value = Scaler(
            min_value=celsius_to_fahrenheit(-273),
            max_value=celsius_to_fahrenheit(1000),
        ).to_neuronet_format(fahrenheit)
        dataset.append(
            (
                (celsius,),
                (waited_output_value,),
            ),
        )
    return dataset


net = Perceptron(make_simple_structure(1, 6, 10, 1))

net = Perceptron.load_from_file(
    '/home/user/Desktop/My_folder/celsius_to_fahrenheit.ann',
)

net.tich_by_genetic(
    size=1000,
    dataset=generate_dataset(10000),
    ann_path='/home/user/Desktop/My_folder/celsius_to_fahrenheit.ann',
)
