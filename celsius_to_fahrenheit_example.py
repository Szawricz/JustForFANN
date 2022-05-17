from random import uniform

from perceptron import Perceptron
from utils import make_simple_structure, Scaler


celsius_scaler = Scaler(min_value=-273, max_value=1000)


def celsius_to_fahrenheit(celsius: float) -> float:
    return 1.8 * celsius + 32


fahrenheit_scaler = Scaler(
            min_value=celsius_to_fahrenheit(-273),
            max_value=celsius_to_fahrenheit(1000),
        )


def generate_dataset() -> list:
    dataset = list()
    for celsius in range(-273, 1000):
        fahrenheit = celsius_to_fahrenheit(celsius)
        input_value = celsius_scaler.to_neuronet_format(celsius)
        waited_output_value = fahrenheit_scaler.to_neuronet_format(fahrenheit)
        dataset.append(
            (
                (input_value,),
                (waited_output_value,),
            ),
        )
    return dataset


if __name__ == '__main__':
    net = Perceptron(make_simple_structure(1, 2, 100, 1))

    net = Perceptron.load_from_file(
        '/home/user/Desktop/My_folder/celsius_to_fahrenheit.ann',
    )

    net.tich_by_genetic(
        size=100,
        dataset=generate_dataset(),
        ann_path='/home/user/Desktop/My_folder/celsius_to_fahrenheit.ann',
    )

    while True:
        inputs = celsius_scaler\
            .to_neuronet_format(float(input(prompt='celsius: ')))
        inputs = [inputs]
        print(
            'fahrenheit:',
            fahrenheit_scaler.from_neuronet_format(net.get_outputs(inputs)[0]),
        )
