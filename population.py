from functools import wraps
from random import choice, uniform
from time import time

from more_itertools import sort_together
from numpy import mean

from utils import measure_execution_time, pickling, time_lenght_str


def _print_spases_line():
    print('\r', 78 * ' ', end='')


def with_current_process_print(string_to_print: str):
    def decorator(function):
        @wraps(function)
        def _wrapper(*args, **kwargs):
            print(f'\r{string_to_print} ', end=' | ')
            resoult = function(*args, **kwargs)
            _print_spases_line()
            return resoult
        return _wrapper
    return decorator


@pickling
class Population:
    def __init__(self, size: int, neuronet: object):
        neuronet_type = neuronet.__class__
        attrs = neuronet.essential_attrs
        self.neuronets = [neuronet_type(*attrs) for _item in range(size)]
        self.generations = int()

    def tich(
        self, dataset: list, mortality=0.4, error=0.25, mutability=0.2,
        time_limit=None, ann_path=None, save_population=False,
    ) -> object:

        self._print_chart_top(dataset, error, time_limit)
        start_time = time()

        while True:
            self._count_errors(dataset, time_limit)

            if ann_path:
                self._save_best_neuronet(ann_path, save_population)

            finish_time = time()
            resoult_time = finish_time - start_time
            start_time = time()

            self._print_chart_line(resoult_time, ann_path, save_population)

            if self.best_neuronet.error < error:
                return self.best_neuronet

            self._sort_by_errors()

            dead_nets_number = round(mortality * self.size)
            survived_nets_number = self.size - dead_nets_number

            self.change_size_to(survived_nets_number)
            couples = self._form_couples(dead_nets_number)
            children = [self._cross_over(*cple, mutability) for cple in couples]
            self.generations += 1
            self.neuronets += children

    def change_size_to(self, neuronets_number):
        if self.size > neuronets_number:
            self.neuronets = self.neuronets[:neuronets_number]
        elif self.size < neuronets_number:
            size = self.size - neuronets_number
            neuronet = self.neuronets[0]
            additional_neuronets = self.__class__(size, neuronet).neuronets
            self.neuronets += additional_neuronets

    @property
    def size(self) -> int:
        return len(self.neuronets)

    @property
    def best_neuronet(self) -> object:
        return self.neuronets[self._errors.index(min(self._errors))]

    @property
    def _errors(self) -> list:
        return [neuronet.error for neuronet in self.neuronets]

    @with_current_process_print('breeding:')
    def _cross_over(
        self, first_neuronet, second_neuronet, mutability: float,
    ) -> object:
        compliment_weights_couples = zip(
            first_neuronet.all_weights, second_neuronet.all_weights,
        )
        child_raw_weights = list()
        for first_weight, second_weight in compliment_weights_couples:
            if choice([True, False]):
                child_raw_weight = first_weight.value
            else:
                child_raw_weight = second_weight.value
            if uniform(0, 1) < mutability:
                child_raw_weight = first_weight.value_generator()
            child_raw_weights.append(child_raw_weight)
        return first_neuronet.__class__.init_from_weights(
            child_raw_weights, first_neuronet.essential_attrs,
        )

    def _sort_by_errors(self):
        self.neuronets = list(
            sort_together([self._errors, self.neuronets])[1],
        )

    def _print_percent(self, name, number, sequence):
        percent = round(number * 100 / len(sequence))
        print(f'\r{name} {percent}%', end=' | ')

    @measure_execution_time
    def _count_errors(self, dataset, time_limit=None):
        neuronets = list()
        for number, neuronet in enumerate(self.neuronets):
            self._print_percent('prepare:', number, self.neuronets)
            if not neuronet.error:
                neuronets.append(neuronet)
        for number, neuronet in enumerate(neuronets, start=1):
            self._print_percent('rprogress:', number, neuronets)
            neuronet.count_error(dataset, time_limit)
        _print_spases_line()

    @with_current_process_print('forming:')
    def _form_couples(self, couples_number) -> tuple:
        first_part = self.neuronets[::2]
        second_part = self.neuronets[1::2]
        half_lenght = int(self.size // 2)
        fuull_passes_number = int(couples_number // half_lenght)
        last_couples_number = int(couples_number % half_lenght)
        number = 1
        for first_neuronet, second_neuronet in zip(first_part, second_part):
            if (not fuull_passes_number) and (number > last_couples_number):
                break
            for _number in range(fuull_passes_number):
                yield (first_neuronet, second_neuronet)
            if number <= last_couples_number:
                yield (first_neuronet, second_neuronet)
            number += 1

    def _print_chart_top(self, dataset, error, time_limit):
        print(
            f'population size: {self.size}',
            f'dataset lenght: {len(dataset)}',
            f'goal error: {error}',
            f'time limit per case: {time_limit} sec',
            sep=' | ',
        )
        print(79 * '=')

    def _print_chart_line(
        self, time: float, ann_path: str, save_population: bool,
    ):
        print(f'\rgeneration: {self.generations} ', end=' | ')
        print(f'mean error: {mean(self._errors)}', end=' | ')
        print(f'best error: {self.best_neuronet.error}', end=' | ')
        print(f'counting time: {time_lenght_str(time)}', end=' | ')
        if ann_path:
            print('saved', end=' ')
        if save_population:
            print('with population', end=' ')
        print('')

    @with_current_process_print('saving:')
    def _save_best_neuronet(self, ann_path: str, with_population: bool):
        if with_population:
            self.best_neuronet.population = self
        self.best_neuronet.save_to_pickle(ann_path)
        if with_population:
            del self.best_neuronet.population
