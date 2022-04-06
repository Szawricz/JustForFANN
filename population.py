from random import choice, uniform

from more_itertools import sort_together
from numpy import mean

from utils import PickleMixin, measure_execution_time, time_lenght_str


class Population(PickleMixin):
    def __init__(self, size: int, neuronet: object):
        neuronet_type = neuronet.__class__
        attrs = neuronet.essential_attrs
        self.neuronets = [neuronet_type(*attrs) for _item in range(size)]
        self.generations = int()


    def change_size_to(self, neuronets_number):
        if self.size > neuronets_number:
            self.neuronets = self.neuronets[:neuronets_number]
        elif self.size < neuronets_number:
            additional_neuronets = self.__class__(
                size = self.size - neuronets_number,
                neuronet = self.neuronets[0],
            ).neuronets
            self.neuronets += additional_neuronets


    @property
    def size(self) -> int:
        return len(self.neuronets)


    @property
    def errors(self) -> list:
        return [neuronet.error for neuronet in self.neuronets]


    @property
    def best_neuronet(self) -> object:
        return self.neuronets[self.errors.index(min(self.errors))]


    @staticmethod
    def cross_over(first_neuronet, second_neuronet, mutability) -> object:
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


    def sort_by_errors(self):
        self.neuronets = list(
            sort_together([self.errors, self.neuronets])[1],
        )


    @measure_execution_time
    def count_errors(self, dataset, time_limit=None):
        neuronets = list()
        for neuronet in self.neuronets:
            if not neuronet.error:
                neuronets.append(neuronet)
        for number, neuronet in enumerate(neuronets, start=1):
            percent = round(number * 100 / len(neuronets))
            print(f'\rprogress: {percent}%', end=' | ')
            neuronet.count_error(dataset, time_limit)


    def form_couples(self, couples_number) -> tuple:
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


    def print_chart_top(self, dataset, error, time_limit):
        print(
            f'population size: {self.size}',
            f'dataset lenght: {len(dataset)}',
            f'goal error: {error}',
            f'time limit per case: {time_limit} sec',
            sep=' | ',
        )
        print(74 * '=')


    def print_chart_line(self, time: float):
        print(f'\rgeneration: {self.generations} ', end=' | ')
        print(f'mean error: {mean(self.errors)}', end=' | ')
        print(f'best error: {self.best_neuronet.error}', end=' | ')
        print(f'counting time: {time_lenght_str(time)}', end=' | ')


    def save_with_population(self, ann_path, save_population: bool):
        if save_population:
            self.best_neuronet.population = self
        elif hasattr(self.best_neuronet, 'population'):
            del self.best_neuronet.population
        self.best_neuronet.save_to_file(ann_path)
        print('saved', end='')


    def enter_new_line(self):
        print('')


    def tich(
        self, dataset: list, mortality=0.4, error=0.25, mutability=0.2,
        time_limit=None, ann_path=None, save_population=False,
    ) -> object:

        self.print_chart_top(dataset, error, time_limit)

        while True:
            time = self.count_errors(dataset, time_limit)

            self.print_chart_line(time)

            if ann_path:
                self.save_with_population(ann_path, save_population)
            
            self.enter_new_line()

            if self.best_neuronet.error < error:
                return self.best_neuronet

            self.sort_by_errors()

            dead_nets_number = round(mortality * self.size)
            survived_nets_number = self.size - dead_nets_number

            self.change_size_to(survived_nets_number)
            couples = self.form_couples(dead_nets_number)
            children = [self.cross_over(*cple, mutability) for cple in couples]
            self.generations += 1
            self.neuronets += children
