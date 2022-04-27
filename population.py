from random import choice, uniform
from time import time

from more_itertools import sort_together
from numpy import mean

from utils import (print_percent, print_spases_line, split_by_evenodd_position,
                   time_lenght_str, with_current_process_print,
                   with_start_and_finish_time_print)


class Population:
    def __init__(self, size: int, neuronet: object):
        self.neuronet_example = neuronet
        self.neuronets_type = self.neuronet_example.__class__

        self.neuronets = list()
        for _item in range(size):
            neuronet = self.neuronet_example.copy_with_random_weights()
            self.neuronets.append(neuronet)

        self.generations = int(0)

    @with_start_and_finish_time_print()
    def tich(
        self, dataset: list, mortality=0.4, error=0.25, mutability=0.2,
        time_limit=None, ann_path=None, save_population=False,
    ) -> object:
        # print the popuilation tichg parameters in chart top
        self._print_chart_top(dataset, error, time_limit)

        # start the time measuring for the ancestors population
        start_time = time()

        while True:
            # count errors and sort neuronets by the errors
            self._count_errors(dataset, time_limit)

            # save tiched neuronet to file
            if ann_path:
                self._save_best_neuronet(ann_path, save_population)

            # time counting finishing
            finish_time = time()

            # output resoults of generation's tiching to console
            resoult_time = finish_time - start_time
            self._print_chart_line(resoult_time, ann_path, save_population)

            # stop the tiching and return resouult if goal arror reached
            if self.best_neuronet.error < error:
                return self.best_neuronet

            # time counting starting
            start_time = time()

            # Killing worst neuronets
            dead_nets_number = round(mortality * self.size)
            survived_nets_number = self.size - dead_nets_number
            self.change_size_to(survived_nets_number)

            # breeding
            couples = self._form_couples(dead_nets_number)
            children = self._make_children(couples, mutability)

            # children adding to population
            self.neuronets += children
            self.generations += 1

    def change_size_to(self, neuronets_number):
        if self.size > neuronets_number:
            self.neuronets = self.neuronets[:neuronets_number]

        if self.size < neuronets_number:
            size = neuronets_number - self.size
            neuronet = self.neuronet_example
            additional_neuronets = self.__class__(size, neuronet).neuronets
            self.neuronets.extend(additional_neuronets)

    @property
    def size(self) -> int:
        return len(self.neuronets)

    @property
    def best_neuronet(self) -> object:
        # best resoult is first element in sorted neuronets list 
        return self.neuronets[0]

    @property
    def _errors(self) -> list:
        return [neuronet.error for neuronet in self.neuronets]

    def _cross_over(self, neuronet_1, neuronet_2, mutability: float):
        child_raw_weights = list()
        for couple in zip(neuronet_1.all_weights, neuronet_2.all_weights):
            child_raw_weight = choice(couple).value
            if uniform(0, 1) < mutability:
                child_raw_weight = couple[0].value_generator()
            child_raw_weights.append(child_raw_weight)
        return self.neuronet_example.copy_with_new_weights(child_raw_weights)

    @with_current_process_print('saving...')
    def _save_best_neuronet(self, ann_path: str, with_population: bool):
        # bound population to the best neuronet will be saved to file
        if with_population:
            self.best_neuronet.population = self

        # save neuronet by "save_to_file" method provided by pickling mixin
        self.best_neuronet.save_to_file(ann_path)

        # delete the binded population to prevent a recursion 
        if with_population:
            del self.best_neuronet.population

    def _make_children(self, couples: list, mutability: float) -> list:
        children = list()
        couples = list(couples)
        for number, couple in enumerate(couples, start=1):
            print_percent('breeding:', number, couples)
            children.append(self._cross_over(*couple, mutability))
        print_spases_line()
        return children

    def _sort_by_errors(self):
        self.neuronets = list(sort_together([self._errors, self.neuronets])[1])

    @property
    def _neuronets_without_counted_error(self):
        neuronets = list()
        for number, neuronet in enumerate(self.neuronets, start=1):
            print_percent('neuronets preparing:', number, self.neuronets)
            if not neuronet.error:
                neuronets.append(neuronet)
        print_spases_line()
        return neuronets

    def _count_errors(self, dataset, time_limit=None):
        if self.neuronet_example.recurent:
            neuronets = self.neuronets
        else:
            neuronets = self._neuronets_without_counted_error

        for number, neuronet in enumerate(neuronets, start=1):
            print_percent('errors counting:', number, neuronets)
            neuronet.count_error(dataset, time_limit)
            print_spases_line()

        # sort all population neuronrts after its errors counting
        self._sort_by_errors()

    @with_current_process_print('forming...')
    def _form_couples(self, couples_number) -> tuple:
        parts = split_by_evenodd_position(self.neuronets)

        half_lenght = int(self.size // 2)
        fuull_passes_number = int(couples_number // half_lenght)
        last_couples_number = int(couples_number % half_lenght)

        for number, couple in enumerate(zip(parts), start=1):
            if (not fuull_passes_number) and (number > last_couples_number):
                break
            for _number in range(fuull_passes_number):
                yield couple
            if number <= last_couples_number:
                yield couple

    def _print_chart_top(self, dataset, error, time_limit):
        print(
            f'| population size: {self.size}',
            f'dataset lenght: {len(dataset)}',
            f'goal error: {error}',
            f'time limit per case: {time_limit} sec',
            sep=' | ',
        )
        print(79 * '=')

    def _print_chart_line(
        self, time: float, ann_path: str, save_population: bool,
        errors_lenght=19, gen_number_lenght=4,
    ):
        best_error = str(self.best_neuronet.error).ljust(errors_lenght, '0')
        mean_error = str(mean(self._errors)).ljust(errors_lenght, '0')
        generation_number = str(self.generations).rjust(gen_number_lenght)

        print(f'\rgeneration: {generation_number}', end=' | ')
        print(f'mean error: {mean_error}', end=' | ')
        print(f'best error: {best_error}', end=' | ')
        print(f'counting time: {time_lenght_str(time)}', end=' | ')
        if ann_path:
            print('saved', end=' ')
        if save_population:
            print('with population', end=' ')
        print('')
