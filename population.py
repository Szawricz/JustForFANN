from random import choice, uniform
from multiprocessing import Process, cpu_count, Pool
from time import gmtime, strftime, time

from more_itertools import sort_together
from numpy import mean

from utils import PickleMixin


class Population(PickleMixin):
    def __init__(self, size: int, neuronet: object):
        neuronet_type = neuronet.__class__
        attrs = neuronet.essential_attrs
        self.neuronets = [neuronet_type(*attrs) for _item in range(size)]
        self.generations = int()

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
    def is_too_similar(first_neuronet, second_neuronet, similarity) -> bool:
        weights_similarity = list()
        first_weights = first_neuronet.all_weights
        second_weights = second_neuronet.all_weights
        for first, second in zip(first_weights, second_weights):
            if first.value == second.value:
                resoult = 1
            else:
                resoult = 0
            weights_similarity.append(resoult)
        return mean(weights_similarity) >= similarity

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

    @staticmethod
    def count_partial_errors(neuronets, dataset, time_limit=None):
        for neuronet in neuronets:
            neuronet.count_error(dataset, time_limit)

    def count_errors(self, dataset, time_limit=None):
        part_lenght = round(self.size / cpu_count())
        processes = list()
        for number in range(cpu_count()):
            start = number * part_lenght
            if number + 1 == cpu_count():
                stop = None
            else:
                stop = (number + 1) * part_lenght
            part = self.neuronets[start: stop]
            process = Process(
                target=self.count_partial_errors,
                args=(part, dataset),
                kwargs=dict(time_limit=time_limit),
            )
            process.start()
            process.join()


    def tich(
        self, dataset: list, mortality=0.4, error=0.25,
        similarity=0.9, mutability=0.1, time_limit=None, file_path=None,
    ) -> object:
        while True:
            start_time = time()
            self.count_errors(dataset, time_limit)
            finish_time = time()
            if file_path:
                self.save_to_file(file_path)
            print(f'generation: {self.generations}', end='    ')
            print(f'error: {self.best_neuronet.error}', end='    ')
            time_lenght = strftime('%X', gmtime(finish_time - start_time))
            print(f'timelenght: {time_lenght}')
            if self.best_neuronet.error < error:
                return self.best_neuronet
            self.sort_by_errors()
            dead_neuronets_number = round(mortality * self.size)
            full_population_size = self.size
            self.neuronets = self.neuronets[:self.size - dead_neuronets_number]
            children = list()
            generating = True
            while generating:
                for position, first_neuronet in enumerate(self.neuronets):
                    for second_neuronet in self.neuronets[position + 1:]:
                        if similarity != 1 and self.is_too_similar(
                            first_neuronet, second_neuronet, similarity,
                        ):
                            continue
                        else:
                            child = self.cross_over(
                                first_neuronet, second_neuronet, mutability,
                            )
                            children.append(child)
                            break
                    if len(children) + self.size == full_population_size:
                        generating = False
                        break
            self.generations += 1
            self.neuronets += children
