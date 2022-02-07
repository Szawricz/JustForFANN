from math import ceil

from more_itertools import sort_together

from utils import cross_over


class Population:
    def __init__(self, size: int, neuronet: object):
        neuronet_type = neuronet.__class__
        structure = neuronet.structure
        self.neuronets = [neuronet_type(structure) for _item in range(size)]
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


    def get_dead_neuronets_number(self, fertility: int) -> int:
        return ceil(self.size * fertility / (2 + fertility))

    def sort_by_errors(self):
        self.neuronets = list(
            sort_together([self.errors, self.neuronets], reverse=True,)[1],
        )

    def count_errors(self, dataset):
        for neuronet in self.neuronets:
            neuronet.count_error(dataset)
    

    def tich(
        self, dataset: list, fertility: int, error: float, mutability: float,
    ) -> object:
        while True:
            self.count_errors(dataset)
            if self.best_neuronet.error < error:
                return self.best_neuronet
            self.sort_by_errors()
            dead_neuronets_number = self.get_dead_neuronets_number(fertility)
            # Kill worst neuronets:
            self.neuronets = self.neuronets[dead_neuronets_number:]
            # Get children number for couples:
            couples_number = ceil(dead_neuronets_number / fertility)
            last_couple_children_number = dead_neuronets_number % fertility
            couples_children_number = [
                fertility for _n in range(couples_number)]
            if last_couple_children_number != 0:
                couples_children_number[0] = last_couple_children_number
            couples_members = list(reversed(self.neuronets))[
                :couples_number * 2]
            couples_and_child_numbers = list(
                zip(
                    couples_members[::2],
                    couples_members[1::2],
                    couples_children_number,
                )
            )
            # Crossingover
            children = list()
            for first, second, children_number in couples_and_child_numbers:
                for _child_number in range(children_number):
                    children.append(cross_over(first, second, mutability))
            self.generations += 1
            print(self.generations)
            print(self.best_neuronet.error)
            self.neuronets += children
