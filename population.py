from math import ceil

from utils import cross_over, get_neuronets_sorted_by_succes, get_succeses


class Population:
    def __init__(self, size: int, neuronet_type: type, arguments: dict):
        self.neuronets = [neuronet_type(**arguments) for item in range(size)]
        self.successes = list()
        self.generations = int()

    @property
    def size(self) -> int:
        return len(self.neuronets)

    @property
    def best_neuronet(self) -> object:
        return self.neuronets[self.successes.index(max(self.successes))]

    def tich(
        self, dataset: list, fertility=2, success=0.75, mutability=0.1,
    ) -> object:
        while True:
            # Get successes:
            self.successes = get_succeses(self.neuronets, dataset)
            if max(self.successes) >= success:
                return self.best_neuronet
            # Sort neuronets by success:
            self.neuronets = get_neuronets_sorted_by_succes(
                self.successes, self.neuronets,
            )
            # Kill worst neuronets:
            mortality = fertility / (2 + fertility)
            dead_neuronets_number = ceil(self.size * mortality)
            winners_neuronets = self.neuronets[dead_neuronets_number:]
            # Get children number for couples:
            couples_number = ceil(dead_neuronets_number / fertility)
            last_couple_children_number = dead_neuronets_number % fertility
            couples_children_number = [
                fertility for n in range(couples_number)]
            if last_couple_children_number != 0:
                couples_children_number[0] = last_couple_children_number
            couples_members = list(reversed(winners_neuronets))[
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
                for child_number in range(children_number):
                    children.append(cross_over(first, second, mutability))
            self.generations += 1
            print(self.generations)
            print(max(self.successes))
            self.neuronets = winners_neuronets + children
