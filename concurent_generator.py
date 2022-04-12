from population import Population
from recurent import JustFinalRecurent, RecurentPerceptron
from utils import make_simple_structure


class ConcurentGenerator(RecurentPerceptron):
    def __init__(
        self, internal_layers_number: int, internal_layer_wide: int,
        control_couples_number: int,
    ):
        super().__init__(make_simple_structure(
                inputs_number=3,
                intermediate_layers_number=internal_layers_number,
                intermediate_layers_neurons_number=internal_layer_wide,
                outputs_number=2,
            ),
            control_couples_number,
        )
        self.critic = JustFinalRecurent(
            make_simple_structure(
                inputs_number=2,
                intermediate_layers_number=internal_layers_number,
                intermediate_layers_neurons_number=internal_layer_wide,
                outputs_number=1,
            ),
            control_couples_number,
        )

    def count_error(self, dataset, time_limit=None):
        cases_errors = list()
        for case_inputs, case_outputs in dataset:
            case_error = self.critic.get_outputs(
                inputs_values_list, time_limit=time_limit,
            )
            cases_errors.apprnd()
        self.error = max(cases_errors)
