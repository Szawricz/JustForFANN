from recurent import RecurentPerceptron
from utils import make_simple_structure


class ChatBot(RecurentPerceptron):
    @staticmethod
    def _string_to_numbers_list(string: str) -> list:
        return [ord(character) for character in string]

    @staticmethod
    def _numbers_list_to_string(numbers_list: list, unicode_max=65535) -> str:
        chars_list = list()
        for number in numbers_list:
            chars_list.append(chr(round((number + 1) / 2 * unicode_max)))
        return str().join(chars_list)

    def get_answer(self, request: str) -> str:
        numbers_list = self._string_to_numbers_list(request)
        inputs_list = [[number] for number in numbers_list]
        outputs_list = self.get_outputs(inputs_list)
        numbers_list = [number.pop() for number in outputs_list]
        return self._numbers_list_to_string(numbers_list)


def chat_with_bot():
    chat_bot = ChatBot(
        structure=make_simple_structure(
            inputs_number=1,
            intermediate_layers_number=7,
            intermediate_layers_neurons_number=100,
            outputs_number=1,
        ),
        control_couples_number=50,
    )
    while True:
        answer = chat_bot.get_answer(input('You: '))
        print(f'\nBot: {answer}\n')


if __name__ == '__main__':
    chat_with_bot()
