from generative import ChatBot
from utils import make_simple_structure


# chat_bot = ChatBot.load_from_file(
#     '/home/user/Desktop/My_folder/brain.ann',
# )


chat_bot = ChatBot(
    make_simple_structure(1, 10, 100, 1),
    control_couples_number=10,
)


dataset = [
    ['Hey!', 'Hallo!'],
    ['0 + 0 = ?', '0'],
    ['0 + 1 = ?', '1'],
    [
        'Say the letters of English alphabet, please.',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    ],
    ['Could you answer nothing?', ''],
    ['Repeat please word "cat"', 'cat. What is cat?'],
    ['Cat is such animal', 'Thanks.'],
    ['What is cat?', 'Cat is such animal'],
    ['1 + 0 = ?', '1'],
]


if __name__ == '__main__':
    chat_bot = chat_bot.tich_by_genetic(dataset, size=10, time_limit=60)

    while True:
        answer = chat_bot.get_outputs(input('You: '), time_limit=10)
        print(f'\nBot: {answer}\n')
