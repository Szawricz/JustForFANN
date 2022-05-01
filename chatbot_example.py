from string import ascii_uppercase

from generative import ChatBot
from utils import make_simple_structure


chat_bot = ChatBot(
    make_simple_structure(1, 10, 100, 1),
    control_couples_number=10,
)

chat_bot = ChatBot.load_from_file('/home/user/Desktop/My_folder/brain.ann')

dataset = [
    ['Hey!', 'Hallo!'],
    ['0 + 0 = ?', '0'],
    ['0 + 1 = ?', '1'],
    ['Say the letters of English alphabet, please.', ascii_uppercase],
    ['Could you answer nothing?', ''],
    ['Repeat please word "cat"', 'cat. What is cat?'],
    ['Cat is such animal', 'Thanks.'],
    ['What is cat?', 'Cat is such animal'],
    ['1 + 0 = ?', '1'],
]


if __name__ == '__main__':
    chat_bot = chat_bot.tich_by_genetic(
        dataset,
        size=20,
        time_limit=60,
        ann_path='/home/user/Desktop/My_folder/brain.ann',
        save_population=True,
    )

    while True:
        answer = chat_bot.get_outputs(input('You: '), time_limit=10)
        print(f'\nBot: {answer}\n')
