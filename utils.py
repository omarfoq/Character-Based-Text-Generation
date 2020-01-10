import os
from urllib.request import urlopen
import numpy as np


def download_data(path, t_url):
    """
    :param path: Path to which save the data
    :param t_url: url of the text data
    :return:
    """
    response = urlopen(t_url)
    data = response.read()
    txt_str = str(data)
    lines = txt_str.split("\\n")
    des_url = os.path.join(path, 'shakespeare.txt')
    fx = open(des_url, "w")
    for line in lines:
        fx.write(line + "\n")
    fx.close()


def read_and_preprocess_text(path_to_file):
    """

    :param path_to_file: Path to text file
    :return:
            text_as_int :numpy array of type int encoding the text
            char2idx: dict object mapping characters to indexes
            idx2char : dict object mapping indexes to characters
    """
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    print('Length of text: {} characters'.format(len(text)))

    vocab = sorted(set(text))
    vocab_size = len(vocab)
    print('{} unique characters'.format(vocab_size))

    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    return text_as_int, char2idx, idx2char

