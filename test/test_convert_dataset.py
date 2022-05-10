import unittest
import numpy as np
import pickle

from src.features.preprocessing import Vocab, load_dataset


length = 24
batch_size = 5

vocab_path = 'references/vocab.pickle'


class TestConvertDataset(unittest.TestCase):
    def test_convert_dataset(self):
        train_tfrecord_path = 'data/tfrecord/train_dataset.tfrecord'
        test_tfrecord_path = 'data/tfrecord/test_dataset.tfrecord'

        train_ds = load_dataset(train_tfrecord_path,
                                batch_size=batch_size,
                                length=length+1)
        test_ds = load_dataset(test_tfrecord_path,
                                batch_size=batch_size,
                                length=length+1)

        with open(vocab_path, 'rb') as f:
            tokenizer = pickle.load(f)
        vocab = Vocab(tokenizer)

        for x, y in train_ds:
            x, y = x.numpy(), y.numpy()
            y = np.squeeze(y)

            x = x[y == 1]

            x = vocab.decode(x, class_token=True)
            print(x)
