import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.under_sampling import RandomUnderSampler


class Vocab:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, texts):
        """ 単語のベクトル化の準備

        Arg:
            texts([str], ndarray): 単語のリスト
                e.g. ['VPTAPP',
                      'ATSQVP']

        Return:
            self(Vocab instance): 学習を完了したインスタンス

        """
        if type(texts).__module__ == 'numpy':
            texts = np.squeeze(texts)
            texts = texts.tolist()

        self.tokenizer.fit_on_texts(texts)
        return self

    def encode(self, texts):
        """ 単語のリストを整数のリストに変換する

        Arg:
            texts(list, ndarray): 単語のリスト

        Return:
            ndarray: shape=(n_samples, n_words)
                e.g. [[0, 1, 2, 3, 4, 4],
                      [3, 2, 5, 6, 0, 4]]

        """
        if type(texts).__module__ == 'numpy':
            texts = np.squeeze(texts)
            texts = texts.tolist()

        sequences = self.tokenizer.texts_to_sequences(texts)
        sequences = pad_sequences(sequences, padding='post', value=0)

        return sequences

    def decode(self, sequences, class_token=False):
        """ 整数のリストを単語のリストに変換する

        Arg:
            sequences(ndarray): 整数の配列
                shape=(n_samples, n_words)

        Return:
            [str]: 単語のリスト

        """
        if class_token:  # class_tokenを削除
            sequences = np.delete(sequences, 0, axis=-1)

        # ndarrayからlistに変換
        sequences = sequences.tolist()

        for i, seq in enumerate(sequences):
            try:  # 0が存在しない場合はValueError
                pad_idx = seq.index(0)
            except ValueError:
                continue

            sequences[i] = seq[:pad_idx]

        if class_token:
            for i, seq in enumerate(sequences):
                sequences[i] = list(map(lambda x: x-1, seq))

        texts = self.tokenizer.sequences_to_texts(sequences)
        texts = [text.replace(' ', '') for text in texts]

        return texts

    def _texts(self, sequences):
        return ['\t'.join(words) for words in sequences]


def pad_dataset(sequences, class_token=False):
    """ paddingを行う

    paddingの値は0，<CLS>トークンは1とする．

    Arg:
        sequences: list of int
        class_token: Trueの場合，Transformerでクラスタリングを行う際に
            用いる<CLS>トークンを先頭に追加する．
    Return:
        ndarray: shape=(len(sequences), max_len)
            class_token=True の時は，shape=(len(sequences), max_len + 1)
    """
    if class_token:
        for i, seq in enumerate(sequences):
            sequences[i] = list(map(lambda x: x+1, seq))

    # shape=(len(sequences), max_len)
    sequences = pad_sequences(sequences, padding='post', value=0)

    if class_token:  # class_tokenを追加
        # class_id = 1
        cls_arr = np.ones((len(sequences), 1))     # shape=(len(sequences), 1)
        sequences = np.hstack([cls_arr, sequences]).astype('int64')

    return sequences

def add_class_token(sequences):
    """ class token を先頭に付加する

    class token として1を用いる．

    Arg:
        sequences: list of int
            shape=(n_sequence, len)

    Return:
        ndarray: shape=(n_sequences, len + 1)
    """
    if isinstance(sequences, list):
        sequences = np.array(sequences)

    sequences += 1

    sequences = np.array(sequences)
    mask = (sequences == 1)
    sequences[mask] = 0

    # class_token = 1
    cls_arr = np.ones((len(sequences), 1))     # shape=(len(sequences), 1)
    sequences = np.hstack([cls_arr, sequences]).astype('int64')

    return sequences

def load_dataset(filename, batch_size, length, buffer_size=1000):
    def dict2tuple(feat):
        return feat["x"], feat["y"]

    dataset = tf.data.TFRecordDataset(filenames=filename) \
        .shuffle(buffer_size) \
        .batch(batch_size) \
        .apply(
            tf.data.experimental.parse_example_dataset({
                "x": tf.io.FixedLenFeature([length], dtype=tf.int64),
                "y": tf.io.FixedLenFeature([1], dtype=tf.int64)
            })).map(dict2tuple)

    return dataset

def underSampling(x, y, sampling_strategy=1.0, random_state=0):
    """ アンダーサンプリングを行う

    Args:
        x(ndarray): shape=(n_samples, n_features)
        y(ndarray): shape=(n_samples,)
        random_state(int): シード値

    Return:
        x(ndarray): shape=(n_samples_new, n_features)
        y(ndarray): shape=(n_samples_new,)

    """
    rus = RandomUnderSampler(random_state=random_state,
            sampling_strategy=sampling_strategy)
    x_resampled, y_resampled = rus.fit_resample(x, y)

    return x_resampled, y_resampled
