import os
import argparse
import json
import tensorflow as tf
import pickle

from features.preprocessing import Vocab


# コマンドライン引数を取得
parser = argparse.ArgumentParser()

parser.add_argument('dataset_dir', type=str)
parser.add_argument('vocab_path', type=str)
parser.add_argument('num_words', type=int)

args = parser.parse_args()

def main():
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=args.num_words,
                oov_token='<UNK>',
                filters='',
                lower=False,
                split='\t',
                char_level=True
    )

    vocab = Vocab(tokenizer)

    json_path = 'references/PTAP_data.json'
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    for data in json_data:
        virusname = data['virus'].replace(' ', '_')

        dataset_path = os.path.join(args.dataset_dir, f'{virusname}.pickle')
        with open(dataset_path, 'rb') as f:
            x = pickle.load(f)

        vocab.fit(x)

    tokenizer = vocab.tokenizer
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(tokenizer, f)


if __name__ == '__main__':
    main()