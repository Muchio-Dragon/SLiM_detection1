import argparse
import numpy as np
import tensorflow as tf
import pickle

from models.transformer import BinaryClassificationTransformer
from scipy.optimize import minimize
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from features.preprocessing import Vocab
from utils import calc_f_beta_score


# コマンドライン引数を取得
parser = argparse.ArgumentParser()

parser.add_argument('length', type=int)
parser.add_argument('batch_size', type=int)
parser.add_argument('num_words', type=int)
parser.add_argument('hopping_num', type=int)
parser.add_argument('head_num', type=int)
parser.add_argument('hidden_dim', type=int)
parser.add_argument('dropout_rate', type=float)
parser.add_argument('lr', type=float)
parser.add_argument('beta', type=float)

parser.add_argument('checkpoint_path', type=str)
parser.add_argument('eval_ds_path', type=str)
parser.add_argument('vocab_path', type=str)
parser.add_argument('result_path', type=str)
parser.add_argument('false_positive_path', type=str)

args = parser.parse_args()


def main():
    model = create_model()
    model.load_weights(args.checkpoint_path)

    with open(args.eval_ds_path, 'rb') as f:
        x_eval = pickle.load(f)
        y_eval = pickle.load(f)

    y_pred = model.predict(x_eval)
    y_pred = np.squeeze(y_pred)

    # 閾値を最適化
    def f_beta_opt(threshold):
        precision = precision_score(y_eval, y_pred >= threshold)
        recall = recall_score(y_eval, y_pred >= threshold)
        return -calc_f_beta_score(precision, recall, beta=args.beta)

    result = minimize(f_beta_opt, x0=np.array([0.5]), method='Nelder-Mead')
    best_threshold = result['x'].item()

    y_pred = model.predict(x_eval)
    y_pred = np.squeeze(y_pred)
    y_pred = y_pred >= best_threshold

    with open(args.result_path, 'w') as f:
        print(f'best threshold: {best_threshold}', file=f)

        cm = confusion_matrix(y_eval, y_pred)
        print('confusion matrix:')
        print(cm, file=f)

        precision = precision_score(y_eval, y_pred)
        recall = recall_score(y_eval, y_pred)
        print(f'precision: {precision}', file=f)
        print(f'recall: {recall}', file=f)

    with open(args.vocab_path, 'rb') as f:
        tokenizer = pickle.load(f)
    vocab = Vocab(tokenizer)

    # 偽陽性データを保存
    x_fp = x_eval[(y_pred == 1) & (y_eval == 0)]
    x_fp = vocab.decode(x_fp, class_token=True)
    with open(args.false_positive_path, 'w') as f:
        for x in x_fp:
            print(x, file=f)


def create_model():
    """ モデルを定義する """
    model = BinaryClassificationTransformer(
                vocab_size=args.num_words,
                hopping_num=args.hopping_num,
                head_num=args.head_num,
                hidden_dim=args.hidden_dim,
                dropout_rate=args.dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(
                                learning_rate=args.lr),
                 loss='binary_crossentropy')

    return model


if __name__ == '__main__':
    main()
