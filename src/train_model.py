import argparse
import json
import tensorflow as tf

from models.transformer import BinaryClassificationTransformer
from features.preprocessing import load_dataset


# コマンドライン引数を取得
parser = argparse.ArgumentParser()

parser.add_argument('length', type=int)
parser.add_argument('num_words', type=int)
parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('hopping_num', type=int)
parser.add_argument('head_num', type=int)
parser.add_argument('hidden_dim', type=int)
parser.add_argument('dropout_rate', type=float)
parser.add_argument('lr', type=float)
parser.add_argument('val_threshold', type=float)

parser.add_argument('train_tfrecord_path', type=str)
parser.add_argument('test_tfrecord_path', type=str)
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('n_pos_neg_path', type=str)

args = parser.parse_args()


def main():
    # クラス重みを設定
    with open(args.n_pos_neg_path, 'r') as f:
        n_pos_neg = json.load(f)

    total = n_pos_neg['n_positive'] + n_pos_neg['n_negative']
    positive_weight = (1/n_pos_neg['n_positive']) * total / 2.0
    negative_weight = (1/n_pos_neg['n_negative']) * total / 2.0
    class_weight = {0: positive_weight, 1: negative_weight}

    train_ds = load_dataset(args.train_tfrecord_path,
                            batch_size=args.batch_size,
                            length=args.length+1)
    test_ds = load_dataset(args.test_tfrecord_path,
                            batch_size=args.batch_size,
                            length=args.length+1)

    # 学習
    model = create_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
                monitor='val_precision', mode='max', patience=5),
        tf.keras.callbacks.ModelCheckpoint(
                args.checkpoint_path, monitor='val_precision',
                mode='max', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_precision', mode='max',
                factor=0.2, patience=3)
    ]
    model.fit(x=train_ds,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=test_ds,
            callbacks=callbacks,
            shuffle=True,
            class_weight=class_weight,
            verbose=1)


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
                 loss='binary_crossentropy',
                 metrics=[tf.keras.metrics.Precision(
                            thresholds=args.val_threshold,
                            name='precision'),
                          tf.keras.metrics.Recall(
                            thresholds=args.val_threshold,
                            name='recall')])

    return model


if __name__ == '__main__':
    main()