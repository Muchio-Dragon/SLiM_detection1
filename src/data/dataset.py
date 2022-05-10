import random
import numpy as np
from Bio import SeqIO
from argparse import ArgumentError

from ..utils import determine_protein_name


class Dataset:

    def __init__(self, SLiM, idx, length=10, proteins=None, neighbor=None,
                 SLiM_proteins=None, remove_X=True, replacement_tolerance=1,
                 threshold=None, random_split=False, n_gram=False):
        self.SLiM = SLiM            # str: アノテーションするSLiM配列
        self.idx = idx              # int: SLiMの開始位置
        self.length = length        # int: 断片の長さ

        self.neighbor = neighbor    # int: SLiMを探索する範囲．
                                    # 	SLiMの位置±neighborの範囲で
                                    #   LiMを探索する．
                                    #   Noneの場合は範囲を無視して探索する．

        self.SLiM_proteins = SLiM_proteins   # [str]: SLiMを持つレコードの
                                             #        descriptionに
                                             #        含まれる文字列

        self.proteins = proteins

        self.remove_X = remove_X    # bool: 未知アミノ酸Xが含まれる
                                    #       配列を無視する．

        # int: 何個までアミノ酸置換を許容するか
        self.replacement_tolerance = replacement_tolerance

        # int: 断片にいくつSLiMが含まれていたら陽性とアノテーションするか
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = len(self.SLiM)

        # bool: Trueのとき，ランダムな位置から分割を開始する
        self.random_split = random_split

        # bool: Trueのとき，n-gramのデータセットを作成する
        self.n_gram = n_gram

    def make_dataset(self, records, dict=False):
        """ レコードのリストからアミノ酸配列・アノテーションリストを取得

        Arg:
            records(list of record): データセットの基となるレコード
            dict(bool): Trueの場合，タンパク質種ごとの辞書として
                結果を出力する．

        Return:
            if dict == False:
                x(ndarray): アミノ酸配列, shape=(n_samples, 1)
                y(ndarray): アノテーションデータ, shape=(n_samples,)
            if dict == True:
                tupple of dict: {protein_name: (sequences, labels)}
                    protein_name: str
                    sequences: ndarray, shape=(n_samples, length)
                    labels: ndarray, shape=(n_samples)

        """
        while (True):
            datasets = self._make_dataset_dict(records)

            for key in datasets.keys():
                has_SLiM = key in self.SLiM_proteins

                if has_SLiM:
                    finish = True
                    x, y = datasets[key]
                    if ((y == 1).sum() == 0):
                        finish = False

            if finish:
                break

        if dict:
            return datasets

        else:
            xs, ys = [], []
            for key in datasets.keys():
                x, y = datasets[key]
                xs.append(x)
                ys.append(y)

            xs = np.vstack(xs)
            ys = np.hstack(ys)
            return xs, ys

    def _make_dataset_dict(self, records):
        seq_dict = {}
        label_dict = {}
        for key in self.proteins.keys():
            seq_dict[key] = []
            label_dict[key] = []

        for record in records:
            seq = record.seq

            if self.remove_X:
                if 'X' in record.seq:
                    continue

            # タンパク質の種別を判別する
            protein_name = determine_protein_name(record.description,
                                                  self.proteins)
            if protein_name is None:
                print(record.description)
                continue

            if self.SLiM_proteins is not None:

                has_SLiM = protein_name in self.SLiM_proteins

                # keywordを持っていない => SLiMを持っていない
                if not has_SLiM:
                    label_list = [0] * len(record.seq)
                else:
                    label_list = self._annotate(seq)

            else:
                label_list = self._annotate(seq)

            seq_dict[protein_name].append(seq)
            label_dict[protein_name].append(label_list)

        # Bio.Seqオブジェクトからstrオブジェクトに変換
        for key, value in seq_dict.items():
            seq_dict[key] = [str(seq) for seq in value]

        result_dict = {}
        for key in self.proteins.keys():
            if self.n_gram:
                x, y = self._n_gram_split(seq_dict[key], label_dict[key])
            else:
                x, y = self._split(seq_dict[key], label_dict[key])

            x = np.array(x).reshape(-1, 1)
            y = np.array(y)

            # タプルに変換
            result_dict[key] = (x, y)

        return result_dict

    def _split(self, seqs, label_lists):
        """ アミノ酸配列を指定された長さに分割する.

        Args:
            seqs, label_lists(list): アミノ酸配列・ラベルリストのリスト

        Returns:
            x(list of str): 分割されたアミノ酸断片
            y(list of int): アミノ酸断片にSLiMが含まれる場合は0,
                含まれない場合は1をラベリングしたリスト

        """
        x, y = [], []

        for seq, label_list in zip(seqs, label_lists):
            if self.random_split:
                surplus = len(seq) % self.length
                i = random.randint(0, surplus)
            else:
                i = 0

            while (i <= len(seq) - self.length):
                fragment = seq[i:(i+self.length)]
                x.append(fragment)

                slim_count = label_list[i:(i+self.length)].count(1)
                has_slim = slim_count >= self.threshold
                y.append(int(has_slim))

                i += self.length

        return x, y

    def _n_gram_split(self, seqs, label_lists):
        """ アミノ酸配列を分割し，n_gramのデータセットを作成

        Args:
            seqs, label_lists(list): アミノ酸配列・ラベルリストのリスト

        Returns:
            x(list of str): アミノ酸断片
            y(list of int): アミノ酸断片にSLiMが含まれる場合は0,
                含まれない場合は1をラベリングしたリスト
        """
        x, y = [], []

        for seq, label_list in zip(seqs, label_lists):

            i = 0
            while (i <= len(seq) - self.length):
                fragment = seq[i:(i+self.length)]
                x.append(fragment)
                slim_count = label_list[i:(i+self.length)].count(1)
                has_slim = slim_count >= self.threshold
                y.append(int(has_slim))

                i += 1

        return x, y


    def _annotate(self, seq):
        """ ラベルリストを作成

        次のようなアミノ酸配列から'APTAPP'というSLiM配列を検出する場合，
            sequence : MGAPTAPPQDN
            label    : 00111111000
        となるから，ラベルリストは次のようになる．
            label_list = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]

        Arg:
            seq(str): アノテーションを行うアミノ酸配列

        Return:
            label_list(list of int): SLiMに該当する部分が1,
                該当しない部分を0としたリスト．

        """
        label_list = []

        i = 0
        while(i < len(seq) - len(self.SLiM) + 1):
            fragment = seq[i:(i+len(self.SLiM))]
            annotation = 0

            if self.neighbor is not None:
                slim_neighbourhood = (i >= self.idx - self.neighbor - 1) \
                    and (i <= self.idx + len(self.SLiM) + self.neighbor - 1)
                if slim_neighbourhood:
                    annotation = self._annotate_one(fragment,
                                            threshold=len(self.SLiM))
            else:
                annotation = self._annotate_one(fragment)

            if annotation == 1:
                label_list += [annotation] * len(self.SLiM)
                i += 6
            else:
                label_list.append(annotation)
                i += 1

        while (i < len(seq)):
            label_list.append(0)
            i += 1

        return label_list

    def _annotate_one(self, seq):
        threshold = len(self.SLiM) - self.replacement_tolerance
        label = 0

        count = 0
        for i, slim_char in enumerate(self.SLiM):
            if slim_char == seq[i]:
                count += 1

        if count >= threshold:
            label = 1

        return label


def extract(fastafile, keywords=None, proteins=None):
    """ FASTAファイルからヘッダー行にkeywordを含む配列を抽出する

    Args:
        fastafile(str): FASTAファイルの名前
        keyword(str): 抽出したい配列のキーワード
            e.g. 'Gene Symbol:L'

    """
    if (keywords is not None) and (proteins is None):
        raise ArgumentError

    records = []

    with open(fastafile, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):

            if keywords is None:
                records.append(record)

            else:
                protein_name = determine_protein_name(
                        record.description, proteins)

                if protein_name is None:
                    print(protein_name)
                    continue

                if protein_name in keywords:
                    records.append(record)

    return records
