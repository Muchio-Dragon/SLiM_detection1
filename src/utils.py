import numpy as np
import re
import random
import math

def determine_protein_name(desc, proteins):
    """ recordのdescriptionからタンパク質種を識別する

    Example:
        desc = '>gb:BAA...53|Protein Name:NS3|Segment:10'
        result = determine_protein_name(desc)
        print(result)
        # output: NS3

    Arg:
        desc(str): recordのdescription

    Return:
        recordのタンパク質種．
        見つからない場合，Noneを返す．

    """
    def _find_keyword(keyword, desc, proteins):
        """ desc の中から keywordを見つける """
        match = re.search(keyword, desc)
        name = desc[match.end():]

        for i, char in enumerate(name):
            if (char == '|') or (char == '\n'):
                name = name[:i]
                break

        result = None
        for key, values in proteins.items():
            for value in values:
                if value in name:
                    result = key
                    break

        return result

    result = None

    if 'Gene Symbol' in desc:
        # Gene Symbol を取得
        result = _find_keyword(r'Gene Symbol:', desc, proteins)

    if result is None:
        # Protein Name を取得
        result = _find_keyword(r'Protein Name:', desc, proteins)

    return result

def drop_SLiM(seqs, slim, replacement_tolerance):
    """ アミノ酸配列中からSLiM部分を抜く

    Args:
        seqs(ndarray): shape=(n_samples, 1)
        slim(str): 抜き出すSLiM配列
        replacement_tolerance(int): この値まで置換を許容して，SLiMと判定する

    Return:
        ndarray: shape=(n_samples, 1)

    """
    dropped_seqs = []
    slim_length = len(slim)

    for seq in seqs:
        seq = seq[0]

        # SLiMを持っているか判定
        has_slim = False
        for i in range(len(seq) - slim_length + 1):
            count = 0

            for j, slim_char in enumerate(slim):
                if (slim_char == seq[i + j]):
                    count += 1

            if (count >= slim_length - replacement_tolerance):
                has_slim = True
                slim_idx = i
                if (count == slim_length):
                    break

        if has_slim:
            dropped_seq = seq[:slim_idx] + seq[(slim_idx + slim_length):]
        else:
            j = random.randint(0, len(seq) - slim_length)
            dropped_seq = seq[:j] + seq[(j + slim_length):]

        dropped_seqs.append(dropped_seq)

    dropped_seqs = np.array(dropped_seqs).reshape(-1, 1)
    return dropped_seqs

def calc_f_beta_score(precision, recall, beta=1.0):
    try:
        f_beta_score = (1 + beta**2) * precision * recall \
                / ((beta ** 2) * precision + recall)
    except ZeroDivisionError:
        f_beta_score = 0.0

    if math.isnan(f_beta_score):
        f_beta_score = 0.0

    return f_beta_score
