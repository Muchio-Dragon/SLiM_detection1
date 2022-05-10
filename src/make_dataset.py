import os
import argparse
import json
import pickle

from src.data.dataset import Dataset, extract


# コマンドライン引数を取得
parser = argparse.ArgumentParser()

parser.add_argument('length', help="Amino acids are fragmentated by this length.",
                    type=int)
parser.add_argument('fasta_dir', help="Path to FASTA files.",
                    type=str)
parser.add_argument('out_dir', help="Path to output files.",
                    type=str)
parser.add_argument('--n_gram', help="If True, n_gram dataset are generated.",
                    type=bool, default=True)

args = parser.parse_args()


def main():
    json_path = 'references/PTAP_data.json'
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    for data in json_data:
        virusname = data['virus'].replace(' ', '_')

        fasta_path = os.path.join(args.fasta_dir, f'{virusname}.fasta')

        records = extract(fasta_path)
        dataset_maker = Dataset(
                SLiM=data['SLiM'],
                idx=data['start_index'],
                length=args.length,
                proteins=data['proteins'],
                SLiM_proteins=data['SLiM_proteins'],
                neighbor=data['neighbor'],
                replacement_tolerance=data['replacement_tolerance'],
                threshold=len(data['SLiM']),
                n_gram=args.n_gram)

        x, y = dataset_maker.make_dataset(records, dict=False)

        out_path = os.path.join(args.out_dir, f'{virusname}.pickle')
        with open(out_path, 'wb') as f:
            pickle.dump(x, f)
            pickle.dump(y, f)

if __name__ == '__main__':
    main()
