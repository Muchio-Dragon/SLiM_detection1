import unittest
import json

from src.data.dataset import Dataset
from src.data.dataset import extract

class TestDatasetMaker(unittest.TestCase):
    """
    def test_make_dataset_dict(self):
        json_path = 'data/SLiM_data.json'
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        src_dir = 'src/origin/'
        for data in json_data:
            virusname = data['virus'].replace(' ', '_')
            src_path = os.path.join(src_dir, virusname + '.fasta')

            dataset_maker = DatasetMaker(slim=data['SLiM'],
                                         idx=data['start_index'],
                                         length=10,
                                         neighbor=data['neighbor'],
                                         SLiM_proteins=data['SLiM_proteins'],
                                         proteins=data['proteins'],
                                         remove_X=True,
                                         replacement_tolerance=data['replacement_tolerance'],
                                         threshold=6,
                                         n_gram=True)

            records = extract(src_path)
            dataset_dict = dataset_maker.make_dataset(records, dict=True)

            # dataset_dict がそれぞれのタンパク質種のkeyを持っているか確認
            for protein_name, key in zip(data['proteins'], dataset_dict.keys()):
                self.assertEqual(protein_name, key)

            # SLiMが確認されているタンパク質種だけに陽性のアノテーションがされていることを確認
            for key, (x, y) in dataset_dict.items():
                if key in data['SLiM_proteins']:    # SLiMが確認されているタンパク質種
                    self.assertTrue((y == 1).any())
                else:    # SLiMが確認されていなタンパク質種
                    self.assertTrue((y == 0).all())

    def test_make_dataset(self):
        json_path = 'data/SLiM_data.json'
        src_dir = 'src/origin/'

        length = 10

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        for data in json_data:
            virusname = data['virus'].replace(' ', '_')
            src_path = os.path.join(src_dir, virusname + '.fasta')

            dataset_maker = DatasetMaker(slim=data['SLiM'],
                                         idx=data['start_index'],
                                         length=length,
                                         neighbor=data['neighbor'],
                                         SLiM_proteins=data['SLiM_proteins'],
                                         proteins=data['proteins'],
                                         remove_X=True,
                                         replacement_tolerance=data['replacement_tolerance'],
                                         threshold=6,
                                         n_gram=True)

            records = extract(src_path)
            x, y = dataset_maker.make_dataset(records, dict=False)

            # データセットの数が合っているか確認
            n_samples = 0
            for record in records:
                n_samples += len(record.seq) - (length - 1)
            self.assertEqual(len(y), n_samples)
    """

    def test_n_gram_split(self):
        json_path = 'references/PTAP_data.json'

        length = 10

        with open(json_path, 'r') as f:
            json_data = json.load(f)
        data = json_data[0]

        dataset_maker = Dataset(SLiM=data['SLiM'],
                                idx=data['start_index'],
                                length=length,
                                neighbor=data['neighbor'],
                                SLiM_proteins=data['SLiM_proteins'],
                                proteins=data['proteins'],
                                remove_X=True,
                                replacement_tolerance=data['replacement_tolerance'],
                                threshold=6,
                                n_gram=True)

        seqs = ['CCCCCCLPTAPPCCCCC',
                'CCCCCCCCCCCCCC']
        label_lists = [[0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        x, y = dataset_maker._n_gram_split(seqs, label_lists)

        correct_x = ['CCCCCCLPTA', 'CCCCCLPTAP', 'CCCCLPTAPP', 'CCCLPTAPPC',
                     'CCLPTAPPCC', 'CLPTAPPCCC', 'LPTAPPCCCC', 'PTAPPCCCCC',
                     'CCCCCCCCCC', 'CCCCCCCCCC', 'CCCCCCCCCC', 'CCCCCCCCCC',
                     'CCCCCCCCCC']
        correct_y = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        self.assertEqual(x, correct_x)
        self.assertEqual(y, correct_y)


if __name__ == '__main__':
    unittest.main()
