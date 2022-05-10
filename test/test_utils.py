import sys
sys.path.append('.')

import unittest
import numpy as np
from src.utils import drop_SLiM


class TestUtils(unittest.TestCase):
    def test_drop_SLiM(self):
        slim = 'XXXXXX'
        replacement_tolerance = 1

        # SLiMを持っている場合
        x = np.array([['ABXXXXXXCDEF'],
                      ['ABCDEFGHXXXXXX'],
                      ['XXXXXXABCDEFG'],
                      ['ABCXXYXXXDEF']])
        correct = np.array([['ABCDEF'],
                            ['ABCDEFGH'],
                            ['ABCDEFG'],
                            ['ABCDEF']])
        x_dropped = drop_SLiM(x, slim, replacement_tolerance)
        self.assertTrue((x_dropped == correct).all())

        # SLiMを持っていない場合
        x = np.array([['ABCXYXXYXDEF'],
                      ['ABCDEXXYYXX'],
                      ['YYXXYXABCDE']])
        x_dropped = drop_SLiM(x, slim, replacement_tolerance)
        self.assertEqual(x.shape[0], x_dropped.shape[0])
        for i in range(x.shape[0]):
            self.assertEqual(len(x[i][0]) - len(slim), len(x_dropped[i][0]))


if __name__ == '__main__':
    unittest.main()
