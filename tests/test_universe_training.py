import ast
from pathlib import Path
import unittest
import numpy as np

class UniverseSplit(unittest.TestCase):
    def test_training_labels_do_not_cross_validation_boundary(self):
        tree=ast.parse(Path('scripts/train_clam_universe.py').read_text())
        node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='split_masks')
        namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'split','exec'),namespace)
        dates=np.array(['2019-12-20','2019-12-30','2019-12-31'],dtype='datetime64[D]')
        ends=np.array(['2019-12-27','2020-01-07','2020-01-08'],dtype='datetime64[D]')
        train,val=namespace['split_masks'](dates,ends,'2019-12-31')
        self.assertEqual(train.tolist(),[True,False,False]);self.assertEqual(val.tolist(),[False,False,True])

if __name__=='__main__':unittest.main()
