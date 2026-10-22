import json
import unittest
import numpy as np
from src.giada_teacher.task41f_evaluation import json_native


class DiagnosticJSONTests(unittest.TestCase):
    def test_numpy_protocol_and_metric_values_roundtrip(self):
        value={'case':{'v':np.arange(6).reshape(2,3),'seed':np.int64(17)},
               'metrics':{'error':np.float32(.25),'passed':np.bool_(True)}}
        actual=json.loads(json.dumps(json_native(value),allow_nan=False))
        self.assertEqual(actual,{'case':{'v':[[0,1,2],[3,4,5]],'seed':17},
                                 'metrics':{'error':.25,'passed':True}})

    def test_nonfinite_is_not_hidden(self):
        with self.assertRaises(ValueError):
            json.dumps(json_native({'value':np.float64('nan')}),allow_nan=False)


if __name__=='__main__':unittest.main()
