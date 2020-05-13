from unittest import TestCase
from multstars.utils import lognorm_peak, estimate_MAP
import numpy as np
import pandas as pd

class test_lognorm_peak(TestCase):

    def test_outputtype(self):
        self.assertTrue(isinstance(lognorm_peak(2,3),float))

    def test_calc(self):
        self.assertAlmostEqual(lognorm_peak(2,3),0.37,2)

samples = pd.DataFrame({'param1':[1,2,3,4],'param2':[2,3,4,5]})
map_test = estimate_MAP(samples,'param1','param2')

class test_estimate_MAP(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(map_test,pd.core.frame.DataFrame))

    def test_output_length(self):
        self.assertEqual(len(estimate_MAP(samples,'param1')),1)
        self.assertEqual(len(map_test),2)

    def test_calc(self):
        self.assertEqual(map_test['MAP'][0],2.5)