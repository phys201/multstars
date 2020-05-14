from unittest import TestCase
from multstars.utils import lognorm_peak, estimate_MAP
import pandas as pd

lognorm_peak_testoutput = lognorm_peak(3,2,4,3,2,4)

class test_lognorm_peak(TestCase):

    def test_outputtype(self):
        self.assertTrue(isinstance(lognorm_peak_testoutput,tuple))
        self.assertTrue(isinstance(lognorm_peak_testoutput[0],float))

    def test_calc(self):
        self.assertEqual(lognorm_peak_testoutput[0],1.0)
        self.assertAlmostEqual(lognorm_peak_testoutput[1],2.83,2)
        self.assertAlmostEqual(lognorm_peak_testoutput[2],5.66,2)

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