from unittest import TestCase
from multstars.model import *
from multstars.data_io import *
from multstars.utils import estimate_MAP
import numpy as np
import pandas

data_path = get_example_data_file_path('hm_data_test.txt')
data = load_data(data_path)

np.random.seed(1)
traces, samples = pymc3_hrchl_fit(data.sample(10),tune=10,nsteps=10,random_seed=1)
MAP_df = estimate_MAP(samples,'center')

class model(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(samples,pandas.core.frame.DataFrame))

    def test_inference(self):
        self.assertAlmostEqual(MAP_df['MAP'][0],7.768,3)