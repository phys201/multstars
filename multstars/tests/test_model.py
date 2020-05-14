from unittest import TestCase
from multstars.model import *
from multstars.data_io import *
import pandas

data_path = get_example_data_file_path('hm_data_test.txt')
data = load_data(data_path)

traces, samples = pymc3_hrchl_fit(data.sample(10),tune=10,nsteps=10)

class model(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(samples,pandas.core.frame.DataFrame))