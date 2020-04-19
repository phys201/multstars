from unittest import TestCase
from multstars.data_io import get_example_data_file_path, load_data
from multstars.model import likelihood, emcee_fit
import pandas



class model(TestCase):


    def test_is_dataFrame(self):
        estimate = [20, 20]
        data_path = get_example_data_file_path('doubles_test.txt')
        data = load_data(data_path)
        samples = emcee_fit(data, estimate)
        self.assertTrue(isinstance(samples, pandas.core.frame.DataFrame))

    def test_is_float(self):
        data_path = get_example_data_file_path('doubles_test.txt')
        data = load_data(data_path)
        parameters = [15,50]
        L = likelihood(data, parameters)
        self.assertTrue(isinstance(L, float))
