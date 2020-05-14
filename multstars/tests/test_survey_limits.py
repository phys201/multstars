from unittest import TestCase
from multstars.survey_limits import *

class test_sep_min(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(sep_min(2),float))

    def test_calc(self):
        self.assertAlmostEqual(sep_min(2), 0.37, 2)

class test_sep_max(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(sep_max(2), float))

class test_cr_max(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(cr_max(2),float))

    def test_calc(self):
        self.assertAlmostEqual(cr_max(2), 5.048, 3)

class test_cr_min(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(cr_min(2), float))
