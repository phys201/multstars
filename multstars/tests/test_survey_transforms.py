from unittest import TestCase
from multstars.survey_transformations import *

class test_imr_to_icr(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(imr_to_icr(2),float))

    def test_calc(self):
        self.assertAlmostEqual(imr_to_icr(2), 0.512, 3)

class test_imr_to_cr(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(imr_to_cr(2),float))

    def test_calc(self):
        self.assertAlmostEqual(imr_to_cr(2), 1.952, 3)