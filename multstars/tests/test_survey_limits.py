from unittest import TestCase
from multstars.survey_limits import icr_max, sep_min

class test_icr_max(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(icr_max(2),float))

    def test_calc(self):
        self.assertAlmostEqual(icr_max(2), 0.198, 3)

class test_sep_min(TestCase):

    def test_output_type(self):
        self.assertTrue(isinstance(sep_min(2),float))

    def test_calc(self):
        self.assertAlmostEqual(sep_min(2), 0.16, 2)
