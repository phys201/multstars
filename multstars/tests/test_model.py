from unittest import TestCase
from multstars.model import log_likelihood, pymc3_fit
import pandas

fake_data = pandas.DataFrame({'SEP_PHYSICAL':[1,1,5,5,5,5,5,5,10,10,10,10,10,10,10,10,10,10,15,15,15,15,15,15,19,19],'E_SEP_PHYSICAL':[0]*26})

log_L = log_likelihood(fake_data,10,5)
fit_df = pymc3_fit(fake_data, nsteps=400, center_max=20)
q = fit_df.quantile([0.16,0.50,0.84], axis=0)

class model(TestCase):

    def test_log_likelihood_type(self):
        self.assertTrue(isinstance(log_L, float))

    def test_log_likelihood_calc(self):
        self.assertTrue(-42 <= log_L <= -41)

    def test_pymc3_fit_type(self):
        self.assertTrue(isinstance(fit_df, pandas.core.frame.DataFrame))

    def test_pymc3_fit_center_value(self):
        c_lower, c, c_upper = q['center']
        self.assertTrue(9 <= c <= 11)

    def test_pymc3_fit_width_value(self):
        w_lower, w, w_upper = q['width']
        self.assertTrue(0 <= w <= 10)
