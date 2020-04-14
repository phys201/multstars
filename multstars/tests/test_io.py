# edited from phys201 example

from unittest import TestCase
from multstars.data_io import get_example_data_file_path, load_data

class TestIo(TestCase):
    def test_data_io(self):
        data = load_data(get_example_data_file_path('doubles_test.txt'))
        assert data.LSPM_ID[0] == 'J0324+3804'