import unittest

from westac.common import file_utility


class Test_ExtractMeta(unittest.TestCase):

    def test_extract_metadata_when_valid_regexp_returns_metadata_values(self):
        filename = 'SOU 1957_5 Namn.txt'
        meta = file_utility.extract_filename_fields(filename, year=r".{4}(\d{4})_.*", serial_no=r".{8}_(\d+).*")
        self.assertEqual(5, meta.serial_no)
        self.assertEqual(1957, meta.year)

    def test_extract_metadata_when_invalid_regexp_returns_none(self):
        filename = 'xyz.txt'
        meta = file_utility.extract_filename_fields(filename, value=r".{4}(\d{4})_.*")
        self.assertEqual(None, meta.value)
