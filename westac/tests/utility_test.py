import unittest

from westac.common import utility

class test_Utility(unittest.TestCase):

    def setUp(self):
        pass

    def test_dehypen(self):

        text = 'absdef\n'
        result = utility.dehyphen(text)
        self.assertEqual(text, result)

        text = 'abs-def\n'
        result = utility.dehyphen(text)
        self.assertEqual(text, result)

        text = 'abs - def\n'
        result = utility.dehyphen(text)
        self.assertEqual(text, result)

        text = 'abs-\ndef'
        result = utility.dehyphen(text)
        self.assertEqual('absdef\n', result)

        text = 'abs- \r\n def'
        result = utility.dehyphen(text)
        self.assertEqual('absdef\n', result)

    def test_compress_whitespaces(self):

        text = 'absdef\n'
        result = utility.compress_whitespaces(text)
        self.assertEqual('absdef', result)

        text = ' absdef \n'
        result = utility.compress_whitespaces(text)
        self.assertEqual( 'absdef', result)

        text = 'abs  def'
        result = utility.compress_whitespaces(text)
        self.assertEqual('abs def', result)

        text = 'abs\n def'
        result = utility.compress_whitespaces(text)
        self.assertEqual('abs def', result)

        text = 'abs- \r\n def'
        result = utility.compress_whitespaces(text)
        self.assertEqual('abs- def', result)

class Test_ExtractMeta(unittest.TestCase):

    def test_extract_metadata_when_valid_regexp_returns_metadata_values(self):
        filename = 'SOU 1957_5 Namn.txt'
        meta = utility.extract_metadata(filename, year=r".{4}(\d{4})_.*", serial_no=r".{8}_(\d+).*")
        self.assertEqual(5, meta.serial_no)
        self.assertEqual(1957, meta.year)

    def test_extract_metadata_when_invalid_regexp_returns_none(self):
        filename = 'xyz.txt'
        meta = utility.extract_metadata(filename, value=r".{4}(\d{4})_.*")
        self.assertEqual(None, meta.value)
