from unittest import TestCase

from ptan.common.utils import SMAQueue


class TestSMAQueue(TestCase):
    def test_repr(self):
        q = SMAQueue(4)
        self.assertEqual(repr(q), "SMAQueue(size=4)")

    def test_len(self):
        q = SMAQueue(4)
        self.assertEqual(0, len(q))
        q += [1]
        self.assertEqual(1, len(q))
        q += 2
        self.assertEqual(2, len(q))
        q += [1, 2, 3]
        self.assertEqual(4, len(q))

    def test_min(self):
        self.fail()

    def test_mean(self):
        self.fail()

    def test_max(self):
        self.fail()
