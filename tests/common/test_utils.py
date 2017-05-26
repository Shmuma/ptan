import time
from unittest import TestCase
import numpy as np
import torch
from torch.autograd import Variable

from ptan.common.utils import SMAQueue, SpeedMonitor, WeightedMSELoss



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
        q = SMAQueue(4)
        self.assertIsNone(q.min())
        q += [1, 10, 0]
        self.assertEqual(0, q.min())

    def test_mean(self):
        q = SMAQueue(4)
        self.assertIsNone(q.min())
        q += [1, 10, 0]
        self.assertAlmostEqual(3.6666666, q.mean(), places=5)

    def test_max(self):
        q = SMAQueue(4)
        self.assertIsNone(q.min())
        q += [1, 10, 0]
        self.assertEqual(10, q.max())


class TestSpeedMonitor(TestCase):
    def test_basic(self):
        m = SpeedMonitor(1, autostart=True)
        time.sleep(0.1)
        self.assertGreaterEqual(m.seconds(), 0.1)
        m = SpeedMonitor(1, autostart=False)
        time.sleep(0.1)
        m.reset()
        time.sleep(0.2)
        self.assertLess(m.seconds(), 0.3)

    def test_epoch(self):
        m = SpeedMonitor(1)
        time.sleep(0.1)
        dt = m.epoch_time()
        self.assertEqual(m.epoches, 0)
        self.assertGreaterEqual(dt.total_seconds(), 0.1)
        m.epoch()
        self.assertEqual(m.epoches, 1)
        time.sleep(0.1)
        dt = m.epoch_time()
        self.assertGreaterEqual(dt.total_seconds(), 0.1)
        self.assertLess(dt.total_seconds(), 0.2)

    def test_batch(self):
        m = SpeedMonitor(1)
        time.sleep(0.1)
        dt = m.batch_time()
        self.assertEqual(m.batches, 0)
        self.assertGreaterEqual(dt.total_seconds(), 0.1)
        m.batch()
        self.assertEqual(m.batches, 1)
        time.sleep(0.1)
        dt = m.batch_time()
        self.assertGreaterEqual(dt.total_seconds(), 0.1)
        self.assertLess(dt.total_seconds(), 0.2)

    def test_samples_per_sec(self):
        m = SpeedMonitor(10)
        time.sleep(0.1)
        ss = m.samples_per_sec()
        self.assertLessEqual(ss, 100)
        m.batch()
        time.sleep(0.2)
        ss = m.samples_per_sec()
        self.assertGreaterEqual(ss, 66)
        self.assertLessEqual(ss, 70)


class TestWeightedMSELoss(TestCase):
    def get_loss(self, loss, input, target, weights=None):
        input_v = Variable(torch.from_numpy(np.array(input)))
        target_v = Variable(torch.from_numpy(np.array(target)))
        if weights is not None:
            weights_v = Variable(torch.from_numpy(np.array(weights)))
        else:
            weights_v = None
        return loss(input_v, target_v, weights_v).data.numpy()

    def test_no_weights(self):
        loss = WeightedMSELoss(size_average=False)

        np.testing.assert_almost_equal(self.get_loss(loss, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]), [0.0])
        np.testing.assert_almost_equal(self.get_loss(loss, [1.0, 1.0, 1.0], [0.0, 2.0, 0.0]), [3.0])
        np.testing.assert_almost_equal(self.get_loss(loss, [1.0, 1.0, 1.0], [1.0, 5.0, 1.0]), [16.0])

        loss = WeightedMSELoss(size_average=True)
        np.testing.assert_almost_equal(self.get_loss(loss, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]), [0.0])
        np.testing.assert_almost_equal(self.get_loss(loss, [1.0, 1.0, 1.0], [0.0, 2.0, 0.0]), [1.0])
        np.testing.assert_almost_equal(self.get_loss(loss, [1.0, 1.0, 1.0], [1.0, 5.0, 1.0]), [16.0/3])

    def test_weights(self):
        loss = WeightedMSELoss(size_average=False)

        np.testing.assert_almost_equal(self.get_loss(loss, [1.0, 1.0, 1.0], [0.0, 2.0, 0.0], [1.0, 3.0, 2.0]), [6.0])
        np.testing.assert_almost_equal(self.get_loss(loss, [1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0/4, 1.0]), [4.0])
