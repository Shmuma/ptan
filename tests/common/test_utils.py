import time
from unittest import TestCase
import numpy as np
import torch
from torch.autograd import Variable

from ptan.common.utils import SMAQueue, SpeedMonitor, WeightedMSELoss, TBMeanTracker



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


class TestTBWriter:
    def __init__(self):
        self.closed = False
        self.buffer = []

    def reset(self):
        self.closed = False
        self.buffer = []

    def close(self):
        self.closed = True

    def add_scalar(self, name, value, iter_idx):
        assert not self.closed
        self.buffer.append((name, value, iter_idx))


class TestTBMeanTracker(TestCase):
    def test_simple(self):
        writer = TestTBWriter()
        with TBMeanTracker(writer, batch_size=4) as tracker:
            tracker.track("param_1", value=10, iter_index=1)
            self.assertEquals(writer.buffer, [])
            tracker.track("param_1", value=10, iter_index=2)
            self.assertEquals(writer.buffer, [])
            tracker.track("param_1", value=10, iter_index=3)
            self.assertEquals(writer.buffer, [])
            tracker.track("param_1", value=10, iter_index=4)
            self.assertEquals(writer.buffer, [("param_1", 10.0, 4)])
            writer.reset()

            tracker.track("param_1", value=1.0, iter_index=1)
            self.assertEquals(writer.buffer, [])
            tracker.track("param_1", value=2.0, iter_index=2)
            self.assertEquals(writer.buffer, [])
            tracker.track("param_1", value=-3.0, iter_index=3)
            self.assertEquals(writer.buffer, [])
            tracker.track("param_1", value=1.0, iter_index=4)
            self.assertEquals(writer.buffer, [("param_1", (1.0 + 2.0 - 3.0 + 1.0) / 4, 4)])
            writer.reset()


        with self.assertRaises(AssertionError):
            writer.add_scalar("bla", 1.0, 10)
        self.assertTrue(writer.closed)

    def test_tensor(self):
        writer = TestTBWriter()

        with TBMeanTracker(writer, batch_size=2) as tracker:
            t = torch.LongTensor([1])
            tracker.track("p1", t, iter_index=1)
            tracker.track("p1", t, iter_index=2)
            self.assertEquals(writer.buffer, [("p1", 1.0, 2)])

    def test_tensor_large(self):
        writer = TestTBWriter()

        with TBMeanTracker(writer, batch_size=2) as tracker:
            t = torch.LongTensor([1, 2, 3])
            tracker.track("p1", t, iter_index=1)
            tracker.track("p1", t, iter_index=2)
            self.assertEquals(writer.buffer, [("p1", 2.0, 2)])

    def test_as_float(self):
        self.assertAlmostEqual(1.0, TBMeanTracker._as_float(1.0))
        self.assertAlmostEqual(0.33333333333333, TBMeanTracker._as_float(1.0/3.0))
        self.assertAlmostEqual(2.0, TBMeanTracker._as_float(torch.LongTensor([1, 2, 3])))
        self.assertAlmostEqual(0.6666666666666, TBMeanTracker._as_float(torch.LongTensor([1, 1, 0])))
        self.assertAlmostEqual(1.0, TBMeanTracker._as_float(np.array([1.0, 1.0, 1.0])))
        self.assertAlmostEqual(1.0, TBMeanTracker._as_float(np.sqrt(np.array([1.0], dtype=np.float32)[0])))
