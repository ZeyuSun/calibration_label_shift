import unittest
import numpy as np
from simulation_calibration import Simulation1
from simulation_label_shift import LabelShiftSimulation
from utils import digitize


class CalibrationSimulationTestCase(unittest.TestCase):

    def setUp(self):
        self.sim = Simulation1()

    def test_run_single(self):
        results = self.sim.run_single(n=100, B=10, i=0)

    # def test_run(self):  # included in test_plot
    #     df = self.sim.run(n_list=[100, 200], B_list=[5, 10], i_list=[0])
    #     self.assertEqual(len(df), 4)

    def test_plot(self):
        df = self.sim.run(n_list=[100, 200], B_list=[5, 10], i_list=[0])
        self.sim.plot(df)


class LabelShiftSimulationTestCase(unittest.TestCase):

    def setUp(self):
        self.sim = LabelShiftSimulation()

    def test_run_single(self):
        results = self.sim.run_single(nt=100, pt=0.1, i=0)

    # def test_run(self):
    #     df = self.sim.run(nt_list=[100, 200], pt_list=[0.1, 0.2], i_list=[0])
    #     self.assertEqual(len(df), 8)

    def test_tabulate(self):
        df = self.sim.run(nt_list=[100, 200], pt_list=[0.1], i_list=[0])
        df_str = self.sim.tabulate(df, to_latex=True)

    def test_plot(self):
        self.sim.plot()


def test_digitize():
    bins = [0, 0.3, 0.6, 1]
    cases = [
        ([0, 0.3, 0.9, 1], [0, 1, 2, 2]),
        (0.9, 2),
    ]
    for x, y in cases:
        assert np.all(digitize(x, bins) == y)


if __name__ == '__main__':
    unittest.main()