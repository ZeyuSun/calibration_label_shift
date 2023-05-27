import unittest
from simulation import Simulation1


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


if __name__ == '__main__':
    unittest.main()