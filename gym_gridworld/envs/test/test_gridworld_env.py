from gym_gridworld import gridworld_env
import unittest

class TestGridWorldEnv(unittest.TestCase):
    def setUp(self):
        self.env = gridworld_env()

    def test_init(self):
        obs = self.env.reset()
        