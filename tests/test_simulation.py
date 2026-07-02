from __future__ import annotations

import unittest

from orbitllm.config import load_config
from orbitllm.profile import default_parameterized_profile
from orbitllm.simulate import (
    DownlinkScheduler,
    Task,
    Window,
    generate_tasks,
    generate_windows,
    run_policy,
    task_costs,
    value_at,
)


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()
        self.profile = default_parameterized_profile()

    def test_value_decay(self) -> None:
        task = Task(0, 0.0, 1.0, 3.0, 10.0, 60.0, 256, 64, "urgent")
        self.assertAlmostEqual(value_at(task, 0.0), 10.0)
        self.assertLess(value_at(task, 60.0), 10.0)

    def test_costs_obey_memory_shape(self) -> None:
        task = Task(0, 0.0, 8e6, 8.0, 10.0, 600.0, 2048, 128, "routine")
        costs = task_costs(task, self.profile, self.cfg)
        self.assertGreater(costs["ON"].energy_j, costs["PRE"].energy_j)
        self.assertGreater(costs["ON"].memory_gb, 0)
        self.assertGreater(costs["DOWN"].bits, costs["PRE"].bits)

    def test_downlink_scheduler_consumes_capacity(self) -> None:
        sched = DownlinkScheduler([Window(0, 10, 10)], 100)
        self.assertAlmostEqual(sched.schedule(0, 50), 5)
        self.assertAlmostEqual(sched.schedule(0, 25), 7.5)

    def test_heuristic_has_no_constraint_violations(self) -> None:
        tasks = generate_tasks(self.cfg, seed=0, limit=30)
        windows = generate_windows(self.cfg)
        metrics, _ = run_policy("Heuristic-TVD", tasks, windows, self.profile, self.cfg)
        self.assertEqual(metrics["oom_count"], 0)
        self.assertEqual(metrics["energy_violation_count"], 0)
        self.assertGreater(metrics["total_value"], 0)


if __name__ == "__main__":
    unittest.main()

