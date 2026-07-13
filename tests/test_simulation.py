from __future__ import annotations

import unittest
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from orbitllm.config import load_config
from orbitllm.dataset_workloads import generate_eurosat_tasks, load_eurosat_records
from orbitllm.profile import default_parameterized_profile
from orbitllm.reviewer_experiments import run_policy_physical
from orbitllm.simulate import (
    SOTA_BASELINE_POLICIES,
    DownlinkScheduler,
    Task,
    Window,
    generate_tasks,
    generate_windows,
    run_policy,
    run_sota_baseline_suite,
    task_costs,
    train_drl_ua_inspired,
    train_leo_drl_inspired,
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

    def test_recent_baselines_share_hard_constraints(self) -> None:
        tasks = generate_tasks(self.cfg, seed=0, limit=24)
        windows = generate_windows(self.cfg)
        for policy in [
            "DPP-LLM",
            "ActiveInf-inspired",
            "LEO-DRL-inspired",
            "Hierarchical-RA-inspired",
            "DRL-UA-inspired",
            "QoS-Multihop-projected",
            "DelayCost-adapted",
            "PBR-adapted",
            "EH-DORA-adapted",
            "PeakMem-inspired",
        ]:
            metrics, _ = run_policy(policy, tasks, windows, self.profile, self.cfg)
            self.assertEqual(metrics["oom_count"], 0, policy)
            self.assertEqual(metrics["energy_violation_count"], 0, policy)
            self.assertGreater(metrics["total_value"], 0, policy)

    def test_jsa_recent_baselines_cover_targeted_resource_models(self) -> None:
        cfg = load_config()
        tasks = generate_tasks(cfg, seed=4, limit=24)
        windows = generate_windows(cfg)

        pbr, _ = run_policy("PBR-adapted", tasks, windows, self.profile, cfg)
        self.assertEqual(pbr["oom_count"], 0)
        self.assertEqual(pbr["energy_violation_count"], 0)

        soc_cfg = load_config()
        soc_cfg["soc"]["initial_soc_fraction"] = 0.45
        eh, _ = run_policy(
            "EH-DORA-adapted", tasks, windows, self.profile, soc_cfg, energy_model="soc"
        )
        self.assertGreaterEqual(float(eh["min_energy_j"]), 0.0)
        self.assertEqual(eh["energy_violation_count"], 0)

        mem_cfg = load_config()
        mem_cfg["satellite"]["memory_gb"] = 8.0
        peak, _ = run_policy("PeakMem-inspired", tasks, windows, self.profile, mem_cfg)
        self.assertEqual(peak["oom_count"], 0)
        self.assertEqual(peak["energy_violation_count"], 0)

    def test_dpp_updates_nonnegative_virtual_queues(self) -> None:
        tasks = generate_tasks(self.cfg, seed=1, limit=36)
        metrics, _ = run_policy("DPP-LLM", tasks, generate_windows(self.cfg), self.profile, self.cfg)
        self.assertGreaterEqual(float(metrics["dpp_energy_queue_final"]), 0.0)
        self.assertGreaterEqual(float(metrics["dpp_bandwidth_queue_final"]), 0.0)

    def test_learning_baseline_uses_separate_training_seeds(self) -> None:
        cfg = load_config()
        cfg["leo_drl"]["training_seeds"] = [101, 102]
        cfg["leo_drl"]["training_epochs"] = 1
        trained = train_leo_drl_inspired(cfg, self.profile, horizon_h=2.0, task_limit=18)
        self.assertTrue(trained["q_table"])
        self.assertNotIn(0, trained["training_seeds"])
        cfg["drl_ua"]["training_seeds"] = [103, 104]
        cfg["drl_ua"]["training_epochs"] = 1
        ua_trained = train_drl_ua_inspired(cfg, self.profile, horizon_h=2.0, task_limit=18)
        self.assertTrue(ua_trained["q_table"])
        self.assertNotIn(0, ua_trained["training_seeds"])
        q_size = len(trained["q_table"])
        df = run_sota_baseline_suite(cfg, self.profile, seeds=[0], horizon_h=2.0, task_limit=18)
        self.assertEqual(len(df), len(SOTA_BASELINE_POLICIES))
        self.assertTrue((df["oom_count"] == 0).all())
        self.assertGreaterEqual(q_size, 1)

    def test_recent_baselines_obey_physical_link_energy(self) -> None:
        cfg = load_config()
        cfg["physical_link"] = {
            "radio_circuit_w": 6.0,
            "pa_static_w": 2.0,
            "pa_peak_w": 16.0,
            "pa_efficiency": 0.35,
        }
        tasks = generate_tasks(cfg, seed=0, limit=20)
        windows = generate_windows(cfg)
        for policy in [
            "DPP-LLM",
            "ActiveInf-inspired",
            "LEO-DRL-inspired",
            "Hierarchical-RA-inspired",
            "DRL-UA-inspired",
            "QoS-Multihop-projected",
            "DelayCost-adapted",
        ]:
            metrics = run_policy_physical(policy, tasks, windows, self.profile, cfg)
            self.assertEqual(metrics["oom_count"], 0, policy)
            self.assertEqual(metrics["energy_violation_count"], 0, policy)
            self.assertGreaterEqual(metrics["residual_energy_j"], 0.0, policy)

    def test_eurosat_workload_uses_encoded_pre_bytes_and_per_image_eta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "EuroSAT.zip"
            rows = []
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index in range(12):
                    key = f"2750/Forest/Forest_{index}.jpg"
                    archive.writestr(key, b"x" * (100 + index))
                    rows.append(
                        {
                            "backend": "mobile_sam",
                            "image": f"data/eurosat/{key}",
                            "label": "Forest",
                            "selector": "sam_top3",
                            "rho": 0.20 + index / 100.0,
                            "rho_bytes_vs_source": 0.10 + index / 100.0,
                            "eta_prob": 0.50 + index / 100.0,
                        }
                    )
            calibration_path = root / "pre_rows.csv"
            pd.DataFrame(rows).to_csv(calibration_path, index=False)
            records = load_eurosat_records(archive_path, calibration_path)
            tasks, trace = generate_eurosat_tasks(
                self.cfg, records, seed=0, task_limit=12
            )
            self.assertEqual(len(tasks), 12)
            self.assertEqual(len(trace), 12)
            self.assertTrue((trace["image_bytes"] > 0).all())
            self.assertTrue(all(task.data_bits > 0 for task in tasks))
            self.assertTrue(
                all(task.pre_retention_fraction is not None for task in tasks)
            )
            self.assertEqual(
                {round(float(task.pre_retention_fraction), 2) for task in tasks},
                {round(0.10 + index / 100.0, 2) for index in range(12)},
            )
            self.assertEqual(
                {round(float(task.pre_value_factor), 2) for task in tasks},
                {round(0.50 + index / 100.0, 2) for index in range(12)},
            )


if __name__ == "__main__":
    unittest.main()
