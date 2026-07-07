from __future__ import annotations

from demos.sequential_mrmp_utils import SequentialMRMPDemo


TASK_CONFIG = {
    "name": "robot_rearrange_3x3_stgcs",
    "instance_id": "robot-rearrange-3x3-stgcs",
    "task_type": "rearrangement",
    "robot_radius": 0.05,
    "vlimit": 0.25,
    "tmax": 60.0,
    "stage_timeout_secs": 60.0,
    "search_eps": 1.0,
    "planner": {
        "coordination": "windowed-pbs",
        "window_alpha": 5.0,
        "window_beta": 1.0,
        "dynamic_window_adjustment": True,
        "child_expansion_rule": "num_conflicts",
    },
    "bounds": [
        [0.0, 0.0],
        [1.0, 1.0],
    ],
    "stages": [
        {
            "name": "grid_to_s",
            "queries": [
                {"start": [0.20, 0.20], "goal": [0.68, 0.8]},
                {"start": [0.50, 0.20], "goal": [0.50, 0.86]},
                {"start": [0.80, 0.20], "goal": [0.34, 0.75]},
                {"start": [0.20, 0.50], "goal": [0.4, 0.6]},
                {"start": [0.50, 0.50], "goal": [0.5, 0.5]},
                {"start": [0.80, 0.50], "goal": [0.6, 0.4]},
                {"start": [0.20, 0.80], "goal": [0.66, 0.25]},
                {"start": [0.50, 0.80], "goal": [0.50, 0.14]},
                {"start": [0.80, 0.80], "goal": [0.32, 0.2]},
            ],
        },
        {
            "name": "s_to_t",
            "queries": [
                {"start": [0.68, 0.8], "goal": [0.2, 0.84]},
                {"start": [0.50, 0.86], "goal": [0.4, 0.84]},
                {"start": [0.34, 0.75], "goal": [0.6, 0.84]},
                {"start": [0.4, 0.6], "goal": [0.8, 0.84]},
                {"start": [0.5, 0.5], "goal": [0.50, 0.70]},
                {"start": [0.6, 0.4], "goal": [0.50, 0.56]},
                {"start": [0.66, 0.25], "goal": [0.50, 0.42]},
                {"start": [0.50, 0.14], "goal": [0.50, 0.28]},
                {"start": [0.32, 0.2], "goal": [0.50, 0.14]},
            ],
        },
        {
            "name": "t_to_g",
            "queries": [
                {"start": [0.2, 0.84], "goal": [0.78, 0.79]},
                {"start": [0.4, 0.84], "goal": [0.55, 0.86]},
                {"start": [0.6, 0.84], "goal": [0.31, 0.78]},
                {"start": [0.8, 0.84], "goal": [0.20, 0.61]},
                {"start": [0.50, 0.70], "goal": [0.19, 0.44]},
                {"start": [0.50, 0.56], "goal": [0.32, 0.25]},
                {"start": [0.50, 0.42], "goal": [0.56, 0.18]},
                {"start": [0.50, 0.28], "goal": [0.78, 0.3]},
                {"start": [0.50, 0.14], "goal": [0.58, 0.46]},
            ],
        },
        {
            "name": "g_to_c",
            "queries": [
                {"start": [0.78, 0.79], "goal": [0.75, 0.8]},
                {"start": [0.55, 0.86], "goal": [0.52, 0.84]},
                {"start": [0.31, 0.78], "goal": [0.3, 0.8]},
                {"start": [0.20, 0.61], "goal": [0.18, 0.64]},
                {"start": [0.19, 0.44], "goal": [0.15, 0.50]},
                {"start": [0.32, 0.25], "goal": [0.18, 0.36]},
                {"start": [0.56, 0.18], "goal": [0.3, 0.2]},
                {"start": [0.78, 0.3], "goal": [0.52, 0.16]},
                {"start": [0.58, 0.46], "goal": [0.75, 0.2]},
            ],
        },
        {
            "name": "c_to_s",
            "queries": [
                {"start": [0.75, 0.8], "goal": [0.68, 0.8]},
                {"start": [0.52, 0.84], "goal": [0.50, 0.86]},
                {"start": [0.3, 0.8], "goal": [0.34, 0.75]},
                {"start": [0.18, 0.64], "goal": [0.4, 0.6]},
                {"start": [0.15, 0.50], "goal": [0.5, 0.5]},
                {"start": [0.18, 0.36], "goal": [0.6, 0.4]},
                {"start": [0.3, 0.2], "goal": [0.66, 0.25]},
                {"start": [0.52, 0.16], "goal": [0.50, 0.14]},
                {"start": [0.75, 0.2], "goal": [0.32, 0.2]},
            ],
        },
        {
            "name": "s_to_grid",
            "queries": [
                {"start": [0.68, 0.8], "goal": [0.20, 0.20]},
                {"start": [0.50, 0.86], "goal": [0.50, 0.20]},
                {"start": [0.34, 0.75], "goal": [0.80, 0.20]},
                {"start": [0.4, 0.6], "goal": [0.20, 0.50]},
                {"start": [0.5, 0.5], "goal": [0.50, 0.50]},
                {"start": [0.6, 0.4], "goal": [0.80, 0.50]},
                {"start": [0.66, 0.25], "goal": [0.20, 0.80]},
                {"start": [0.50, 0.14], "goal": [0.50, 0.80]},
                {"start": [0.32, 0.2], "goal": [0.80, 0.80]},
            ],
        },
    ],
}


if __name__ == "__main__":
    SequentialMRMPDemo.run_cli(
        TASK_CONFIG,
        description="Solve the script-defined ST-GCS rearrangement sequential MRMP demo.",
    )
