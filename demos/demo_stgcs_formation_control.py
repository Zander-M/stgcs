from __future__ import annotations

from demos.sequential_mrmp_utils import SequentialMRMPDemo


TASK_CONFIG = {
    "name": "formation_control_3x3_stgcs_horizontal_blocks",
    "instance_id": "formation-control-3x3-stgcs-horizontal-blocks",
    "task_type": "formation_control",
    "robot_radius": 0.05,
    "vlimit": 0.25,
    "tmax": 1000.0,
    "stage_timeout_secs": 180.0,
    "search_eps": 5.0,
    "planner": {
        "coordination": "windowed-pbs",
        "window_alpha": 5.0,
        "window_beta": 1.0,
        "dynamic_window_adjustment": True,
        "child_expansion_rule": "num_conflicts",
    },
    "cspace": [
        [[0.0, 0.0], [4.8, 0.0], [4.8, 1.0], [0.0, 1.0]],
    ],
    "stages": [
        {
            "name": "grid_to_s",
            "queries": [
                {"start": [0.16, 0.20], "goal": [1.344, 0.80]},
                {"start": [0.40, 0.20], "goal": [1.20, 0.86]},
                {"start": [0.64, 0.20], "goal": [1.072, 0.75]},
                {"start": [0.16, 0.50], "goal": [1.12, 0.60]},
                {"start": [0.40, 0.50], "goal": [1.20, 0.50]},
                {"start": [0.64, 0.50], "goal": [1.28, 0.40]},
                {"start": [0.16, 0.80], "goal": [1.328, 0.25]},
                {"start": [0.40, 0.80], "goal": [1.20, 0.14]},
                {"start": [0.64, 0.80], "goal": [1.056, 0.20]},
            ],
        },
        {
            "name": "s_to_t",
            "queries": [
                {"start": [1.344, 0.80], "goal": [1.76, 0.84]},
                {"start": [1.20, 0.86], "goal": [1.92, 0.84]},
                {"start": [1.072, 0.75], "goal": [2.08, 0.84]},
                {"start": [1.12, 0.60], "goal": [2.24, 0.84]},
                {"start": [1.20, 0.50], "goal": [2.00, 0.70]},
                {"start": [1.28, 0.40], "goal": [2.00, 0.56]},
                {"start": [1.328, 0.25], "goal": [2.00, 0.42]},
                {"start": [1.20, 0.14], "goal": [2.00, 0.28]},
                {"start": [1.056, 0.20], "goal": [2.00, 0.14]},
            ],
        },
        {
            "name": "t_to_g",
            "queries": [
                {"start": [1.76, 0.84], "goal": [3.024, 0.79]},
                {"start": [1.92, 0.84], "goal": [2.84, 0.86]},
                {"start": [2.08, 0.84], "goal": [2.648, 0.78]},
                {"start": [2.24, 0.84], "goal": [2.56, 0.61]},
                {"start": [2.00, 0.70], "goal": [2.552, 0.44]},
                {"start": [2.00, 0.56], "goal": [2.656, 0.25]},
                {"start": [2.00, 0.42], "goal": [2.848, 0.18]},
                {"start": [2.00, 0.28], "goal": [3.024, 0.30]},
                {"start": [2.00, 0.14], "goal": [2.864, 0.46]},
            ],
        },
        {
            "name": "g_to_c",
            "queries": [
                {"start": [3.024, 0.79], "goal": [3.80, 0.80]},
                {"start": [2.84, 0.86], "goal": [3.616, 0.84]},
                {"start": [2.648, 0.78], "goal": [3.44, 0.80]},
                {"start": [2.56, 0.61], "goal": [3.344, 0.64]},
                {"start": [2.552, 0.44], "goal": [3.32, 0.50]},
                {"start": [2.656, 0.25], "goal": [3.344, 0.36]},
                {"start": [2.848, 0.18], "goal": [3.44, 0.20]},
                {"start": [3.024, 0.30], "goal": [3.616, 0.16]},
                {"start": [2.864, 0.46], "goal": [3.80, 0.20]},
            ],
        },
        {
            "name": "c_to_s",
            "queries": [
                {"start": [3.80, 0.80], "goal": [4.544, 0.80]},
                {"start": [3.616, 0.84], "goal": [4.40, 0.86]},
                {"start": [3.44, 0.80], "goal": [4.272, 0.75]},
                {"start": [3.344, 0.64], "goal": [4.32, 0.60]},
                {"start": [3.32, 0.50], "goal": [4.40, 0.50]},
                {"start": [3.344, 0.36], "goal": [4.48, 0.40]},
                {"start": [3.44, 0.20], "goal": [4.528, 0.25]},
                {"start": [3.616, 0.16], "goal": [4.40, 0.14]},
                {"start": [3.80, 0.20], "goal": [4.256, 0.20]},
            ],
        },
        {
            "name": "s_to_grid",
            "queries": [
                {"start": [4.544, 0.80], "goal": [0.16, 0.20]},
                {"start": [4.40, 0.86], "goal": [0.40, 0.20]},
                {"start": [4.272, 0.75], "goal": [0.64, 0.20]},
                {"start": [4.32, 0.60], "goal": [0.16, 0.50]},
                {"start": [4.40, 0.50], "goal": [0.40, 0.50]},
                {"start": [4.48, 0.40], "goal": [0.64, 0.50]},
                {"start": [4.528, 0.25], "goal": [0.16, 0.80]},
                {"start": [4.40, 0.14], "goal": [0.40, 0.80]},
                {"start": [4.256, 0.20], "goal": [0.64, 0.80]},
            ],
        },
    ],
}


if __name__ == "__main__":
    SequentialMRMPDemo.run_cli(
        TASK_CONFIG,
        description="Solve the script-defined ST-GCS formation-control sequential MRMP demo.",
    )
