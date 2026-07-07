import subprocess
import sys
import json
from pathlib import Path
from typing import Any, Sequence

def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def start_instance_viewer(
    manifest_path: Path,
    base_root: Path,
    instance_id: str,
    solution_json: Path | Sequence[Path],
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    solution_paths = [solution_json] if isinstance(solution_json, Path) else list(solution_json)
    for path in solution_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing saved solution: {path}")

    print(f"Viewer: http://{host}:{port}/")
    solution_args = []
    for path in solution_paths:
        solution_args.extend(["--solution-json", str(path)])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "visualization.viewer.serve_instance_viewer",
            "--input",
            str(manifest_path),
            "--base-root",
            str(base_root),
            "--instance-id",
            instance_id,
            *solution_args,
            "--host",
            host,
            "--port",
            str(port),
        ],
        check=True,
    )
