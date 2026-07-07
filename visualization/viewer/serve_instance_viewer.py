from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from visualization.viewer.build_viewer_manifest import ViewerManifestBuilder, build_viewer_manifest, load_raw_manifest
from visualization.viewer.solution_visualization import SolutionVisualizationService
from visualization.viewer.static_assets import ViewerStaticAssets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the WebGL instance viewer against a raw base, mrmp, or ST heuristic-ablation manifest."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw benchmark manifest.json or pilot_manifest.json file.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Specific benchmark record id to expose. Can be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of exposed records.",
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        default=None,
        help="Optional override for the shared base manifest root.",
    )
    parser.add_argument(
        "--solution-json",
        action="append",
        default=[],
        type=Path,
        help="Viewer solution JSON to preload. Can be repeated.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_raw_manifest(args.input)
    payload = build_viewer_manifest(
        records,
        source_manifest_path=args.input,
        instance_ids=args.instance_id,
        limit=args.limit,
        base_root=args.base_root,
    )
    solution_payloads = []
    for solution_path in args.solution_json:
        solution_payload = json.loads(Path(solution_path).read_text())
        if isinstance(solution_payload, list):
            solution_payloads.extend(solution_payload)
        else:
            solution_payloads.append(solution_payload)
    if solution_payloads:
        payload["solutions"] = solution_payloads
    payload_bytes = json.dumps(payload).encode("utf-8")
    exposed_instance_ids = {
        str(instance["instance_id"])
        for instance in payload["instances"]
    }
    records_by_id = {
        str(record.instance_id): record
        for record in records
        if str(record.instance_id) in exposed_instance_ids
    }
    solution_service = SolutionVisualizationService(
        ViewerManifestBuilder.base_root_for_manifest(args.input, args.base_root)
    )
    viewer_url = ViewerStaticAssets.viewer_url()

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            if ViewerStaticAssets.is_viewer_path(parsed.path):
                ViewerStaticAssets.write_html_response(self)
                return
            if parsed.path == "/":
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", viewer_url)
                self.end_headers()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if ViewerStaticAssets.is_viewer_path(parsed.path):
                ViewerStaticAssets.write_html_response(self)
                return
            if parsed.path == "/api/manifest":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload_bytes)))
                self.end_headers()
                self.wfile.write(payload_bytes)
                return
            if parsed.path == "/api/solution-planners":
                self.json_response(HTTPStatus.OK, solution_service.planner_payload())
                return
            if parsed.path == "/":
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", viewer_url)
                self.end_headers()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/solution":
                self.error_response(HTTPStatus.NOT_FOUND, FileNotFoundError(parsed.path))
                return
            try:
                data = self.read_json()
                instance_id = str(data["instance_id"])
                if instance_id not in records_by_id:
                    raise KeyError(f"Instance {instance_id!r} is not exposed by this viewer.")
                planner_defaults = solution_service.planner_payload()
                result = solution_service.run_solution(
                    records_by_id[instance_id],
                    planner_key=str(data["planner_key"]),
                    budget=float(data.get("budget", solution_service.DEFAULT_BUDGET)),
                    seed_offset=int(data.get("seed_offset", 0)),
                    window_span_factor=float(data.get("window_alpha", planner_defaults["default_window_alpha"])),
                    execution_horizon_factor=float(data.get("window_beta", planner_defaults["default_window_beta"])),
                    epsilon=float(data.get("epsilon", planner_defaults["default_epsilon"])),
                )
                self.json_response(HTTPStatus.OK, result)
            except Exception as exc:
                self.error_response(HTTPStatus.BAD_REQUEST, exc)

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            payload_data = self.rfile.read(length)
            return json.loads(payload_data.decode("utf-8")) if payload_data else {}

        def json_response(self, status: HTTPStatus, response_payload: dict) -> None:
            response_bytes = json.dumps(response_payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

        def error_response(self, status: HTTPStatus, exc: Exception) -> None:
            self.json_response(status, {"ok": False, "error": str(exc)})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "source_kind": payload["source_kind"],
                "instance_count": payload["instance_count"],
                "url": f"http://{args.host}:{args.port}{viewer_url}",
            },
            indent=2,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
