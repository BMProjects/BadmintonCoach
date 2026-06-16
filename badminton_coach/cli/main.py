"""`badminton` CLI: run perception, inspect available backends."""

from __future__ import annotations

import argparse

from ..core.config import load_config
from ..core.pipeline import Phase1Pipeline


def _cmd_analyze(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    pipeline = Phase1Pipeline.from_config(cfg)
    result = pipeline.run(args.video)
    print(f"source           : {result.source}")
    print(f"fps / frames     : {result.fps:.2f} / {result.frame_count}")
    print(f"court reproj err : {result.court.reprojection_error_px:.2f} px")
    print(f"detections       : {len(result.detections)}")
    print(f"player tracks    : {len(result.player_tracks)}")
    print(f"poses            : {len(result.poses)}")
    print(f"shuttle 2D pts   : {len(result.shuttle_2d)}")
    if result.shuttle_3d is not None:
        print(
            f"shuttle 3D pts   : {len(result.shuttle_3d)} "
            f"(method={result.shuttle_3d.method}, "
            f"reproj={result.shuttle_3d.reprojection_error_px:.1f} px)"
        )


def _cmd_backends(_args: argparse.Namespace) -> None:
    import badminton_coach.perception  # noqa: F401  (registers backends)

    from ..core.registry import available_backends

    for kind, names in available_backends().items():
        print(f"{kind:18s}: {', '.join(names)}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="badminton", description="Badminton video analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Run the Phase-1 perception pipeline")
    p_analyze.add_argument("--video", required=True)
    p_analyze.add_argument("--config", required=True)
    p_analyze.set_defaults(func=_cmd_analyze)

    p_backends = sub.add_parser("backends", help="List registered backends")
    p_backends.set_defaults(func=_cmd_backends)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
