"""Phase-1 perception pipeline orchestrator.

Wires the six L1 backends (selected by config) into a single run() that turns a
video into a PerceptionResult. The orchestrator knows only the interfaces, never
the concrete backends — that is what makes every stage swappable.
"""

from __future__ import annotations

from pathlib import Path

from . import registry
from .config import AppConfig
from .interfaces import (
    CourtCalibrator,
    Detector,
    PlayerTracker,
    PoseEstimator,
    Reconstructor3D,
    ShuttleTracker,
)
from .io import VideoReader, sliding_clips
from .schemas import (
    Frame,
    ObjectClass,
    PerceptionResult,
    ShuttlePoint2D,
    ShuttleTrajectory2D,
)


class Phase1Pipeline:
    """Perception pipeline: detect -> track players -> pose -> shuttle -> 3D."""

    def __init__(
        self,
        config: AppConfig,
        detector: Detector,
        shuttle_tracker: ShuttleTracker,
        pose_estimator: PoseEstimator,
        player_tracker: PlayerTracker,
        court_calibrator: CourtCalibrator,
        reconstructor: Reconstructor3D | None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.shuttle_tracker = shuttle_tracker
        self.pose_estimator = pose_estimator
        self.player_tracker = player_tracker
        self.court_calibrator = court_calibrator
        self.reconstructor = reconstructor

    @classmethod
    def from_config(cls, config: AppConfig) -> Phase1Pipeline:
        """Build every backend from config via the registry.

        Importing the perception package triggers backend self-registration.
        """
        import badminton_coach.perception  # noqa: F401  (registers backends)

        p = config.perception
        reconstructor = None
        if p.reconstructor.enabled:
            reconstructor = registry.build("reconstructor", p.reconstructor.model_dump())

        return cls(
            config=config,
            detector=registry.build("detector", p.detector.model_dump()),
            shuttle_tracker=registry.build("shuttle_tracker", p.shuttle_tracker.model_dump()),
            pose_estimator=registry.build("pose_estimator", p.pose_estimator.model_dump()),
            player_tracker=registry.build("player_tracker", p.player_tracker.model_dump()),
            court_calibrator=registry.build("court_calibrator", p.court_calibrator.model_dump()),
            reconstructor=reconstructor,
        )

    def run(self, video_path: str | Path, estimate_3d: bool | None = None) -> PerceptionResult:
        """Run perception over a video and return the aggregate result.

        estimate_3d: include the 2D-court-calibration -> 3D-estimation subsystem
        (stable_background + court calibrate + per-frame presence + 3D reconstruction).
        None -> use the config default (perception.estimate_3d). Set False for a fast
        2D-only run (detection/pose/tracking/shuttle-2D) to compare efficiency.
        """
        io = self.config.io
        with VideoReader(video_path, max_frames=io.max_frames, stride=io.stride) as reader:
            frames: list[Frame] = list(reader)
            fps = reader.fps

        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}")

        do_3d = self.config.perception.estimate_3d if estimate_3d is None else estimate_3d

        # 3D-estimation subsystem (toggleable): court calibration + per-frame presence.
        # Court calibration may fail (amateur/phone footage); degrade gracefully — the
        # 2D pipeline (player/pose/shuttle) still runs without a court.
        court, court_frames = (None, None)
        if do_3d:
            court, court_frames = self._estimate_court(frames, fps)

        # Unified perception: when the pose backend can also detect (YOLO-pose yields
        # boxes + keypoints in one forward), skip the separate detector pass and take
        # both from a single forward with exact box<->pose pairing.
        unified = (self.config.perception.unified_perception
                   and hasattr(self.pose_estimator, "detect_and_pose"))
        if unified:
            detections, player_dets_per_frame, poses = self._detect_pose_unified(frames, court)
        else:
            detections, player_dets_per_frame, poses = self._detect_then_pose(frames, court)

        player_tracks = self.player_tracker.track(player_dets_per_frame, frames)

        shuttle_2d = self._track_shuttle(frames, fps)

        shuttle_3d = None
        if do_3d and self.reconstructor is not None and court is not None and len(shuttle_2d) > 0:
            shuttle_3d = self.reconstructor.reconstruct(shuttle_2d, court)

        return PerceptionResult(
            source=str(video_path),
            fps=fps,
            frame_count=len(frames),
            court=court,
            detections=tuple(detections),
            player_tracks=tuple(player_tracks),
            poses=tuple(poses),
            shuttle_2d=shuttle_2d,
            shuttle_3d=shuttle_3d,
            court_frames=court_frames,
        )

    def _estimate_court(self, frames, fps):
        """3D-estimation front half: calibrate the court (stable-background bootstrap,
        or load a cached profile for fixed-camera reuse) then, if enabled, batch the
        per-frame presence gate. Returns (court, court_frames)."""
        court = self._load_or_calibrate_court(frames, fps)
        court_frames = None
        if court is not None and self.config.perception.court_per_frame_presence:
            court_frames = frozenset(self.court_calibrator.present_frames(frames))
        return court, court_frames

    def _load_or_calibrate_court(self, frames, fps):
        """Fixed-camera reuse: load a saved CalibrationProfile if present (skips
        stable_background + calibrate), else calibrate once and persist it."""
        from pathlib import Path

        from .schemas import CalibrationProfile

        ppath = self.config.perception.court_profile_path
        if ppath and Path(ppath).exists():
            return CalibrationProfile.load(ppath).to_calibration()

        court = self._try_calibrate_court(frames, fps)
        if court is not None and ppath:
            w, h = frames[0].width, frames[0].height
            CalibrationProfile(
                source_key=Path(ppath).stem,
                image_size=(w, h),
                image_corners=[],  # learned backends have no 4-corner markers
                homography=court.homography,
                reprojection_error_px=court.reprojection_error_px,
                camera=court.camera,
            ).save(ppath)
        return court

    def _try_calibrate_court(self, frames, fps):
        """Fixed-camera bootstrap. Primary: calibrate on the stable-segment MEDIAN
        background (players removed, no replay/transition) — robust to bad first
        frames and line occlusion. Fallback: try several evenly-sampled raw frames.
        Returns None if all fail (caller degrades gracefully)."""
        from .io import stable_background
        from .schemas import Frame

        bg, _info = stable_background(frames, fps)
        if bg is not None:
            try:
                return self.court_calibrator.calibrate(Frame(index=0, timestamp=0.0, image=bg))
            except Exception:  # noqa: BLE001 - fall back to raw frames
                pass
        n = len(frames)
        idxs = sorted({int(i * (n - 1) / 7) for i in range(8)})
        for i in idxs:
            try:
                return self.court_calibrator.calibrate(frames[i])
            except Exception:  # noqa: BLE001 - try next frame
                continue
        return None

    def _filter_to_court(self, players, court):
        if court is not None and self.config.perception.filter_players_to_court:
            return self._filter_players_to_court(players, court)
        return players

    def _detect_then_pose(self, frames, court):
        """Separate detector + pose passes (two forwards/frame)."""
        detections, player_dets_per_frame, poses = [], [], []
        for frame in frames:
            frame_dets = self.detector.detect(frame)
            players = [d for d in frame_dets if d.cls is ObjectClass.PLAYER]
            non_players = [d for d in frame_dets if d.cls is not ObjectClass.PLAYER]
            players = self._filter_to_court(players, court)
            player_dets_per_frame.append(players)
            detections.extend(non_players)
            detections.extend(players)
        for frame, fdets in zip(frames, player_dets_per_frame, strict=True):
            poses.extend(self.pose_estimator.estimate(frame, [d.bbox for d in fdets]))
        return detections, player_dets_per_frame, poses

    def _detect_pose_unified(self, frames, court):
        """One YOLO-pose forward/frame yields boxes + keypoints; pose stays paired to
        its box through court filtering (no separate detector, no IoU association)."""
        detections, player_dets_per_frame, poses = [], [], []
        for frame in frames:
            dets, fposes = self.pose_estimator.detect_and_pose(frame)
            pose_by_det = {id(d): p for d, p in zip(dets, fposes, strict=True)}
            players = self._filter_to_court([d for d in dets if d.cls is ObjectClass.PLAYER], court)
            player_dets_per_frame.append(players)
            detections.extend(players)
            poses.extend(pose_by_det[id(d)] for d in players)
        return detections, player_dets_per_frame, poses

    def _filter_players_to_court(self, players, court):
        """Keep player detections whose ground point is inside the court (+margin),
        then the largest `max_players` — removes umpire/line-judges/audience."""
        from .geometry.court_model import COURT_LENGTH_M, COURT_WIDTH_DOUBLES_M

        m = self.config.perception.court_margin_m
        on_court = []
        for d in players:
            g = court.image_to_ground(d.bbox.foot)
            if -m <= g.x <= COURT_WIDTH_DOUBLES_M + m and -m <= g.y <= COURT_LENGTH_M + m:
                on_court.append(d)
        on_court.sort(key=lambda d: d.bbox.width * d.bbox.height, reverse=True)
        return on_court[: self.config.perception.max_players]

    def _track_shuttle(self, frames: list[Frame], fps: float) -> ShuttleTrajectory2D:
        """Run the heatmap shuttle tracker over sliding windows, keep last vote per frame."""
        window = self.config.io.clip_window
        if len(frames) < window:
            window = len(frames)
        by_index: dict[int, ShuttlePoint2D] = {}
        for clip in sliding_clips(frames, window=window, fps=fps, stride=window):
            traj = self.shuttle_tracker.track(clip)
            for pt in traj.points:
                by_index[pt.frame_index] = pt
        points = tuple(by_index[i] for i in sorted(by_index))
        return ShuttleTrajectory2D(points=points)
