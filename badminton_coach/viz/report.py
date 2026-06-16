"""Player report: aggregate a MatchAnalysis + PlayerProfile into a Markdown report.

Summarises tactics (rallies, stroke mix, movement, landings) and biomechanics (per-joint
load, kinematic-sequence quality, peak-effort stroke) into a coaching-style report.
"""

from __future__ import annotations

from collections import defaultdict


def build_player_report(analysis, profile, backend: str = "lift3d") -> str:
    if analysis is None:
        return "## Player Report\n\n_No analysis yet — process a clip first._"
    lines = ["# 🏸 Player Report", ""]
    if profile is not None:
        lines.append(f"**Profile** · {profile.height_m:.2f} m · {profile.mass_kg:.0f} kg · "
                     f"{profile.handedness}-handed")
        lines.append("")

    # --- match / tactics ---
    st = analysis.stats
    lines.append("## Match")
    lines.append(f"- Rallies **{st.rally_count if st else 0}** · "
                 f"shots **{len(analysis.shots)}** · "
                 f"avg **{st.avg_shots_per_rally:.1f}** shots/rally" if st
                 else f"- shots **{len(analysis.shots)}**")
    if st and st.shot_type_counts:
        mix = " · ".join(f"{k} {v}" for k, v in sorted(st.shot_type_counts.items(),
                                                       key=lambda kv: -kv[1]))
        lines.append(f"- Stroke mix: {mix}")
    if st and st.player_movement:
        lines.append("")
        lines.append("## Movement")
        for m in st.player_movement:
            lines.append(f"- Player {m.track_id}: **{m.distance_m:.1f} m** covered · "
                         f"avg {m.avg_speed_ms:.1f} · max **{m.max_speed_ms:.1f} m/s**")

    # --- biomechanics ---
    bio = analysis.biomechanics
    if bio and bio.strokes:
        type_by_idx = {i: s.shot_type.value for i, s in enumerate(analysis.shots, 1)}
        loads = defaultdict(list)
        angles = defaultdict(list)
        for sb in bio.strokes:
            for j in sb.joints:
                loads[j.name].append(j.peak_torque_nm)
                angles[j.name].append(j.peak_angle_deg)
        dom = max(loads, key=lambda k: sum(loads[k]) / len(loads[k])) if loads else "-"
        seq_ok = sum(sb.sequence_ok for sb in bio.strokes)
        peak = max(bio.strokes, key=lambda sb: sb.effort_nm)
        lines += ["", f"## Biomechanics ({backend})",
                  f"- Strokes analysed: **{len(bio.strokes)}**",
                  f"- Kinematic sequence proper (proximal→distal): "
                  f"**{seq_ok}/{len(bio.strokes)}** ({seq_ok / len(bio.strokes):.0%})",
                  f"- Dominant load joint: **{dom}**",
                  f"- Peak effort: shot **#{peak.shot_index}** "
                  f"({type_by_idx.get(peak.shot_index, '?')}) · **{peak.effort_nm:.0f} N·m**",
                  "",
                  "### Per-joint (avg over strokes)",
                  "| joint | avg peak angle | avg load (N·m) |",
                  "|---|---|---|"]
        for name in ("hip", "knee", "shoulder", "elbow"):
            if name in loads:
                lines.append(f"| {name} | {sum(angles[name]) / len(angles[name]):.0f}° "
                             f"| {sum(loads[name]) / len(loads[name]):.0f} |")
        lines += ["", "### Per-stroke",
                  "| # | type | effort (N·m) | sequence | ok |", "|---|---|---|---|---|"]
        for sb in bio.strokes:
            seq = "→".join(s[:4] for s in sb.kinematic_sequence)
            lines.append(f"| {sb.shot_index} | {type_by_idx.get(sb.shot_index, '?')} | "
                         f"{sb.effort_nm:.0f} | {seq} | {'✓' if sb.sequence_ok else '✗'} |")

        # --- insights ---
        lines += ["", "## Insights"]
        pct = seq_ok / len(bio.strokes)
        if pct >= 0.6:
            lines.append("- ✅ Energy transfer is mostly proximal→distal (good kinematic chain).")
        else:
            lines.append("- ⚠️ Kinematic sequence often out of order — work on hip/trunk-led "
                         "energy transfer before the arm.")
        lines.append(f"- Highest joint load is at the **{dom}**; monitor for overuse.")
    else:
        lines += ["", "## Biomechanics", "_Enable 3D estimation to compute biomechanics._"]

    lines += ["", "_Note: monocular single-view estimates — relative/coaching metrics, "
              "not lab-grade absolute torques._"]
    return "\n".join(lines)
