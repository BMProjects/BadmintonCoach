"""Download ShuttleSet source match videos (YouTube) listed in set/match.csv.

ShuttleSet ships stroke annotations + YouTube URLs, not the videos (copyright). This
fetches selected matches at 720p (the annotation resolution) into a target dir, named by
the dataset `video` id so the bundled CSV annotations line up. For BST validation a couple
of matches is enough; pass --ids to limit, --limit N for the first N, or nothing for all.

    uv run python -m scripts.download_shuttleset --ids 1 2 --cookies-from-browser chrome

YouTube blocks server IPs behind three layers; all must be in place:
  1. auth cookies         — a logged-in browser (--cookies-from-browser chrome) or cookies.txt
  2. PO token provider    — `uv add bgutil-ytdlp-pot-provider` + run the bgutil node server
                            (clone Brainicism/bgutil-ytdlp-pot-provider, `npx tsc`,
                            `node server/build/main.js` -> listens on :4416)
  3. nsig 'n-challenge'   — a JS runtime (deno) on PATH; --remote-components ejs:github
                            fetches the EJS solver (added automatically by this script)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_MATCH_CSV = Path("third_party/BST/ShuttleSet/set/match.csv")


def _rows():
    with _MATCH_CSV.open() as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, help="dataset match ids (1-based)")
    ap.add_argument("--limit", type=int, help="download the first N matches")
    ap.add_argument("--out", default="data/shuttleset_videos")
    ap.add_argument("--height", type=int, default=720, help="max video height")
    ap.add_argument("--list", action="store_true", help="just print the matches")
    # YouTube blocks server IPs with a bot check; pass authenticated cookies to bypass.
    ap.add_argument("--cookies", help="path to a cookies.txt (exported from your browser)")
    ap.add_argument("--cookies-from-browser", help="browser name, e.g. chrome/firefox")
    args = ap.parse_args()

    rows = _rows()
    if args.ids:
        rows = [r for r in rows if int(r["id"]) in set(args.ids)]
    elif args.limit:
        rows = rows[: args.limit]

    if args.list:
        for r in rows:
            print(f"  {r['id']:>2} {r['video']}  {r['url']}")
        print(f"{len(rows)} match(es)")
        return

    import shutil
    import subprocess

    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp not installed — run: uv add yt-dlp")

    # YouTube on server IPs needs the full anti-bot chain (see module docstring):
    #   1. auth cookies (--cookies / --cookies-from-browser)
    #   2. a PO-token provider (bgutil server on :4416 + bgutil-ytdlp-pot-provider)
    #   3. a JS runtime + EJS solver for the nsig 'n-challenge' (--remote-components ejs:github)
    h = args.height
    # Prefer H.264 (avc1) — AV1/VP9 may not decode in the OpenCV build used downstream.
    fmt = (f"bestvideo[height<={h}][vcodec^=avc1]+bestaudio/"
           f"best[height<={h}][vcodec^=avc1]/bestvideo[height<={h}]+bestaudio/best")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    common = ["yt-dlp", "-f", fmt, "--merge-output-format", "mp4",
              "--remote-components", "ejs:github"]
    if args.cookies:
        common += ["--cookies", args.cookies]
    if args.cookies_from_browser:
        common += ["--cookies-from-browser", args.cookies_from_browser]

    ok, fail = 0, []
    for r in rows:
        dst = str(out / f"{r['id']}_{r['video']}.%(ext)s")
        print(f"[{r['id']}] {r['video']} ... ", end="", flush=True)
        proc = subprocess.run([*common, "-o", dst, r["url"]], capture_output=True, text=True)
        if proc.returncode == 0:
            ok += 1
            print("done")
        else:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
            fail.append((r["id"], tail[0][:100]))
            print(f"FAILED: {tail[0][:100]}")
    print(f"\n{ok} downloaded, {len(fail)} failed")
    for i, msg in fail:
        print(f"  id {i}: {msg}")


if __name__ == "__main__":
    main()
