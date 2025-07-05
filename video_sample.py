from pathlib import Path

from utils import process_video

# ---------------------------------------------------------------------------
# Edit this to point at any clip in your dataset
video_path = Path("assets/RomeSampleMorningDriveDowntown_144p.mp4")
# ---------------------------------------------------------------------------

outdir = Path("outputs")
outdir.mkdir(parents=True, exist_ok=True)

result = process_video(str(video_path))

# console preview
print("Event  :", result["event"])
print("Summary:\n", result["summary"])

# save to disk
summary_file = outdir / f"{video_path.stem}_summary.txt"
summary_file.write_text(result["summary"], encoding="utf-8")
print(f"\n[✓] Summary saved to {summary_file.relative_to(Path.cwd())}")
