from pathlib import Path

from utils import process_video

# ---------------------------------------------------------------------------
# Edit this to point at any clip in your dataset
video_path = Path("assets/RomeSampleMorningDriveDowntown_144p.mp4")
# ---------------------------------------------------------------------------

outdir = Path("outputs")
outdir.mkdir(parents=True, exist_ok=True)

result = process_video(str(video_path))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
log = logging.getLogger(__name__)

# ---- pipeline -------------------------------------------------------------
result = process_video(str(video_path))

log.info("Event  : %s", result["event"])
log.info("Summary:\n%s", result["summary"])

summary_file = outdir / f"{video_path.stem}_summary.txt"
summary_file.write_text(result["summary"], encoding="utf-8")
log.info("\n[✓] Summary saved to %s", summary_file.relative_to(Path.cwd()))
