import csv
import subprocess
import os

ANNOTATIONS = "annotations/k400_val.csv"
OUTPUT_ROOT = "videos"
TARGET_CLASS = "salsa dancing"

MAX_VIDEOS = 50  # start small

os.makedirs(os.path.join(OUTPUT_ROOT, TARGET_CLASS), exist_ok=True)

def download_and_trim(youtube_id, start, end, out_path):
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "--quiet",
        "--no-warnings",
        "--download-sections", f"*{start}-{end}",
        "-o", out_path,
        url
    ]

    subprocess.run(cmd, check=False)

count = 0

with open(ANNOTATIONS, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["label"] != TARGET_CLASS:
            continue

        youtube_id = row["youtube_id"]
        start = float(row["time_start"])
        end = float(row["time_end"])

        filename = f"{youtube_id}_{int(start)}_{int(end)}.mp4"
        out_path = os.path.join(OUTPUT_ROOT, TARGET_CLASS, filename)

        if os.path.exists(out_path):
            continue

        print(f"Downloading {filename}")
        download_and_trim(youtube_id, start, end, out_path)

        count += 1
        if count >= MAX_VIDEOS:
            break
