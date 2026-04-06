import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================
# Configuration
# =====================================================
# Absolute path to ffmpeg.exe on Windows
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

# Use the directory where this script is located
CWD = Path(__file__).resolve().parent

# Raw video segments (parts) to crop and concatenate
FILES = [
    "GX010056 - Part 1.MP4",
    "GX020056 - Part 2.MP4",
    "GX030056 - Part 3.MP4",
    "GX040056 - Part 4.MP4",
    "GX050056 - Part 5.MP4",
]

# ROI coordinates for cropping (reactor core area)
# Format: X-offset, Y-offset, Width, Height
X, Y, W, H = 1341, 528, 1422, 996

# Output videos
MERGED_VIDEO = CWD / "merged_crop.mp4"
DENOISED_VIDEO = CWD / "merged_crop_denoised.mp4"

# Trim settings (seconds)
TRIM_START_S = 60.0
TRIM_END_S = 4400.0

MERGED_VIDEO_TRIMMED = CWD / "merged_crop_trimmed.mp4"
DENOISED_VIDEO_TRIMMED = CWD / "merged_crop_denoised_trimmed.mp4"

# Denoiser filter (high-quality 3D denoiser)
DENOISE_FILTER = "hqdn3d=6:4:8:6"

# Saturated-pixel analysis settings (grayscale threshold and time intervals)
SAT_THRESHOLD = 250
INTERVALS = [
    (60, 500),
    (850, 1160),
    (1167, 1470),
    (1490, 1777),
    (1790, 2078),
    (2095, 2389),
    (2418, 2723),
    (2747, 3034),
    (3051, 3331),
    (3355, 3627),
    (3658, 3929),
    (3954, 4232),
]

# Output CSVs for per-frame RGB mean
RGB_ORG_CSV = CWD / "rgb_means_original.csv"
RGB_DN_CSV = CWD / "rgb_means_denoised.csv"
RGB_ORG_TRIM_CSV = CWD / "rgb_means_original_trimmed.csv"
RGB_DN_TRIM_CSV = CWD / "rgb_means_denoised_trimmed.csv"


# =====================================================
# Helpers
# =====================================================
def run_ffmpeg(cmd: str) -> None:
    """
    Execute an ffmpeg command.
    Note: shell=True is used for Windows quoting stability.
    """
    subprocess.run(cmd, check=True, shell=True)


def crop_and_concat(files, output_video: Path) -> None:
    """
    Crop each video part using the ROI and concatenate into a single video.
    Intermediate cropped parts are deleted after concatenation.
    """
    cropped_parts = []

    for name in files:
        inp = CWD / name
        if not inp.exists():
            print(f"[WARN] File not found: {name}")
            continue

        out = CWD / f"{inp.stem}_crop.mp4"
        cropped_parts.append(out)

        cmd = [
            f'"{FFMPEG}"', "-y",
            "-i", f'"{str(inp)}"',
            "-vf", f"crop={W}:{H}:{X}:{Y}",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",
            f'"{str(out)}"',
        ]
        print(f"[INFO] Cropping: {name}")
        run_ffmpeg(" ".join(cmd))

    if len(cropped_parts) == 0:
        raise RuntimeError("No cropped parts were created. Check input FILES and paths.")

    # Create concat list file for ffmpeg concat demuxer
    concat_list = CWD / "concat_list.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for v in cropped_parts:
            f.write(f"file '{v.resolve().as_posix()}'\n")

    # Concatenate by stream copy (fast)
    concat_cmd = [
        f'"{FFMPEG}"', "-y",
        "-f", "concat", "-safe", "0",
        "-i", f'"{str(concat_list)}"',
        "-c", "copy",
        f'"{str(output_video)}"',
    ]
    print(f"[INFO] Concatenating into: {output_video.name}")
    run_ffmpeg(" ".join(concat_cmd))

    # Cleanup
    if concat_list.exists():
        concat_list.unlink()
    for v in cropped_parts:
        if v.exists():
            v.unlink()


def denoise_video(inp: Path, outp: Path) -> None:
    """
    Apply hqdn3d denoising filter to a video.
    """
    cmd = [
        f'"{FFMPEG}"', "-y",
        "-i", f'"{str(inp)}"',
        "-vf", DENOISE_FILTER,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        f'"{str(outp)}"',
    ]
    print(f"[INFO] Denoising -> {outp.name}")
    run_ffmpeg(" ".join(cmd))


def trim_video(input_path: Path, output_path: Path, start_time: float, end_time: float) -> None:
    """
    Trim a video using FFmpeg with re-encoding to ensure accurate frame cutting.
    """
    cmd = [
        f'"{FFMPEG}"', "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", f'"{str(input_path)}"',
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        f'"{str(output_path)}"',
    ]
    print(f"[INFO] Trimming {input_path.name} -> {output_path.name} ({start_time}s to {end_time}s)")
    run_ffmpeg(" ".join(cmd))


def per_frame_rgb_mean(video_path: Path) -> pd.DataFrame:
    """
    Compute mean RGB values for every frame.
    Returns a dataframe with columns: time_s, mean_red, mean_green, mean_blue
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    t, r, g, b = [], [], [], []

    with tqdm(total=nframes, desc=f"Analyzing {video_path.name}", unit="fr") as pbar:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            mean_bgr = frame.mean(axis=(0, 1))
            b.append(mean_bgr[0])
            g.append(mean_bgr[1])
            r.append(mean_bgr[2])

            t.append(i / fps)
            i += 1
            pbar.update(1)

    cap.release()
    return pd.DataFrame({"time_s": t, "mean_red": r, "mean_green": g, "mean_blue": b})


def saturated_pixel_analysis(video_path: Path, intervals, threshold: int = 250) -> list:
    """
    For each time interval, compute the average count of saturated pixels per frame.
    Saturated pixel condition: grayscale value > threshold
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Loaded video: {video_path}")
    print(f"[INFO] FPS: {fps:.2f}, Total frames: {total_frames}")
    print("=" * 60)

    results = []

    # Progress bar over intervals
    for (t_start, t_end) in tqdm(intervals, desc="Processing intervals", unit="interval"):
        print(f"\n[INFO] Interval: {t_start}–{t_end} sec")

        f_start = int(t_start * fps)
        f_end = int(t_end * fps)
        n_frames = min(f_end, total_frames) - f_start

        print(f"[INFO] Frame range: {f_start}–{f_end}  ({n_frames} frames)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

        count_pixels_total = 0
        frame_count = 0

        # Progress bar within interval
        for _ in tqdm(range(max(n_frames, 0)), desc=f"Interval {t_start}-{t_end}", leave=False, unit="fr"):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale robustly
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            count_pixels_total += int(np.sum(gray > threshold))
            frame_count += 1

        avg_pixels = count_pixels_total / frame_count if frame_count > 0 else 0.0

        print(f"[INFO] Finished interval {t_start}–{t_end} sec")
        print(f"[INFO] Frames processed: {frame_count}")
        print(f"[INFO] Total saturated pixels (>{threshold}): {count_pixels_total}")
        print(f"[INFO] Average per frame: {avg_pixels:.2f}")

        results.append(((t_start, t_end), avg_pixels))

    cap.release()
    
    # ====================================================
    # Convert to DataFrame
    # ====================================================
    df_pixels = pd.DataFrame(
        results,
        columns=["Interval", "avg_pixels_gt250"]
    )

    print("\n====================================================")
    print("Interval-wise average saturated pixel counts:")
    display(df_pixels)

    # Optional: save to CSV
    df_pixels.to_csv("avg_saturated_pixels_by_interval.csv", index=False)

    print("\n" + "=" * 60)
    print("Final average saturated pixel counts:")
    for (t_start, t_end), avg_val in results:
        print(f"{t_start}-{t_end} sec: avg saturated pixels (>{threshold}) = {avg_val:.2f}")

    return results


# =====================================================
# Main execution
# =====================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Reactor Power Monitoring - Video Preprocessing Pipeline")
    print("=" * 60 + "\n")

    # Step 1) Crop each part and concatenate
    print("[Step 1/6] Cropping and concatenating raw segments...")
    crop_and_concat(FILES, MERGED_VIDEO)

    # Step 2) Denoise the merged video
    print("\n[Step 2/6] Denoising merged video (this may take a while)...")
    denoise_video(MERGED_VIDEO, DENOISED_VIDEO)

    # Step 3) Trim the merged and denoised videos
    print("\n[Step 3/6] Trimming videos (re-encoding for accurate cuts)...")
    trim_video(MERGED_VIDEO, MERGED_VIDEO_TRIMMED, TRIM_START_S, TRIM_END_S)
    trim_video(DENOISED_VIDEO, DENOISED_VIDEO_TRIMMED, TRIM_START_S, TRIM_END_S)

    # Step 4) Extract RGB means from original merged video (untrimmed)
    print("\n[Step 4/6] Extracting RGB means from original merged video...")
    df_org = per_frame_rgb_mean(MERGED_VIDEO)
    df_org.to_csv(RGB_ORG_CSV, index=False)
    print(f"[INFO] Saved: {RGB_ORG_CSV}")

    # Step 5) Extract RGB means from denoised merged video (untrimmed)
    print("\n[Step 5/6] Extracting RGB means from denoised merged video...")
    df_dn = per_frame_rgb_mean(DENOISED_VIDEO)
    df_dn.to_csv(RGB_DN_CSV, index=False)
    print(f"[INFO] Saved: {RGB_DN_CSV}")

    # Optional: RGB means from trimmed videos (often useful for alignment)
    print("\n[Optional] Extracting RGB means from trimmed videos...")
    df_org_trim = per_frame_rgb_mean(MERGED_VIDEO_TRIMMED)
    df_org_trim.to_csv(RGB_ORG_TRIM_CSV, index=False)
    print(f"[INFO] Saved: {RGB_ORG_TRIM_CSV}")

    df_dn_trim = per_frame_rgb_mean(DENOISED_VIDEO_TRIMMED)
    df_dn_trim.to_csv(RGB_DN_TRIM_CSV, index=False)
    print(f"[INFO] Saved: {RGB_DN_TRIM_CSV}")

    # Step 6) Saturated pixel analysis on the trimmed original video
    print("\n[Step 6/6] Saturated pixel analysis on trimmed original video...")
    saturated_pixel_analysis(MERGED_VIDEO_TRIMMED, INTERVALS, threshold=SAT_THRESHOLD)

    print("\n" + "=" * 60)
    print("COMPLETED SUCCESSFULLY")
    print(f"Outputs saved in: {CWD}")
    print("=" * 60)