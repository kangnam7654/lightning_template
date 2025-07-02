import os
import subprocess


def extract_frames(video_path: str, output_dir: str, fps: int | float | None = 1):
    """
    Extracts frames from a video file using ffmpeg.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory where extracted images will be saved.
        fps (int or float): Number of frames per second to extract.

    Example:
        extract_frames("video.mp4", "frames/", fps=2)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "frame_%06d.jpg")

    if fps is None:
        # If fps is None, extract all frames
        cmd = ["ffmpeg", "-i", video_path, output_pattern]
    else:
        cmd = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", output_pattern]

    try:
        subprocess.run(cmd, check=True)
        print(f"Frames saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print("Error extracting frames:", e)
