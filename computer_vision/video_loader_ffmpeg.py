import subprocess

import numpy as np


def ffmpeg_frame_generator(video_path, pix_fmt="rgb24", width=None, height=None):
    """
    Generator that yields video frames using ffmpeg.
    Args:
        video_path (str): Path to the video file
        pix_fmt (str): Pixel format (default: rgb24)
        width (int, optional): Width to resize to
        height (int, optional): Height to resize to
    Yields:
        np.ndarray: Frames as numpy arrays of shape (height, width, 3) for rgb24 format.
    """
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-f",
        "image2pipe",
        "-pix_fmt",
        pix_fmt,
        "-vcodec",
        "rawvideo",
        "-",
    ]
    if width and height:
        cmd[2:2] = ["-vf", f"scale={width}:{height}"]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
    )

    # Calculate frame size
    if width is None or height is None:
        # Extract original size using ffprobe
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            video_path,
        ]
        out = subprocess.check_output(probe_cmd).decode().strip()
        w, h = map(int, out.split("x"))

        width = width or w
        height = height or h

    frame_size = width * height * 3  # rgb24: 3 bytes per pixel

    while True:
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            break
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        yield frame.reshape((height, width, 3))
