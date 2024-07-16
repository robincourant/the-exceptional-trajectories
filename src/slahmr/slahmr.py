"""
This code is adapted from https://github.com/vye16/slahmr
"""

from pathlib import Path
import subprocess


def sliding_window(indices, window_lenght, window_overlap):
    windows = []
    for window_start in range(0, len(indices), window_lenght - window_overlap):
        window_end = window_start + window_lenght
        window = indices[window_start:window_end]
        if len(window) < window_lenght:
            remaining = window_lenght - len(window)
            window_start = max(0, window_start - remaining)
            window = indices[window_start:window_end]
        windows.append([window[0], window[-1]])

    return windows


def launch_slahmr(
    start_index: int,
    end_index: int,
    root_dir: Path,
    img_dirname: str,
    cam_dirname: str,
    track_dirname: str,
    shot_dirname: str,
    out_dirname: str,
):
    cmd_args = [
        "python scripts/extraction/run_slahmr.py",
        "data=video",
        "run_opt=True",
        "run_vis=False",
        f"data.start_idx={start_index}",
        f"data.end_idx={end_index}",
        f"data.root={root_dir}",
        f"data.src_path={root_dir}",
        "data.seq=clip",
        f"data.sources.images={root_dir/img_dirname}",
        f"data.sources.cameras={root_dir/cam_dirname}",
        f"data.sources.tracks={root_dir/track_dirname}",
        f"data.sources.shots={root_dir/shot_dirname}",
        f"data.name='{str(start_index).zfill(5)}_{str(end_index).zfill(5)}'",
        f"log_root={root_dir}",
        f"log_dir={root_dir}",
        f"exp_name={out_dirname}",
    ]
    cmd = " ".join(cmd_args)
    res = subprocess.call(cmd, shell=True)
    return res


def run_slahmr(
    root_dir: Path,
    img_dirname: str,
    cam_dirname: str,
    track_dirname: str,
    shot_dirname: str,
    out_dirname: str,
):
    num_frames = len(list((root_dir / img_dirname).glob("*.jpg")))
    windows = sliding_window(range(num_frames), 100, 10)
    for start_index, end_index in windows:
        res = launch_slahmr(
            int(start_index),
            int(end_index),
            root_dir,
            img_dirname,
            cam_dirname,
            track_dirname,
            shot_dirname,
            out_dirname,
        )
        if res != 0:
            return 1
    return 0
