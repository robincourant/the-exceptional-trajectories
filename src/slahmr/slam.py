"""
This code is adapted from https://github.com/vye16/slahmr
"""

from typing import List
import subprocess
from concurrent import futures
from pathlib import Path

# import multiprocessing as mp


def launch_slam(gpus: List[int], img_dir: Path, res_dir: Path, overwrite: bool = False):
    """
    run slam using GPU pool
    """
    # cur_proc = mp.current_process()
    # # 1-indexed processes
    # worker_id = cur_proc._identity[0] - 1 if len(cur_proc._identity) > 0 else 0
    # gpu = gpus[worker_id % len(gpus)]

    cmd_args = [
        "python lib/slahmr/slahmr/preproc/run_slam.py",
        "-i",
        str(img_dir),
        "--map_dir",
        str(res_dir),
        "--disable_vis",
    ]
    if overwrite:
        cmd_args.append("--overwrite")
    cmd = " ".join(cmd_args)
    res = subprocess.call(cmd, shell=True)

    return res


def run_slam(
    root_dir: Path, gpus: List[int], img_dirname: str, out_dirname: str, overwrite: bool
) -> int:
    img_dir = root_dir / img_dirname
    res_dir = root_dir / out_dirname

    if len(gpus) > 1:
        with futures.ProcessPoolExecutor(max_workers=len(gpus)) as exe:
            res = exe.submit(launch_slam, gpus, img_dir, res_dir, overwrite=overwrite)
    else:
        res = launch_slam(gpus, img_dir, res_dir, overwrite=overwrite)

    return res
