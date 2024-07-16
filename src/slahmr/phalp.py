"""
This code is adapted from https://github.com/vye16/slahmr
"""

from concurrent import futures
from pathlib import Path
import os
import subprocess
from typing import List

from lib.slahmr.slahmr.preproc.export_phalp import export_sequence_results

# import multiprocessing as mp


def launch_phalp(
    gpus: List[int], img_dir: Path, res_dir: Path, overwrite: bool = False
):
    """
    run phalp using GPU pool
    """
    # cur_proc = mp.current_process()
    # # 1-indexed processes
    # worker_id = cur_proc._identity[0] - 1 if len(cur_proc._identity) > 0 else 0
    # gpu = gpus[worker_id % len(gpus)]

    cmd_args = [
        "python lib/slahmr/slahmr/preproc/track.py",
        f"video.source={img_dir}",
        f"video.output_dir={res_dir}",
        f"overwrite={overwrite}",
        "detect_shots=True",
        "video.extract_video=False",
    ]
    cmd = " ".join(cmd_args)
    res = subprocess.call(cmd, shell=True)
    return res


def process_seq(gpus: List[int], img_dir: Path, res_dir: Path, overwrite: bool = False):
    """
    Run and export PHALP results
    """
    res = launch_phalp(gpus, img_dir, res_dir, overwrite)
    if res != 0:
        return 1

    # export the PHALP predictions
    out_root, out_name = os.path.split(res_dir)
    export_sequence_results(
        out_root,
        res_name=f"{out_name}/results",
        seq="demo_images",
        track_name="track_preds",
        shot_name="shot_idcs",
    )
    return 0


def run_phalp(
    root_dir: Path, gpus: List[int], img_dirname: str, out_dirname: str, overwrite: bool
) -> int:
    img_dir = root_dir / img_dirname
    res_dir = root_dir / out_dirname

    if len(gpus) > 1:
        with futures.ProcessPoolExecutor(max_workers=len(gpus)) as exe:
            res = exe.submit(process_seq, gpus, img_dir, res_dir, overwrite=overwrite)
    else:
        res = process_seq(gpus, img_dir, res_dir, overwrite=overwrite)
    return res
