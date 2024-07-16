from datetime import datetime
from pathlib import Path
import random
import time
from typing import Any, Dict, Callable

import pandas as pd

NUM_REF_TRIAL = 10


class Logger(object):
    def __init__(self, save_dir: Path):
        super(Logger, self).__init__()
        self.timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M")
        self.save_path = self.get_filename(save_dir, f"{self.timestamp}-logs.csv")
        self.logs = self.set_buffer()
        self.buffer = self.set_buffer()
        self._save()

    @staticmethod
    def get_filename(save_dir: Path, filename: str) -> str:
        k = random.randint(1, 10e5)
        unique_filename = filename
        while (save_dir / unique_filename).exists():
            unique_filename = f"{filename[:-4]}_{str(k).zfill(6)}{filename[-4:]}"
            k = random.randint(1, 10e5)
        return save_dir / unique_filename

    def set_buffer(self):
        buffer = pd.DataFrame(
            columns=[
                "start_time",
                "video_id",
                "task",
                "status",
                "duration",
                "res",
                "num_frames",
            ]
        )
        return buffer

    def run(
        self,
        function: Callable,
        shot_id: str,
        file_dir: Path,
        num_frames: int,
        *args,
        **kwargs,
    ):
        # Initialize logs
        start_time = time.time()
        out_dict = {
            "start_time": start_time,
            "video_id": shot_id,
            "task": getattr(function, "__name__", repr(function)),
            "status": "RUNNING",
            "num_frames": num_frames,
        }
        self._log(out_dict)
        self._save()

        # Run function
        try:
            res = function(file_dir, *args, **kwargs)
            status = "SUCCESS" if res == 0 else "FAILED"
        except Exception as e:
            res = e
            status = "FAILED"
        duration = time.time() - start_time
        start_time = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")

        # Update logs
        out_dict["status"] = status
        out_dict["duration"] = duration
        out_dict["res"] = res
        self.logs.iloc[-1] = out_dict
        self.buffer.iloc[-1] = out_dict
        self._save()

        return out_dict

    def missing_data(self, shot_id: str, num_frames: int):
        # Initialize logs
        start_time = time.time()
        out_dict = {
            "start_time": start_time,
            "video_id": shot_id,
            "task": "read_video",
            "status": "FAILED",
            "num_frames": num_frames,
            "duration": 0.0,
            "res": "Missing data",
        }
        self._log(out_dict)
        self._save()

    def skipping_data(self, shot_id: str, num_frames: int, res: str = "Skipping data"):
        # Initialize logs
        start_time = time.time()
        out_dict = {
            "start_time": start_time,
            "video_id": shot_id,
            "task": "read_video",
            "status": "SKIPPED",
            "num_frames": num_frames,
            "duration": 0.0,
            "res": res,
        }
        self._log(out_dict)
        self._save()

    def _log(self, out_dict: Dict[str, Any]):
        new_log = pd.DataFrame([out_dict])
        self.logs = pd.concat([self.logs, new_log], ignore_index=True)
        self.buffer = pd.concat([self.buffer, new_log], ignore_index=True)

    def _save(self):
        self.logs.to_csv(self.save_path, index=False)

    def update_ref(self, clip_filename: Path, ref_filename: Path):
        def aggregate_status(group):
            if "FAILURE" in group.values:
                return "FAILURE"
            else:
                return group.iloc[0]

        def split_video_id(video_id):
            parts = video_id.split("-")
            year = int(float(parts[0]))
            shot_idx = int(parts[-1])
            video_id = "-".join(parts[1:-1])
            return pd.Series([year, video_id, shot_idx])

        clip_df = pd.read_csv(clip_filename)
        clip_df["year"] = clip_df["year"].astype(int)
        result_df = (
            self.buffer.groupby("video_id")["status"]
            .agg(aggregate_status)
            .reset_index()
        )

        result_df[["year", "id", "shot_idx"]] = result_df["video_id"].apply(
            split_video_id
        )

        merged_df = clip_df.merge(
            result_df,
            right_on=["id", "year"],
            left_on=["videoid", "upload_year"],
            how="inner",
            suffixes=["", "_"],
        )
        merged_df = merged_df.drop(columns=["video_id", "year_", "id"])
        merged_df["shot_idx"] = merged_df["shot_idx"].astype(int)
        merged_df["year"] = merged_df["year"].astype(int)

        try:
            ref_df = pd.read_csv(ref_filename, error_bad_lines=False)
            ref_df = pd.concat([ref_df, merged_df], ignore_index=True)
            ref_df = ref_df.sort_values(
                by=["videoid", "clip_idx", "shot_idx"], ignore_index=True
            )
            ref_df.to_csv(ref_filename, index=False)
            self.buffer = self.set_buffer()

        except:
            k = 1
            unique_filename = ref_filename
            while (Path(ref_filename).parent / unique_filename).exists():
                unique_filename = f"{ref_filename.stem}_{k}{ref_filename.suffix}"
                k += 1
            ref_df = merged_df
            ref_df = ref_df.sort_values(
                by=["videoid", "clip_idx", "shot_idx"], ignore_index=True
            )
            ref_df.to_csv(ref_filename, index=False)
            self.buffer = self.set_buffer()
