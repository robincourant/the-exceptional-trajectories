from pathlib import Path
from typing import Callable, List, Tuple

import decord
from decord import VideoReader, cpu
import pandas as pd
from tqdm import tqdm
import torch
from torchtyping import TensorType
from torch.utils.data import Dataset

from helper.files import load_txt

num_frames, width, height = None, None, None

decord.bridge.set_bridge("torch")


class CMDataset(Dataset):
    """CondensedMovies dataset."""

    def __init__(
        self,
        data_dir: str,
        metadata_filename: str = "clips.csv",
        frame_step: int = 1,
        video_extension: str = ".mkv",
        transform: Callable = None,
        **kwargs,
    ):
        super(CMDataset, self).__init__()
        self.data_dir = Path(data_dir)
        self.metadata_filename = Path(metadata_filename)
        self.frame_step = frame_step
        self.ext = video_extension
        self.transform = transform

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> List[Tuple[str, int, int]]:
        """ """
        metadata_path = self.data_dir / "metadata" / self.metadata_filename
        metadata_list = pd.read_csv(metadata_path)

        metadata = []
        for index, row in tqdm(metadata_list.iterrows(), total=metadata_list.shape[0]):
            clip_path = (
                self.data_dir
                / "videos"
                / str(int(row["upload_year"]))
                / (str(row["videoid"]) + self.ext)
            )

            shot_path = (
                self.data_dir
                / "shots"
                / "raw"
                / str(int(row["upload_year"]))
                / str(row["videoid"])
                / "shot_txt"
                / (str(row["videoid"]) + ".txt")
            )
            if shot_path.exists():
                shot_txt = load_txt(shot_path)
                for shot_index, shot in enumerate(shot_txt.split("\n")):
                    shot_boundaries = [int(t) for t in shot.split(" ")[:2]]
                    shot_metadata = row.to_dict() | {"shot_index": shot_index}
                    metadata.append((clip_path, shot_boundaries, shot_metadata))
            # Consider the whole video as a single shot
            else:
                print(f"WARNING: Shot file not found: {shot_path}")
                try:
                    with open(clip_path, "rb") as f:
                        vr = VideoReader(f, ctx=cpu(0))
                    shot_metadata = row.to_dict() | {"shot_index": 0}
                    metadata.append((clip_path, [0, len(vr) - 1], shot_metadata))
                except FileNotFoundError:
                    pass

        return metadata

    @staticmethod
    def _load_shot(
        video_path: str, start_frame: int, end_frame: int, frame_step: int
    ) -> TensorType["num_frames", "height", "width", 3]:
        try:
            with open(video_path, "rb") as f:
                vr = VideoReader(f, ctx=cpu(0))
            step = frame_step if end_frame - start_frame > 100 else 1
            frames = vr.get_batch(range(start_frame, end_frame + 1, step))
            return frames
        except RuntimeError:
            return torch.empty(())

    def __getitem__(self, index):
        video_path, (start_frame, end_frame), metadata = self.metadata[index]
        frames = self._load_shot(video_path, start_frame, end_frame, self.frame_step)

        if self.transform:
            frames = self.transform(frames)

        return frames, metadata

    def __len__(self):
        return len(self.metadata)
