import os
import sys
import logging
from pathlib import Path

from evo.tools.file_interface import read_kitti_poses_file
import numpy as np
import bpy

sys.path.append(os.path.dirname(bpy.data.filepath))

from blender.src.render import render  # noqa
from blender.src.tools import delete_objs  # noqa

logger = logging.getLogger(__name__)


class Renderer:
    def __init__(self):
        self.obj_names = None

    def render_cli(
        self, input_dir: str, sample_id: str, selected_rate: float, mode: str
    ):
        if self.obj_names is not None:
            # remove every object created
            delete_objs(self.obj_names)
            delete_objs(["Plane", "myCurve", "Cylinder"])

        input_dir = Path(input_dir)

        traj_path = input_dir / "traj_raw" / (sample_id + ".txt")
        char_path = input_dir / "vert_raw" / (sample_id + ".npy")
        cam_seg_path = input_dir / "cam_segments" / (sample_id + ".npy")
        char_seg_path = input_dir / "char_segments" / (sample_id + ".npy")

        cam_segments = np.load(cam_seg_path, allow_pickle=True)
        cam_segments = np.concatenate([[cam_segments[0]], cam_segments])
        char_segments = np.load(char_seg_path, allow_pickle=True)
        char_segments = np.concatenate([[char_segments[0]], char_segments])

        traj = np.array(read_kitti_poses_file(traj_path).poses_se3)
        traj = traj[:, [0, 2, 1]]
        traj[:, 2] = -traj[:, 2]
        char = np.load(char_path, allow_pickle=True)[()]
        vertices = char["vertices"]
        vertices = vertices[..., [0, 2, 1]]
        vertices[..., 2] = -vertices[..., 2]
        faces = char["faces"]
        faces = faces[..., [0, 2, 1]]

        nframes = traj.shape[0]
        # Set the final frame of the playback
        if "video" in mode:
            bpy.context.scene.frame_end = nframes - 1
        num = int(selected_rate * nframes)
        self.obj_names = render(
            traj=traj,
            vertices=vertices,
            faces=faces,
            cam_segments=cam_segments,
            char_segments=char_segments,
            # ---------------------------- #
            denoising=True,
            oldrender=True,
            res="low",
            canonicalize=True,
            exact_frame=0.5,
            num=num,
            mode=mode,
            init=False,
        )
