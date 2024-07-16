"""Caption all trajectories."""

from pathlib import Path

import hydra
from evo.tools.file_interface import read_kitti_poses_file
import numpy as np
from omegaconf import DictConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from helper.files import save_txt
from helper.progress import PROGRESS
from src.processing.captioning import caption_trajectories
from src.processing.segmentation import (
    segment_rigidbody_trajectories,
    segment_translation_trajectories,
)


# ------------------------------------------------------------------------------------- #


@hydra.main(
    version_base=None,
    config_path="./configs/captioning/",
    config_name="caption_cam+char.yaml",
)
def launch_captioning(config: DictConfig):
    data_dir = Path(config.data_dir)
    traj_dir = data_dir / "traj_raw"

    # Get all trajectory paths
    traj_paths = [
        traj_path for traj_path in sorted(traj_dir.iterdir(), reverse=config.reverse)
    ]

    if "char" in config:
        char_dir = data_dir / "char_raw"
        char_paths = [char_dir / (x.stem + ".npy") for x in traj_paths]

    # Initialize the chatbot and its tokenizer
    if "llm" in config:
        tokenizer = AutoTokenizer.from_pretrained(Path(config.llm.model_dir))

        model = {}
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if "Mixtral-8x7B" in Path(config.llm.model_dir).stem:
            model["attn_implementation"] = "flash_attention_2"
            model["load_in_4bit"] = True
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model["quantization_config"] = bnb_config
            model["device_map"] = config.llm.device
            # model["device_map"] = "auto"
        else:
            model["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            Path(config.llm.model_dir), **model
        )

        if "Mixtral-8x7B" not in Path(config.llm.model_dir).stem:
            model.to(config.llm.device)
    else:
        # Perform segmentation only
        tokenizer, model = None, None

    with PROGRESS:
        task = PROGRESS.add_task("[green]Processing...", total=len(traj_paths), step=0)
        for traj_index, traj_path in enumerate(traj_paths):
            traj = torch.from_numpy(
                np.stack(read_kitti_poses_file(traj_path).poses_se3)
            )
            # ------------------------------------------------------------------------- #
            traj_name = traj_path.stem

            cam_segment_path = data_dir / "cam_segments" / (traj_name + ".npy")
            cam_segment_path.parent.mkdir(parents=True, exist_ok=True)
            pass_sample = cam_segment_path.exists()

            if "char" in config:
                char_path = char_paths[traj_index]
                char = np.load(char_path)
                char_segment_path = data_dir / "char_segments" / (traj_name + ".npy")
                char_segment_path.parent.mkdir(parents=True, exist_ok=True)
                pass_sample = pass_sample and char_segment_path.exists()

            if model is not None:
                caption_path = data_dir / config.name / (traj_name + ".txt")
                caption_path.parent.mkdir(parents=True, exist_ok=True)
                pass_sample = pass_sample and caption_path.exists()

            if (not config.overwrite) and pass_sample:
                PROGRESS.update(task, advance=1)
                continue

            # ------------------------------------------------------------------------- #

            # Segment trajectory
            cam_segments = segment_rigidbody_trajectories(
                traj,
                cam_static_threshold=config.cam.static_threshold,
                cam_diff_threshold=config.cam.diff_threshold,
                fps=config.fps,
                min_chunk_size=config.min_chunk_size,
                smoothing_window_size=config.smoothing_window_size,
            )
            np.save(cam_segment_path, cam_segments)

            if "char" in config:
                char_segments = segment_translation_trajectories(
                    char,
                    char_static_threshold=config.char.static_threshold,
                    char_diff_threshold=config.char.diff_threshold,
                    fps=config.fps,
                    min_chunk_size=config.min_chunk_size,
                    smoothing_window_size=config.smoothing_window_size,
                )
                np.save(char_segment_path, char_segments)
            else:
                char_segments = None

            # Infer the caption
            if model is not None:
                caption = caption_trajectories(
                    cam_segments=cam_segments,
                    char_segments=char_segments,
                    model=model,
                    tokenizer=tokenizer,
                    context_prompt=config.llm.prompt.context,
                    instruction_prompt=config.llm.prompt.instruction,
                    constraint_prompt=config.llm.prompt.constraint,
                    demonstration_prompt=config.llm.prompt.demonstration,
                )
                save_txt(caption, caption_path)

            PROGRESS.update(task, advance=1)


if __name__ == "__main__":
    launch_captioning()
