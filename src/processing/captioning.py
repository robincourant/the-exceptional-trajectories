from typing import Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.processing.segmentation import (
    CAM_INDEX_TO_PATTERN,
    BODY_INDEX_TO_PATTERN,
    find_consecutive_chunks,
)


# ------------------------------------------------------------------------------------- #

MAX_TRIALS = 10

# ------------------------------------------------------------------------------------- #


def get_description(
    segment_index: int, start: int, end: int, index_to_pattern: Dict[int, str]
) -> str:
    return f"Between frames {start} and {end}: {index_to_pattern[segment_index]}"


def get_caption_prompt(
    traj_description: str,
    context_prompt: str,
    instruction_prompt: str,
    constraint_prompt: str,
    demonstration_prompt: str,
) -> str:
    raw_prompt = (
        f"{context_prompt}\n{instruction_prompt}\n{constraint_prompt}"
        f"{demonstration_prompt}{traj_description}\nDescription: "
    )
    caption_prompt = [{"role": "user", "content": raw_prompt}]
    return caption_prompt


def get_caption(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, caption_prompt: str
) -> str:
    encodeds = tokenizer.apply_chat_template(caption_prompt, return_tensors="pt")
    model_inputs = encodeds.to(model.device)
    generated_ids = model.generate(
        model_inputs,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=300,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        do_sample=True,
    )
    out_caption = tokenizer.batch_decode(generated_ids)[0].split("[/INST]")[-1][:-4]
    return out_caption


# ------------------------------------------------------------------------------------- #


def caption_trajectories(
    cam_segments: List[int],
    char_segments: List[int],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_prompt: str,
    instruction_prompt: str,
    constraint_prompt: str,
    demonstration_prompt: str,
) -> str:
    # Find consecutive chunks of patterns
    cam_chunks = find_consecutive_chunks(cam_segments)
    # Describe each chunk and join them
    cam_description = []
    for index, start, end in cam_chunks:
        description = get_description(index, start, end, CAM_INDEX_TO_PATTERN)
        cam_description.append(description)

    if char_segments is not None:
        char_chunks = find_consecutive_chunks(char_segments)
        char_description = []
        for index, start, end in char_chunks:
            description = get_description(index, start, end, BODY_INDEX_TO_PATTERN)
            char_description.append(description)

        traj_description = (
            f"\n\nOutline: Total frames {end}; "
            + "[Camera motion] "
            + ";".join(cam_description)
            + ". "
            + "[Main character motion] "
            + ";".join(char_description)
            + "."
        )
    else:
        traj_description = (
            f"\n\nOutline: Total frames {end}; "
            + "[Camera motion] "
            + ";".join(cam_description)
            + ". "
        )

    # Prompt the chatbot with the trajectory description
    caption_prompt = get_caption_prompt(
        traj_description,
        context_prompt,
        instruction_prompt,
        constraint_prompt,
        demonstration_prompt,
    )
    num_trials = 0
    while True:
        caption = get_caption(model, tokenizer, caption_prompt)
        if num_trials > MAX_TRIALS:
            break
        # If the caption contains the frame indices, prompt again
        elif any(char.isdigit() for char in caption) and num_trials == 0:
            caption_prompt.append({"role": "assistant", "content": caption})
            caption_prompt.append({"role": "user", "content": "Remove frame indices."})
        else:
            break
        num_trials += 1

    # Ad-hoc post-processing
    caption = caption.replace("Description: ", "").strip()

    return caption
