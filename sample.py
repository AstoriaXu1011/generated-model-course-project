import argparse
from PIL import Image
import torch

from minigpt4.common.config import Config
from minigpt4.common.eval_utils import init_model, prepare_texts
from minigpt4.conversation.conversation import (
    CONV_VISION_minigptv2,
    CONV_VISION_Vicuna0,
    CONV_VISION_LLama2,
)


DEFAULT_IMAGE_PATH = "sample_result/sample_in/image.png"
DEFAULT_QUESTION = "What is shown in the image?"


def build_conv_template(cfg):
    if cfg.model_cfg.arch == "minigpt_v2":
        conv = CONV_VISION_minigptv2.copy()
        conv.system = ""
        return conv
    conv_dict = {
        "pretrain_vicuna0": CONV_VISION_Vicuna0,
        "pretrain_llama2": CONV_VISION_LLama2,
    }
    return conv_dict.get(cfg.model_cfg.model_type, CONV_VISION_Vicuna0).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniGPT-4 sample inference (no GUI).")
    parser.add_argument("--cfg-path", required=True, help="Path to eval config yaml.")
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH, help="Path to image.")
    args = parser.parse_args()

    cfg = Config(args)
    model, vis_processor = init_model(args)
    model.eval()

    conv = build_conv_template(cfg)
    image = Image.open(args.image).convert("RGB")
    image_tensor = vis_processor(image)

    texts = prepare_texts([DEFAULT_QUESTION], conv)
    with torch.no_grad():
        answers = model.generate(
            image_tensor.unsqueeze(0),
            texts,
            max_new_tokens=20,
            do_sample=False,
        )

    print(f"Q: {DEFAULT_QUESTION}")
    print(f"A: {answers[0]}")


if __name__ == "__main__":
    main()
