import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_id_map(
    path: str,
    list_id_key: str,
    list_value_key: str,
    dict_value_key: str,
) -> Dict[str, str]:
    data = load_json(path)
    if isinstance(data, dict):
        if not data:
            return {}
        sample = next(iter(data.values()))
        if isinstance(sample, dict):
            return {
                str(k): str(v.get(dict_value_key))
                for k, v in data.items()
                if dict_value_key in v
            }
        return {str(k): str(v) for k, v in data.items()}
    if isinstance(data, list):
        out = {}
        for item in data:
            if list_id_key in item and list_value_key in item:
                out[str(item[list_id_key])] = str(item[list_value_key])
        return out
    raise ValueError(f"Unsupported id map format in {path}")


def iter_problems(data) -> Iterable[Tuple[str, dict]]:
    if isinstance(data, dict):
        for k, v in data.items():
            yield str(k), v
        return
    if isinstance(data, list):
        for item in data:
            if "id" in item:
                yield str(item["id"]), item
        return
    raise ValueError("Unsupported problems.json format")


def build_question(raw: dict, append_hint: bool) -> str:
    question = raw.get("question", "").strip()
    hint = raw.get("hint")
    if append_hint and hint:
        question = f"{question} Hint: {hint.strip()}"
    return question


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare IconQA problems.json into MiniGPT evaluation format."
    )
    parser.add_argument("--problems", required=True, help="Path to problems.json")
    parser.add_argument(
        "--img-root",
        required=True,
        help="Root folder containing per-id image folders (expects <id>/image.png)",
    )
    parser.add_argument(
        "--out", required=True, help="Output json path (e.g. choose_text_val.json)"
    )
    parser.add_argument(
        "--split", default="val", help="Split name to export (default: val)"
    )
    parser.add_argument(
        "--keep-types",
        default="choose_text",
        help="Comma-separated ques_type list to keep (default: choose_text)",
    )
    parser.add_argument(
        "--append-hint",
        action="store_true",
        help="Append hint to question when available",
    )
    parser.add_argument(
        "--id-map",
        default="",
        help="Optional json mapping from problem id to image id",
    )
    parser.add_argument(
        "--list-id-key",
        default="id",
        help="List mapping: key name for problem id (default: id)",
    )
    parser.add_argument(
        "--list-value-key",
        default="image_id",
        help="List mapping: key name for image id (default: image_id)",
    )
    parser.add_argument(
        "--dict-value-key",
        default="image_id",
        help="Dict mapping: value key name for image id (default: image_id)",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Skip entries whose image.png is missing",
    )
    args = parser.parse_args()

    problems = load_json(args.problems)
    keep_types = {t.strip() for t in args.keep_types.split(",") if t.strip()}

    id_map = {}
    if args.id_map:
        id_map = load_id_map(
            args.id_map, args.list_id_key, args.list_value_key, args.dict_value_key
        )

    kept: List[dict] = []
    skipped_type = 0
    skipped_split = 0
    skipped_missing_img = 0
    skipped_missing_choice = 0

    for pid, raw in iter_problems(problems):
        if raw.get("split") != args.split:
            skipped_split += 1
            continue

        qtype = raw.get("ques_type")
        if qtype not in keep_types:
            skipped_type += 1
            continue

        choices = raw.get("choices")
        if not choices or not isinstance(choices, list):
            skipped_missing_choice += 1
            continue

        image_id = id_map.get(pid, pid)
        image_path = os.path.join(args.img_root, image_id, "image.png")
        if args.check_images and not os.path.exists(image_path):
            skipped_missing_img += 1
            continue

        item = {
            "image_id": image_id,
            "question": build_question(raw, args.append_hint),
            "choices": choices,
            "answer": raw.get("answer"),
        }
        kept.append(item)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(kept, f, ensure_ascii=True)

    print(
        f"saved {len(kept)} items to {args.out}. "
        f"skipped split={skipped_split}, type={skipped_type}, "
        f"no_choices={skipped_missing_choice}, missing_img={skipped_missing_img}"
    )


if __name__ == "__main__":
    main()
