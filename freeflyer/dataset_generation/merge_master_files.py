"""
Merge multiple master_file*.json files and output one JSON with exactly
100 commands per behavior mode (0..26). For each mode, concatenates all
entries from all input files, deduplicates by text, then keeps the first
100 and renumbers command_id to 0..99.

Usage:
  python -m freeflyer.dataset_generation.merge_master_files \\
    dataset/master_file_gen_me.json dataset/master_file_gen_me2.json \\
    -o dataset/master_file.json
  # or from dataset_generation/:
  python merge_master_files.py ../dataset/master_file_gen_me.json ../dataset/master_file_gen_me2.json -o ../dataset/master_file.json
"""
import argparse
import json
from pathlib import Path

N_BEHAVIOR_MODES = 27
TARGET_PER_MODE = 100


def load_master(path: Path) -> dict:
    """Load a master JSON. Keys are mode indices "0".."26", values are lists of {command_id, text}."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_and_cap(
    input_paths: list[Path],
    target_per_mode: int = TARGET_PER_MODE,
    n_modes: int = N_BEHAVIOR_MODES,
) -> dict:
    """
    Merge all input master files. For each behavior mode:
    - Concatenate all entries from all files (order: first file first, then second, etc.)
    - Deduplicate by text (keep first occurrence)
    - Take first target_per_mode and set command_id to 0..target_per_mode-1
    """
    merged = {}
    for mode_key in (str(i) for i in range(n_modes)):
        seen_texts = set()
        ordered_entries = []
        for path in input_paths:
            data = load_master(path)
            if mode_key not in data:
                continue
            for entry in data[mode_key]:
                text = entry.get("text", "").strip()
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)
                ordered_entries.append(text)
        # Cap to target_per_mode and assign command_id 0..target_per_mode-1
        capped = ordered_entries[:target_per_mode]
        merged[mode_key] = [{"command_id": i, "text": t} for i, t in enumerate(capped)]
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge master JSON files and output 100 commands per behavior mode."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Paths to master_file*.json files to merge (order preserved when concatenating)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output merged JSON path",
    )
    parser.add_argument(
        "--per-mode",
        type=int,
        default=TARGET_PER_MODE,
        help=f"Target number of commands per behavior mode (default: {TARGET_PER_MODE})",
    )
    args = parser.parse_args()

    for p in args.inputs:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    merged = merge_and_cap(args.inputs, target_per_mode=args.per_mode)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in merged.values())
    print(f"Merged {len(args.inputs)} file(s) -> {args.output}")
    print(f"Total: {total} commands ({args.per_mode} per mode × {N_BEHAVIOR_MODES} modes)")
    for mode_key in sorted(merged.keys(), key=int):
        n = len(merged[mode_key])
        if n < args.per_mode:
            print(f"  Mode {mode_key}: {n} (below target {args.per_mode})")


if __name__ == "__main__":
    main()

# python3 merge_master_files.py ../dataset/master_file_gen_me.json ../dataset/master_file_gen_me2.json -o ../dataset/ master_file.json