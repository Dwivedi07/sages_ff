import os
from typing import Dict, List, Optional
import json
from pathlib import Path

from openai import OpenAI

# Optional: only needed if you add Gemini paths later
# import google.generativeai as genai
# from google.api_core.exceptions import ResourceExhausted

#                  y ↑

#          +----------------+----------------+----------------+
#          |   region 6     |   region 7     |   region 8     |
#          |   left_top     |   center_top   |   right_top    |
#  high y  |                |                |                |
#          +----------------+----------------+----------------+
#          |   region 3     |   region 4     |   region 5     |
#          |   left_mid     |   center_mid   |   right_mid    |
#  mid y   |                |                |                |
#          +----------------+----------------+----------------+
#          |   region 0     |   region 1     |   region 2     |
#          | left_bottom    | center_bottom  | right_bottom   |
#  low y   |                |                |                |
#          +----------------+----------------+----------------+

#                 left x          center x         right x

# --------------------- table space boundary at x = 1.2 --------------------> x

N_BEHAVIOR_MODES = 27
K_T_BY_TIME_ID = {0: 60, 1: 80, 2: 100}

# 3×3 goal grid (row-major): machine id + short cardinal phrase for prompts only
REGION_INFO = [
    ("left_bottom", "lower-left", "low-y low-x"),
    ("center_bottom", "bottom-center", "low-y center-x"),
    ("right_bottom", "lower-right", "low-y high-x"),
    ("left_mid", "middle-left", "mid-y low-x"),
    ("center_mid", "center", "mid-y center-x"),
    ("right_mid", "middle-right", "mid-y high-x"),
    ("left_top", "upper-left", "high-y low-x"),
    ("center_top", "top-center", "high-y center-x"),
    ("right_top", "upper-right", "high-y high-x"),
]

# Second column of REGION_INFO — every generated command must name the goal with one of these
GOAL_ZONE_PHRASES = tuple(z for _, z, _ in REGION_INFO)


def goal_zone_phrase_for_mode(mode_id: int) -> str:
    """Hyphenated goal-sector label for this behavior (region_id = mode_id % 9)."""
    return REGION_INFO[int(mode_id) % 9][1]


def text_names_goal_zone(text: str, zone_phrase: str) -> bool:
    """True if sentence includes the zone phrase (hyphenated or with a space)."""
    if not text or not zone_phrase:
        return False
    t = text.lower()
    z = zone_phrase.lower()
    return z in t or z.replace("-", " ") in t


# UPDATED: Added moderate cadence and clearer distinctions
TIME_HORIZON_INFO = {
    0: ("short", "short horizon", "time-prioritized", "agile profile", "between 20-24s", "in 60 steps"),
    1: ("mid", "mid horizon", "balanced timing", "moderate cadence", "within 24-32s", "in 80 steps"),
    2: ("full", "full horizon", "margin-prioritized", "extended coast", "nearly 32-40s", "in 100 steps"),
}

def _speed_mode_for_behavior(mode_idx: int) -> str:
    """Vocabulary bucket for prompts: short k_T → fast, mid → moderate, long → slow."""
    time_id = mode_idx // 9
    if time_id == 0:
        return "fast"
    if time_id == 1:
        return "moderate"
    return "slow"


def _build_behavior_meta() -> Dict[int, dict]:
    meta = {}
    for bm in range(N_BEHAVIOR_MODES):
        region_id = bm % 9
        time_id = bm // 9
        k_T = K_T_BY_TIME_ID[time_id]
        short_r, zone_plain, _band_hint = REGION_INFO[region_id]
        _, long_t, tempo_hint, cadence_hint, wall_time_hint, steps_hint = TIME_HORIZON_INFO[
            time_id
        ]
        label = f"{short_r}_t{k_T}"
        canonical = (
            f"Reach a goal in the {zone_plain} goal zone with a {long_t} ({steps_hint}, "
            f"{wall_time_hint}); KOZ-compliant trajectory ({tempo_hint})."
        )
        geom_hints = [zone_plain, _band_hint, tempo_hint, cadence_hint, wall_time_hint, steps_hint]
        meta[bm] = dict(label=label, canonical=canonical, geom_hints=geom_hints, k_T=k_T)
    return meta

BEHAVIOR_META = _build_behavior_meta()

def annotate_traj_behaviors_gpt(
    ids: List[int],
    api_key: str,
    model: str = "gpt-4o",
    speed_mode: str = "moderate",
    *,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> Dict[int, Dict[str, str]]:
    # UPDATED: Mode-specific vocabulary hints
    speed_hints = {
        "fast": ["agile", "brisk", "quick", "snappy", "rapid", "swift",  "between 20-24s", "in 60 steps"],
        "moderate": ["steady", "moderate", "deliberate", "balanced", "intermediate", "within 24-32s", "in 80 steps"],
        "slow": ["sluggish", "slow", "gradual", "leisurely", "crawling", "measured", "nearly 32-40s", "in 100 steps"],
    }

    speed_clause = {
        "fast": "FAST MODE: Target ~20-24s. Use words like 'agile' or 'rapid'.",
        "moderate": "MODERATE MODE: Target ~24-32s. Use words like 'steady' or 'balanced'.",
        "slow": "SLOW MODE: Target ~32-40s. Use words like 'sluggish' or 'gradual'.",
    }[speed_mode]

    client = OpenAI(api_key=api_key)
    out: Dict[int, Dict[str, str]] = {}

    zone_catalog = ", ".join(f'"{p}"' for p in GOAL_ZONE_PHRASES)
    system_msg = (
        "You are an expert GNC writer. Generate a single technical sentence for a robot trajectory.\n"
        f"Goal location vocabulary (exact hyphenated phrases only): {zone_catalog}.\n"
        "Every sentence MUST include the REQUIRED goal-zone phrase given in the user message, verbatim "
        "(same spelling and hyphens).\n"
        "Speed rule: use EITHER a speed adjective OR a time/horizon hint, not both in the same sentence."
    )

    for i, mode_id in enumerate(ids):
        meta = BEHAVIOR_META[mode_id]
        hints = ", ".join(speed_hints[speed_mode])
        zone_required = goal_zone_phrase_for_mode(mode_id)

        base_user = (
            f"REQUIRED goal-sector phrase (must appear in your sentence): \"{zone_required}\".\n"
            f"Context: {meta['canonical']}\n"
            f"Speed Guidance: {speed_clause}\n"
            f"Allowed adjectives for this mode: {hints}\n"
            "If you use a time estimate or horizon length, omit the speed adjective; if you use a speed "
            "adjective, omit time/horizon numbers.\n"
            "Format: ONE sentence, ≤18 words, technical tone (KOZ, standoff, or RCS where natural)."
        )

        desc = ""
        for attempt in range(5):
            retry_note = (
                ""
                if attempt == 0
                else f"\n\nRetry {attempt}: Previous text did not contain \"{zone_required}\". "
                f'Output ONE new sentence that MUST include exactly "{zone_required}".'
            )
            user_msg = base_user + retry_note
            try:
                rsp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                raw = rsp.choices[0].message.content
                desc = (raw or "").strip()
                if text_names_goal_zone(desc, zone_required):
                    break
            except Exception as e:
                desc = f"Error in generation: {e}"

        if not text_names_goal_zone(desc, zone_required) and not str(desc).startswith("Error in generation"):
            desc = f"KOZ-safe transit into the {zone_required} goal zone with controlled thrust."

        out[i] = {"id": mode_id, "behavior": meta["label"], "description": desc}
    return out

################################## HELPER VARS #################################
# 27 behavior modes: behavior_mode = 9 * time_id + region_id
#   region_id 0..8: 3×3 grid (row-major) for x > 1.2
#   time_id 0..2:   k_T = 60, 80, 100

############################## ANNOTATE FUNCTIONS ##############################
# ======================================================
#  HELPER FUNCTIONS
# ======================================================
def _diversity_schedules(n: int) -> List[Dict]:
    base = [
        dict(temperature=0.6, top_p=0.9),
        dict(temperature=0.7, top_p=0.9),
        dict(temperature=0.8, top_p=0.95),
        dict(temperature=0.9, top_p=0.97),
        dict(temperature=1.0, top_p=0.98),
    ]
    runs_per = max(1, n // len(base))
    sched = []
    for cfg in base:
        sched.append({**cfg, "runs": runs_per})
    total = sum(s["runs"] for s in sched)
    if total < n:
        sched[-1]["runs"] += (n - total)
    return sched


def generate_100_prompts_for_mode(
    mode_idx: int,
    api_key: str,
    model_name: str = "gpt-4o",
    target_n: int = 100,
) -> List[str]:
    """
    mode_idx in {0..26}, matching BEHAVIOR_META (9 regions × 3 time horizons).
    """
    speed_mode = _speed_mode_for_behavior(mode_idx)
    zone_fixed = goal_zone_phrase_for_mode(mode_idx)

    prompts: List[str] = []
    seen = set()
    sched = _diversity_schedules(target_n)

    def _keep(txt: str) -> bool:
        t = txt.strip()
        if not t or t in seen:
            return False
        if t.startswith("Error in generation"):
            return False
        return text_names_goal_zone(t, zone_fixed)

    for block in sched:
        runs = block["runs"]
        result = annotate_traj_behaviors_gpt(
            ids=[mode_idx] * runs,          # <<< key change: use mode index directly
            api_key=api_key,
            model=model_name,
            max_tokens=30,
            temperature=block["temperature"],
            top_p=block["top_p"],
            presence_penalty=0.3,
            frequency_penalty=0.2,
            speed_mode=speed_mode,
        )

        for k in sorted(result.keys()):
            txt = result[k]["description"].strip()
            if _keep(txt):
                seen.add(txt)
                prompts.append(txt)

    # top-up loop (bounded — strict goal-zone filter may need many rounds)
    topup_round = 0
    while len(prompts) < target_n and topup_round < 2000:
        topup_round += 1
        top = annotate_traj_behaviors_gpt(
            ids=[mode_idx] * 10,
            api_key=api_key,
            model=model_name,
            max_tokens=30,
            temperature=0.95,
            top_p=0.98,
            presence_penalty=0.35,
            frequency_penalty=0.25,
            speed_mode=speed_mode,
        )
        for k in sorted(top.keys()):
            txt = top[k]["description"].strip()
            if _keep(txt):
                seen.add(txt)
                prompts.append(txt)
            if len(prompts) >= target_n:
                break

    return prompts[:target_n]


def write_master_json(
    api_key: str,
    out_path: Optional[str] = "master_file_gen.json",
    model_name: str = "gpt-4o",
    n_per_mode: int = 100,
    resume: bool = True,
) -> dict:
    """
    Generates n_per_mode prompts for each behavior mode (0..26)
    and writes a single nested JSON file with structure:
      { "0": [ {"command_id":0,"text":"..."}, ... ], ..., "26": [ ... ] }

    Writes to disk after each mode so that if the script fails, all modes
    generated so far are already saved. If resume=True and the output file
    exists, skips modes that are already present in the file.
    """
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    out_file = dataset_dir / Path(out_path or "master_file_gen.json").name

    # Load existing data so we can resume and so we write incrementally
    master = {}
    if resume and out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                master = json.load(f)
            print(f" Resumed: loaded {len(master)} mode(s) from {out_file.name}")
        except (json.JSONDecodeError, IOError) as e:
            print(f" Could not load existing file ({e}); starting fresh.")

    for mode_idx in range(N_BEHAVIOR_MODES):
        key = str(mode_idx)
        if resume and key in master and len(master[key]) >= n_per_mode:
            print(f" Mode {mode_idx}: Skipped (already have {len(master[key])} prompts).")
            continue

        prompts = generate_100_prompts_for_mode(
            mode_idx=mode_idx,
            api_key=api_key,
            model_name=model_name,
            target_n=n_per_mode,
        )

        mode_entries = [{"command_id": i, "text": txt} for i, txt in enumerate(prompts)]
        master[key] = mode_entries
        print(f" Mode {mode_idx}: Generated {len(prompts)} unique prompts.")

        # Write immediately after each mode so failures don't lose data
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2, ensure_ascii=False)

    return master

# ======================================================
#  CALL
# ======================================================
if __name__ == "__main__":
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "<YOUR_API_KEY>")
    # GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "<YOUR_API_KEY>")

    n_per_mode = 100
    # model_name = "gemini-2.0-flash"
    model_name = "gpt-4o"  # gpt model

    out_name = "master_file_gen_me33.json"
    print(f"Generating {n_per_mode} prompts for each of {N_BEHAVIOR_MODES} behavior modes...")
    master_data = write_master_json(
        api_key=OPENAI_API_KEY,
        out_path=out_name,
        model_name=model_name,
        n_per_mode=n_per_mode,
    )

    total_prompts = sum(len(v) for v in master_data.values())
    out_path = Path(__file__).resolve().parent.parent / "dataset" / out_name
    print(f"Wrote {total_prompts} total prompts to {out_path}")