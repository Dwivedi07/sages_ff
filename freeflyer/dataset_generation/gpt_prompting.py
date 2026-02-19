
import os
from typing import Dict, List
import random
import time
import json
from pathlib import Path
from openai import OpenAI
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted  # rate limit exception
# from dotenv import load_dotenv
# load_dotenv()

################################## HELPER VARS #################################
CASE_NAMES = [
    "left_fast",   # 0
    "left_slow",   # 1
    "right_fast",  # 2
    "right_slow",  # 3
    "direct_fast", # 4
    "direct_slow"  # 5
]

MODE_IS_FAST = {0: True, 1: False, 2: True, 3: False, 4: True, 5: False}

# New: per-mode metadata used by the annotator
BEHAVIOR_META = {
    0: dict(   # left_fast
        label="left_fast",
        canonical=(
            "fast left-side bypass of the obstacle with tight but safe clearance to the KOZ"
        ),
        geom_hints=[
            "sharp left-side arc", "rapid port-side bypass",
            "compressed schedule around left body", "left-biased sprint"
        ],
    ),
    1: dict(   # left_slow
        label="left_slow",
        canonical=(
            "slow left-side approach that widens clearance and allows extended loiter near the goal"
        ),
        geom_hints=[
            "broad left-side arc", "port-side drift pass",
            "expanded left margin", "left-biased loiter"
        ],
    ),
    2: dict(   # right_fast
        label="right_fast",
        canonical=(
            "fast right-side bypass of the obstacle with tight but safe clearance to the KOZ"
        ),
        geom_hints=[
            "sharp right-side arc", "rapid starboard bypass",
            "compressed schedule around right body", "right-biased sprint"
        ],
    ),
    3: dict(   # right_slow
        label="right_slow",
        canonical=(
            "slow right-side approach that widens clearance and allows extended loiter near the goal"
        ),
        geom_hints=[
            "broad right-side arc", "starboard drift pass",
            "expanded right margin", "right-biased loiter"
        ],
    ),
    4: dict(   # direct_fast
        label="direct_fast",
        canonical=(
            "fast transit through the central corridor with aggressive but KOZ-compliant timing"
        ),
        geom_hints=[
            "central corridor sprint", "tight straight-through channel",
            "compressed median-lane run", "direct, time-critical traverse"
        ],
    ),
    5: dict(   # direct_slow
        label="direct_slow",
        canonical=(
            "slow, conservative central-corridor transit with generous time and lateral margin"
        ),
        geom_hints=[
            "broad central corridor", "gentle straight-through channel",
            "relaxed median-lane run", "direct but margin-heavy traverse"
        ],
    ),
}

############################## ANNOTATE FUNCTIONS ##############################

def annotate_traj_behaviors_gpt(
    ids: List[int],
    api_key: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 30,
    temperature: float = 0.7,
    top_p: float = 0.9,
    presence_penalty: float = 0.3,
    frequency_penalty: float = 0.2,
    speed_mode: str | None = None,   # still used, but now consistent with mode
) -> Dict[int, Dict[str, str]]:

    # NOTE: ids are now in {0,1,2,3,4,5} matching CASE_NAMES / BEHAVIOR_META
    voices = ["active voice", "passive voice"]

    base_tones = [
        "proximity-operations tone",
        "navigation-performance tone",
        "safety-justification tone",
        "design-insight tone",
        "line-of-sight preservation tone",
    ]
    fast_tones = base_tones + [
        "time-optimality tone",
        "agile-maneuvering tone",
        "high pulse-cadence tone",
    ]
    slow_tones = base_tones + [
        "conservative-velocity tone",
        "risk-averse tone",
        "low-thrust economy tone",
        "extended coast-phase tone",
    ]

    structures = [
        "state goal before path",
        "start with a verb phrase",
        "start with a noun phrase",
        "start with a subordinate clause",
        "use a nominalization once",
        "use a gerund once",
        "use a concise cause-effect clause",
        "state clearance before geometry",
    ]


    forbidden = [
        "exhibits", "employs", "circumferential", "designed to", "effectively",
        "maintaining", "linear path", "negligible lateral", "restricted lateral",
        "navigational integrity", "navigate", "path to the right", "path to the left",
    ]

    # Speed-specific vocabulary hints
    speed_hints = {
        "fast": [
            "at elevated speed", "rapid transit",
            "time-prioritized routing", "high-velocity pass", "agile profile",  "agile maneuver"
        ],
        "slow": [
            "at reduced speed", "cautious velocity",
            "margin-prioritized routing", "low-velocity pass", "conservative profile", "sluggish maneuver"
        ],
        None: [],
    }

    client = OpenAI(api_key=api_key)
    out: Dict[int, Dict[str, str]] = {}

    system_msg = (
        "You are an expert GNC technical writer for proximity operations on a microgravity bench. "
        "Produce ONE sentence per input describing a goal-directed trajectory with KOZ compliance and left/right/central corridor behavior. "
        "Be concise (≤15 words), precise, and varied in style. Avoid jargon bloat."
    )

    chosen_tones = (
        fast_tones if speed_mode == "fast"
        else slow_tones if speed_mode == "slow"
        else base_tones
    )

    speed_clause = {
        "fast": "Bias toward quicker transits and higher RCS pulse cadence.",
        "slow": "Bias toward longer coasts, low Δv expenditure, and wider safety margins.",
        None:   "No specific speed bias; emphasize KOZ compliance and stable standoff.",
    }[speed_mode]

    for i, mode_id in enumerate(ids):
        meta = BEHAVIOR_META[mode_id]
        beh_text = meta["canonical"]
        geom_hints = "; ".join(meta["geom_hints"])
        spd_hints = "; ".join(speed_hints.get(speed_mode, []))

        style_voice = random.choice(voices)
        style_tone = random.choice(chosen_tones)
        style_structure = random.choice(structures)
        do_not_use = ", ".join(forbidden)

        user_msg = (
            f"Behavior description: {beh_text}\n"
            "Task: Produce ONE sentence that characterizes this specific trajectory mode on a microgravity bench "
            "(path geometry, avoidance strategy, corridor usage, and implied tempo).\n"
            f"Speed context: {speed_clause}\n"
            f"Style controls: Use {style_voice}; {style_tone}; {style_structure}. "
            "Prefer precise proximity-ops phrasing (KOZ, standoff, LOS, Δv, RCS). "
            "Avoid reusing long phrases from the behavior description.\n"
            f"Strict constraints: ≤15 words; neutral, technically precise; no bullet points; no quotes; "
            f"avoid these terms: {do_not_use}.\n"
            f"Vocabulary hints (optional, but keep them mode-specific): {geom_hints}; {spd_hints}\n"
        )

        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            desc = (rsp.choices[0].message.content or "").strip()
        except Exception as e:
            desc = f"ERROR: {e.__class__.__name__}"

        out[i] = {
            "id": mode_id,
            "behavior": meta["label"],
            "description": desc,
        }

    return out

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
    mode_idx in {0..5}, matching CASE_NAMES / BEHAVIOR_META.
    """
    speed_mode = "fast" if MODE_IS_FAST[mode_idx] else "slow"

    prompts: List[str] = []
    seen = set()
    sched = _diversity_schedules(target_n)

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
            if txt and txt not in seen:
                seen.add(txt)
                prompts.append(txt)

    # top-up loop as before
    while len(prompts) < target_n:
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
            if txt and txt not in seen:
                seen.add(txt)
                prompts.append(txt)
            if len(prompts) >= target_n:
                break

    return prompts[:target_n]


def write_master_json(
    api_key: str,
    out_path: str = None,
    model_name: str = None,
    n_per_mode: int = 100,
) -> dict:
    """
    Generates n_per_mode prompts for each behavior mode (0..5)
    and writes a single nested JSON file with structure:
      { "0": [ {"command_id":0,"text":"..."}, ... ], ..., "5": [ ... ] }
    """
    master = {}
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir.parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)  

    for mode_idx in range(6):
        prompts = generate_100_prompts_for_mode(
            mode_idx=mode_idx,
            api_key=api_key,
            model_name=model_name,
            target_n=n_per_mode,
        )

        # Build list of dicts for this mode
        mode_entries = [{"command_id": i, "text": txt} for i, txt in enumerate(prompts)]
        master[str(mode_idx)] = mode_entries
        print(f" Mode {mode_idx}: Generated {len(prompts)} unique prompts.")
    
    out_path = dataset_dir / Path(out_path).name
    # Write as a single JSON file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)

    return master

# ======================================================
#  CALL
# ======================================================
if __name__ == "__main__":
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "<YOUR_API_KEY>")
    # GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "<YOUR_API_KEY>")

    n_per_mode = 120
    # model_name = "gemini-2.0-flash"
    model_name = "gpt-4o" # gpt model

    print("Generating 100 prompts for each of 6 behavior modes...")
    master_data = write_master_json(
        api_key=OPENAI_API_KEY, # GEMINI_API_KEY
        out_path="master_file_gen_me.json",
        model_name=model_name,
        n_per_mode=n_per_mode,
    )

    total_prompts = sum(len(v) for v in master_data.values())
    print(f"Wrote {total_prompts} prompts to dataset/master_file.json")