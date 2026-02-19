import json
from pathlib import Path

# ---- paths ----
IN_PATH = Path("master_file_original.json")
OUT_PATH = Path("master_file_test.json")

new_commands = {
    "0": [  # left_fast
        "Command a swift port-side arc that cuts close to the KOZ boundary while driving rapidly toward the goal.",
        "Direct a high-energy leftward sweep that bends around the KOZ with minimal lateral expansion.",
        "Instruct a rapid left-side surge that threads through the tight spacing near the KOZ flank.",
        "Guide the vehicle along a sharp left-side acceleration contour that races around the obstacle.",
        "Initiate a fast left-skimming maneuver that preserves tight KOZ standoff while minimizing path length.",
        "Deliver a high-rate left-hand glide that presses aggressively toward the target through the open corridor.",
        "Execute a port-side velocity-heavy bypass that curves tightly around the KOZ envelope.",
        "Command a brisk left-margin traversal that aims for shortest-time routing around the obstacle.",
        "Direct a forceful left-side turn that transitions quickly from approach to corridor penetration.",
        "Instruct a rapid left-hand slide that clings closely to the KOZ curvature while advancing.",
        "Guide a fast-moving left-channel run that capitalizes on minimal detour geometry.",
        "Initiate a left-flank burst maneuver that leverages high thrust to wrap around the KOZ edge.",
        "Execute a dynamic left-lane push that funnels the vehicle through a narrow clearance band.",
        "Command a brisk port-side pivot that reshapes the trajectory tightly around the KOZ.",
        "Direct an accelerated left-skirt path that balances high velocity with precise obstacle tracking.",
        "Instruct a left-bound rapid sweep that leans aggressively into the available cross-range.",
        "Guide a swift left corridor dive that holds close proximity to the KOZ contour.",
        "Initiate a high-energy port-side threading motion that limits unnecessary arc length.",
        "Execute a leftward drive-through that maintains steep closing velocity toward the target.",
        "Command a curved left-turn sprint that winds closely around the KOZ boundary.",
        "Direct a high-speed left-edge maneuver that compresses lateral spacing while preserving clearance.",
        "Instruct a vigorous left-hand bypass that capitalizes on the narrow safe band near the KOZ.",
        "Guide a quick port-side glide that preserves tight standoff as the vehicle closes range.",
        "Initiate a rapid left-deflection route that churns forward with minimal directional hesitation.",
        "Execute a tightly coiled left-side sprint that reshapes the path around the KOZ with urgency.",
        "Command a lean left-moving pass that pulls the vehicle strongly through the open space.",
        "Direct a velocity-rich left-channel pass that prioritizes forward progression.",
        "Instruct a bold left-hand surge that compresses the curvature envelope near the KOZ.",
        "Guide a sharp-angled leftward transition that hastens the approach trajectory.",
        "Initiate a rapid contour-following left pass that races along the KOZ's exterior boundary."
    ],

    "1": [  # left_slow
        "Command a gentle port-side drift that keeps wide margins along the KOZ perimeter.",
        "Direct a gradual leftward wrap that prioritizes stability over rapid transit.",
        "Instruct a slow left-side meander that rounds the KOZ in a broad, cautious arc.",
        "Guide a wide left-deflected route that aims for smooth curvature and relaxed pacing.",
        "Initiate a controlled port-side contour path that slowly reforms the approach alignment.",
        "Execute a soft leftward detour that builds a long, even buffer around the KOZ boundary.",
        "Command a calm left-biased sweep that minimizes abrupt steering near the obstacle.",
        "Direct a broad left-hand bend that maintains steady progression at low velocity.",
        "Instruct a slow-paced left flank reach that distributes standoff generously.",
        "Guide a mellow left-side approach that expands the safety envelope around the KOZ.",
        "Initiate an easy-going left arc that prioritizes predictable spacing and gentle turns.",
        "Execute a slow drifting port turn that keeps the KOZ at an ample offset.",
        "Command a wide-angle left move that evolves gradually toward the goal direction.",
        "Direct a moderated left contouring motion that upholds stable separation from the KOZ.",
        "Instruct a low-energy left-lane bypass that avoids abrupt curvature changes.",
        "Guide a tempered left path that uses the outer corridor region for safe passage.",
        "Initiate a slow, sweeping left trajectory that tracks comfortably outside the KOZ.",
        "Execute an unhurried portward detour with consistently large margins.",
        "Command a measured left-circuiting path that favors gentle steering over directness.",
        "Direct a quiet left flank move that circles the KOZ with smooth, steady displacement.",
        "Instruct a slow port-side expansion maneuver that preemptively creates margin before entry.",
        "Guide a broad, low-speed left deviation that transitions softly back toward the target.",
        "Initiate a lightly curved left-hand bypass that emphasizes safety and predictability.",
        "Execute a gradual leftward slide that avoids angular spikes during KOZ circumnavigation.",
        "Command a slow-spaced port-side flow that maintains large lateral spacing at all times.",
        "Direct a relaxed left-channel passage that minimizes dynamic changes.",
        "Instruct a cautious left-guided arc that aligns carefully before converging toward the goal.",
        "Guide a slow, deliberate leftward push that respects conservative standoff guidelines.",
        "Initiate a smooth left-drift envelope that comfortably envelopes the KOZ periphery.",
        "Execute a wide slow-turning port track that ensures repeatable, stable motion."
    ],

    "2": [  # right_fast
        "Command a swift starboard-side arc that hugs the KOZ curvature while accelerating forward.",
        "Direct a high-speed rightward pivot that navigates around the KOZ with minimal path inflation.",
        "Instruct a rapid right-lane surge that moves decisively toward the target corridor.",
        "Guide a fast starboard bypass that compresses lateral margins into a narrow clearance band.",
        "Initiate a quick right-skimming maneuver that clings closely to the KOZ outline.",
        "Execute a dynamic starboard carve that reshapes the trajectory at elevated velocity.",
        "Command a sharp right-leaning sprint that drives straight toward the goal after bypass.",
        "Direct a forceful right-hand sweep that sharply realigns heading beyond the KOZ.",
        "Instruct a swift right-angle detour that leverages the shortest navigable bypass.",
        "Guide a high-rate right flank move that threads the safe region beside the KOZ.",
        "Initiate a speed-intensive starboard corridor dive that minimizes cross-range motion.",
        "Execute a brisk right-hand redirect that leverages available clearance tightly.",
        "Command a momentum-heavy right pivot that closes distance rapidly after rounding the KOZ.",
        "Direct a rapid right contour-following path that stays just outside the KOZ envelope.",
        "Instruct an aggressive right-side dash that arcs into the free space decisively.",
        "Guide a quick starboard glide that trails the KOZ boundary with measured precision.",
        "Initiate a steep rightward adjustment that turns rapidly onto the direct line of sight.",
        "Execute a fast starboard-edge bypass that drives forward with minimal curvature.",
        "Command a right-focused surge that accelerates through the tight safe channel.",
        "Direct a rapid right-led progression that uses high-rate updates to clear the KOZ swiftly.",
        "Instruct a starboard-driven velocity push that trims unnecessary lateral expenditure.",
        "Guide an assertive right-lane motion that holds a tight contour along the KOZ.",
        "Initiate a swift right-bound wrap that quickly reorients toward the desired terminal state.",
        "Execute a vigorous right-channel sprint that capitalizes on the open region beyond the KOZ.",
        "Command a quick starboard deviation that emphasizes minimal trajectory dilation.",
        "Direct a fast-moving right-hand hook that folds tightly around the KOZ curvature.",
        "Instruct a bold rightward vector change that keeps momentum high throughout the maneuver.",
        "Guide a rapid right-skimming pass that positions the vehicle efficiently toward the goal.",
        "Initiate a right-leaning accelerated bypass that compresses both time and arc length.",
        "Execute a velocity-rich starboard trajectory that threads through a narrow access lane."
    ],

    "3": [  # right_slow
        "Command a gentle starboard drift that expands clearance along the KOZ exterior.",
        "Direct a broad right-hand curve that emphasizes soft transitions and stability.",
        "Instruct a slow right-lane drift that maintains ample standoff while circling the KOZ.",
        "Guide a loosely curved right-biased path that avoids tight turns during obstacle bypass.",
        "Initiate a cautious starboard sweep that reinforces reliable KOZ spacing.",
        "Execute a low-tempo rightward arc that ensures smooth heading evolution.",
        "Command a relaxed starboard bypass that prioritizes predictability.",
        "Direct a slow right-margin glide that stays well outside the KOZ boundary.",
        "Instruct a gradual rightward divergence that transitions into a long, safe arc.",
        "Guide a wide right-skirt path that maintains abundant space around the KOZ.",
        "Initiate a gentle starboard-facing bend that carries the vehicle cleanly past the obstacle.",
        "Execute a slow-rolling right-hand detour that stabilizes attitude throughout.",
        "Command a measured right contour that avoids angular acceleration near the KOZ.",
        "Direct a low-speed right-leaning route that unwraps the approach gradually.",
        "Instruct an unhurried right-side drift that keeps the KOZ consistently distant.",
        "Guide a smooth starboard circumnavigation that progresses incrementally.",
        "Initiate a relaxed right flank circling motion that maintains wide safety buffers.",
        "Execute a mild rightward offset trajectory that flows around the KOZ without urgency.",
        "Command a leisurely starboard separation that emphasizes clear margins.",
        "Direct a slow right-threaded path that glides between boundary constraints.",
        "Instruct a calm right-hand curvature that reorients heading without sharp deviations.",
        "Guide a wide-angle right channel entry that prepares for gradual convergence.",
        "Initiate a slow pivot to the right that redistributes spacing safely.",
        "Execute a broad starboard bubble-maneuver that preserves constant clearance.",
        "Command a subdued right-led envelope that reshapes approach geometry safely.",
        "Direct a low-reactivity starboard bypass with minimal directional stress.",
        "Instruct a long, slow right arc that preserves a stable corridor profile.",
        "Guide a smooth-flowing right-drift pattern that envelops the KOZ comfortably.",
        "Initiate a gentle right-hand loop that allows the system to maintain conservative standoff.",
        "Execute a soft starboard rerouting that moves around the KOZ with assured spacing."
    ],

    "4": [  # direct_fast
        "Command a swift axial surge straight through the central gap with minimal lateral drift.",
        "Direct a high-speed corridor insertion that channels momentum along the midline.",
        "Instruct a rapid line-of-sight run that exploits the clear central passageway.",
        "Guide a fast forward projection that maintains tight control within the corridor walls.",
        "Initiate a velocity-heavy straight-through push that uses symmetry for stable routing.",
        "Execute a central-lane burst that balances speed with minimal path distortion.",
        "Command a streamlined dash toward the goal while holding central corridor alignment.",
        "Direct a forward sprint that funnels through the available axial gap with precision.",
        "Instruct a high-energy midline trajectory that avoids unnecessary cross-range actions.",
        "Guide a rapid straight-arrow maneuver that channels thrust along the corridor spine.",
        "Initiate a brisk corridor pass that minimizes lateral effort.",
        "Execute a sharp axial acceleration motion that threads through the middle zone.",
        "Command a fast-moving central track that adheres to a narrow lateral envelope.",
        "Direct a forward-locked corridor motion that maintains high translational rate.",
        "Instruct a quick path-through that leverages the central void between obstacles.",
        "Guide a near-ballistic forward vector that remains strictly corridor-aligned.",
        "Initiate a steep forward advance that upholds a direct approach to the goal.",
        "Execute a high-speed axial routing that avoids detours on either side.",
        "Command a tightly focused central-axis sprint that clamps lateral drift.",
        "Direct a hard-charging straight-in maneuver that leverages unobstructed space.",
        "Instruct a fast corridor-constrained press that limits curvature sharply.",
        "Guide a forward glide at elevated speed that occupies the corridor midline.",
        "Initiate a rapid mid-gap insertion that propels directly along the target line.",
        "Execute a long, straight, high-velocity run between the KOZ boundaries.",
        "Command a central-axis racing profile that strongly favors minimum transit time.",
        "Direct an assertive forward advance that threads the symmetric gap efficiently.",
        "Instruct a swift corridor charge that compresses heading deviations.",
        "Guide an unwavering central-lane acceleration that remains almost perfectly straight.",
        "Initiate a powerful forward sweep through the central window toward the goal.",
        "Execute a velocity-driven axial glide that remains centered from entry to exit."
    ],

    "5": [  # direct_slow
        "Command a calm central-axis drift that proceeds evenly between the KOZ boundaries.",
        "Direct a low-speed corridor entry that preserves wide symmetric margins.",
        "Instruct a gentle forward passage that avoids hugging either side of the corridor.",
        "Guide a slow and balanced progression along the midline channel.",
        "Initiate a steady central walk-through that prioritizes safety over rapid transit.",
        "Execute a relaxed axial slide that maintains evenly distributed clearance.",
        "Command a soft forward approach that avoids abrupt lateral changes.",
        "Direct a slow corridor carriage motion that advances with stable spacing.",
        "Instruct a mild central-line roll that maintains a wide corridor footprint.",
        "Guide a gentle progression straight through the central aperture.",
        "Initiate a conservative forward glide that preserves ample headroom on both sides.",
        "Execute a light-paced corridor crossing that emphasizes predictability.",
        "Command a steady central channel interface that slowly carries toward the goal.",
        "Direct a controlled axial movement that keeps equal buffers left and right.",
        "Instruct a slow, unwavering midline transit that minimizes steering impulses.",
        "Guide a calm straight-ahead shift that flows through the symmetric gap.",
        "Initiate a deliberate central corridor advance that shapes a coherent, slow path.",
        "Execute a low-rate axial migration that stabilizes orientation during passage.",
        "Command a smooth forward entry that retains large margins to both KOZ edges.",
        "Direct a slow approach along the geometric center of the corridor.",
        "Instruct a mild, center-anchored glide that avoids any drift toward the flanks.",
        "Guide a steady forward slip that maintains symmetric standoff while moving.",
        "Initiate an easy axial channel flow that evolves gradually.",
        "Execute a gentle straight-down-lane path that sustains broad clearance.",
        "Command a corridor-centered slow motional arc that keeps heading stable.",
        "Direct a low-speed channel walk that maintains high confidence in spacing.",
        "Instruct a measured forward line that skims through the central opening with poise.",
        "Guide a slow, symmetric envelope that respects corridor boundaries cleanly.",
        "Initiate a calm axial lane traverse that advances at relaxed tempo.",
        "Execute a careful mid-gap forward slide that maintains uniform lateral room."
    ]
}

# ---- load existing file ----
with IN_PATH.open("r") as f:
    data = json.load(f)

# sanity check and append
for key, texts in new_commands.items():
    if key not in data:
        raise KeyError(f"Behavior key {key} not found in input JSON")

    cmds = data[key]
    # make sure there are 100 commands and next id is 100
    max_id = max(c["command_id"] for c in cmds)
    if len(cmds) != 100 or max_id != 99:
        raise ValueError(f"Behavior {key} does not have 100 commands with ids 0-99 (len={len(cmds)}, max_id={max_id})")

    next_id = max_id + 1
    for i, text in enumerate(texts):
        cmds.append({
            "command_id": next_id + i,
            "text": text
        })

# ---- write new file ----
with OUT_PATH.open("w") as f:
    json.dump(data, f, indent=2)

print(f"Wrote updated file with 130 commands/behavior to: {OUT_PATH}")
