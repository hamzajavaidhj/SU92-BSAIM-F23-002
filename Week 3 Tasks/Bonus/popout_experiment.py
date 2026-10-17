"""
Week 1, carried forward into code.  (Bonus - 5 marks.)   popout_experiment.py

    python3 popout_experiment.py            draw the four panels; chart results if they exist
    python3 popout_experiment.py --run      run the timing experiment on one participant
    python3 popout_experiment.py --chart    (re)draw the result chart from popout_results.csv
    python3 popout_experiment.py --demo     SIMULATED data, only to test the chart pipeline

Workflow
--------
1. Run the script once -> popout_panels.png (the four panels).
2. For each of FIVE classmates run:  python3 popout_experiment.py --run
   (enter participant number 1..5). Each person sees the four panels one at a
   time, in an order drawn at random for THAT person. You press Enter when
   they say "done", then type the number they counted. Every trial is written
   to popout_results.csv immediately, so a crash loses nothing.
3. Run the script again -> popout_chart.png + popout_findings.txt.

--demo never touches popout_results.csv. Its output is stamped SIMULATED so it
can never be mistaken for the real experiment.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    sys.path.insert(0, p)
from vizlib import PALETTE, save_fig, style_defaults   # noqa: E402

ROOT = HERE
style_defaults()

ROWS, COLS = 12, 18
TARGET = "3"
N_TARGETS = 9
SEED = 1947

# panel id -> (short label for the chart axis, spec of channels used)
PANELS = {
    1: ("1  No cue",         dict()),
    2: ("2  Colour",         dict(colour=True)),
    3: ("3  Size",           dict(size=True)),
    4: ("4  Colour + shape", dict(colour=True, shape=True)),
}
N_PARTICIPANTS = 5

RESULTS_CSV = os.path.join(ROOT, "popout_results.csv")
DEMO_CSV = os.path.join(ROOT, "popout_results_DEMO.csv")
FIELDS = ["participant", "trial", "panel", "seconds", "count", "correct"]


# --------------------------------------------------------------- stimulus
def make_grid(seed=SEED):
    """A grid of digits with exactly N_TARGETS threes, in fixed positions."""
    rng = np.random.default_rng(seed)
    others = [d for d in "0123456789" if d != TARGET]
    grid = np.array([[rng.choice(others) for _ in range(COLS)]
                     for _ in range(ROWS)], dtype=object)
    flat = rng.choice(ROWS * COLS, size=N_TARGETS, replace=False)
    for k in flat:
        grid[k // COLS, k % COLS] = TARGET
    return grid


def draw(ax, grid, title, colour=False, size=False, shape=False):
    """
    Draw one panel.

    colour -> targets get the highlight hue   (a preattentive channel)
    size   -> targets are drawn larger        (another one)
    shape  -> targets get a ring behind them  (survives greyscale printing
              and colour blindness)
    """
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.5, ROWS - 0.5)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, loc="left")
    for r in range(ROWS):
        for c in range(COLS):
            d = grid[r, c]
            hit = d == TARGET
            if shape and hit:
                ax.add_patch(plt.Circle((c, r), 0.40, fill=False,
                                        edgecolor=PALETTE["ink"], lw=1.6))
            ax.text(c, r, d, ha="center", va="center",
                    fontsize=15 if (size and hit) else 11,
                    fontweight="bold" if (size and hit) else "normal",
                    color=PALETTE["highlight"] if (colour and hit)
                    else PALETTE["ink"])


def draw_panels():
    grid = make_grid()
    fig, ax = plt.subplots(2, 2, figsize=(13.0, 8.0))

    # 1. no pop-out: serial search
    draw(ax[0, 0], grid, "1. No channel - you have to read every cell")
    # 2. colour: one preattentive channel
    draw(ax[0, 1], grid, "2. Colour - the targets appear before you look",
         colour=True)
    # 3. TODO 1 - SIZE ONLY, no colour (targets larger + bold, same ink)
    draw(ax[1, 0], grid, "3. Size only - bigger, but the same ink colour",
         size=True)
    # 4. TODO 2 - redundant coding: colour AND shape (ring)
    draw(ax[1, 1], grid, "4. Colour AND shape - two channels saying the same thing",
         colour=True, shape=True)

    fig.suptitle(f"Count the {TARGET}s. Same {N_TARGETS} targets in all four "
                 f"panels, same positions.",
                 x=0.006, ha="left", fontsize=13.5, fontweight="bold",
                 color=PALETTE["ink"])
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = save_fig(fig, os.path.join(ROOT, "popout_panels.png"),
                   f"{ROWS}x{COLS} grid, {N_TARGETS} targets, seed {SEED}. "
                   f"Week 1 activity, rebuilt in Matplotlib.")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------- experiment
def participant_order(pid):
    """
    A random permutation of the four panels, different for every participant
    but reproducible: numpy default_rng seeded with 1000 + participant number.
    """
    return list(np.random.default_rng(1000 + pid).permutation([1, 2, 3, 4]) + 0)


def run_participant():
    pid = int(input("Participant number (1-5): ").strip())
    order = participant_order(pid)
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            done = {int(r["participant"]) for r in csv.DictReader(f)}
    if pid in done:
        sys.exit(f"Participant {pid} already has results in {RESULTS_CSV}. "
                 f"Delete those rows first if you want to redo them.")

    new_file = not os.path.exists(RESULTS_CSV)
    grid = make_grid()
    print(f"\nOrder for participant {pid}: {order}  (do not tell them)\n"
          f"Tell them: 'Count the {TARGET}s as fast as you can, then say the "
          f"number out loud.'\n")
    plt.ion()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for trial, panel in enumerate(order, start=1):
            input(f"[trial {trial}/4] Press Enter to SHOW the panel... ")
            fig, ax = plt.subplots(figsize=(9.5, 6.4))
            draw(ax, grid, f"Count the {TARGET}s", **PANELS[panel][1])
            fig.tight_layout()
            plt.show(block=False)
            plt.pause(0.05)
            t0 = time.perf_counter()
            input("   Press Enter the instant they say 'done'... ")
            seconds = time.perf_counter() - t0
            plt.close(fig)
            count = int(input("   Number they counted: ").strip())
            w.writerow(dict(participant=pid, trial=trial, panel=panel,
                            seconds=round(seconds, 2), count=count,
                            correct=int(count == N_TARGETS)))
            f.flush()
            print(f"   saved: panel {panel}, {seconds:.2f}s, count {count}\n")
    print("Participant done. Thank them.")


def simulate_demo():
    """
    SIMULATED data. Only for testing that the chart code works before real
    data exists. Assumed means follow the textbook ordering - which is exactly
    why this must never be presented as a result.
    """
    rng = np.random.default_rng(7)
    mu = {1: 14.0, 2: 3.0, 3: 5.5, 4: 3.1}
    rows = []
    for pid in range(1, N_PARTICIPANTS + 1):
        for trial, panel in enumerate(participant_order(pid), start=1):
            s = float(rng.lognormal(np.log(mu[panel]), 0.25))
            ok = int(rng.random() > (0.30 if panel == 1 else 0.05))
            rows.append(dict(participant=pid, trial=trial, panel=panel,
                             seconds=round(s, 2),
                             count=N_TARGETS if ok else N_TARGETS - 1,
                             correct=ok))
    with open(DEMO_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return DEMO_CSV


# --------------------------------------------------------------- analysis
def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    data = {p: [] for p in PANELS}
    ok = {p: [] for p in PANELS}
    for r in rows:
        p = int(r["panel"])
        data[p].append(float(r["seconds"]))
        ok[p].append(int(r["correct"]))
    n = len({r["participant"] for r in rows})
    return data, ok, n


def findings(means, n):
    """Draft of the three sentences. Edit them in your own words."""
    fast = min(means, key=means.get)
    slow = max(means, key=means.get)
    name = {p: PANELS[p][0].split("  ")[1].lower() for p in PANELS}
    s1 = (f"The fastest channel was {name[fast]} "
          f"(mean {means[fast]:.1f} s, n = {n}).")
    s2 = (f"The slowest was {name[slow]} "
          f"(mean {means[slow]:.1f} s), "
          f"{means[slow] / means[fast]:.1f}x the fastest.")
    diff = means[4] - means[2]
    rel = diff / means[2]
    if abs(rel) <= 0.10:
        s3 = (f"Redundant coding (colour + shape) cost essentially nothing: "
              f"{means[4]:.1f} s vs {means[2]:.1f} s for colour alone "
              f"({diff:+.1f} s).")
    elif diff > 0:
        s3 = (f"Redundant coding did cost time here: {means[4]:.1f} s vs "
              f"{means[2]:.1f} s for colour alone ({diff:+.1f} s, {rel:+.0%}) "
              f"- with n = {n} this may be noise or the ring adding clutter.")
    else:
        s3 = (f"Redundant coding was actually faster than colour alone "
              f"({means[4]:.1f} s vs {means[2]:.1f} s, {diff:+.1f} s) - "
              f"free, as predicted, though n = {n} cannot rule out noise.")
    return [s1, s2, s3]


def finding_title(means, demo):
    fast = min(means, key=means.get)
    slow = max(means, key=means.get)
    nm = lambda p: PANELS[p][0].split("  ")[1].lower()   # noqa: E731
    diff = means[4] - means[2]
    if abs(diff / means[2]) <= 0.10:
        tail = "adding shape to colour cost no measurable time"
    elif diff > 0:
        tail = f"adding shape to colour cost {diff:.1f} s"
    else:
        tail = f"adding shape to colour saved {-diff:.1f} s"
    t = (f"{nm(fast).capitalize()} was fastest ({means[fast]:.1f} s) and "
         f"{nm(slow)} slowest ({means[slow]:.1f} s); {tail}")
    return ("SIMULATED DEMO - " if demo else "") + t


def draw_chart(path, demo=False):
    data, ok, n = load(path)
    means = {p: float(np.mean(v)) for p, v in data.items()}
    labels = [PANELS[p][0] for p in PANELS]
    xs = np.arange(len(PANELS))
    rng = np.random.default_rng(3)          # fixed jitter -> reproducible chart
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(xs, [means[p] for p in PANELS], width=0.58,
           color=PALETTE["muted"], alpha=0.35, zorder=2)   # faded summary
    for i, p in enumerate(PANELS):                          # raw data on top
        v = np.array(data[p])
        good = np.array(ok[p], dtype=bool)
        j = rng.uniform(-0.14, 0.14, len(v))
        ax.scatter(i + j[good], v[good], s=46, marker=markers[p],
                   color=PALETTE["ink"], zorder=4)
        ax.scatter(i + j[~good], v[~good], s=46, marker=markers[p],
                   facecolor="none", edgecolor=PALETTE["highlight"],
                   linewidths=1.6, zorder=4)
        ax.text(i, max(v) + 0.03 * max(max(w) for w in data.values()),
                f"mean {means[p]:.1f} s", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=PALETTE["ink"])
    top = max(max(v) for v in data.values())
    ax.set_ylim(0, top * 1.18)                              # zero baseline
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Seconds to count the 3s")
    ax.grid(axis="x", visible=False)
    ax.set_title("\n".join(textwrap.wrap(finding_title(means, demo), 78)))

    n_wrong = sum(len(v) - sum(v) for v in ok.values())
    note = (("SIMULATED DATA - NOT A REAL EXPERIMENT. " if demo else "") +
            f"n = {n} participants, each timed on all 4 panels. Order randomised "
            f"per participant: numpy default_rng(1000 + participant no.) "
            f"permutation of the 4 panels. Same grid & positions in every panel.\n"
            f"Time = stopwatch from panel shown to spoken answer. Hollow "
            f"orange markers = wrong count ({n_wrong} of {sum(len(v) for v in ok.values())} "
            f"trials). Bars = mean; dots = individual times.")
    note = "\n".join(textwrap.fill(line, 125) for line in note.split("\n"))
    out_png = os.path.join(ROOT, "popout_chart_DEMO.png" if demo else "popout_chart.png")
    out = save_fig(fig, out_png, note)
    plt.close(fig)
    print("wrote", out)

    txt = "\n".join(findings(means, n))
    if not demo:
        with open(os.path.join(ROOT, "popout_findings.txt"), "w") as f:
            f.write(txt + "\n")
    print("\nDraft three sentences (rewrite in your own words, and explain any "
          "result that disagrees with the textbook):\n" + txt)
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="time one participant")
    ap.add_argument("--chart", action="store_true", help="chart popout_results.csv")
    ap.add_argument("--demo", action="store_true", help="SIMULATED data test")
    a = ap.parse_args()

    if a.run:
        run_participant()
        return
    if a.demo:
        draw_chart(simulate_demo(), demo=True)
        return

    draw_panels()
    if os.path.exists(RESULTS_CSV):
        draw_chart(RESULTS_CSV)
    else:
        print(f"\nNo {os.path.basename(RESULTS_CSV)} yet. Run "
              f"`python3 popout_experiment.py --run` once per classmate "
              f"({N_PARTICIPANTS} of them), then run this again.")


if __name__ == "__main__":
    main()
