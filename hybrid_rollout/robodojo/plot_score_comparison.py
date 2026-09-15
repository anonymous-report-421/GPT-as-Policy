"""Offline, reproducible figures: official simulation references vs. hybrid v3.

No rollout imports, model calls, or scheduler mutations. Remote JavaScript is
parsed as data, never evaluated. Existing output directories are not overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import statistics
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SITE = "https://robodojo-benchmark.com/"
TASKS = (
    "organize_table", "classify_objects_by_language", "imitate_sorting_sequence",
    "arrange_largest_number", "pack_objects_into_box", "classify_objects",
    "build_tower", "make_kong", "fold_clothes", "put_bottles_into_dustbin",
)
LABELS = {
    "organize_table": "Organize\ntable",
    "classify_objects_by_language": "Classify by\nlanguage",
    "imitate_sorting_sequence": "Imitate sorting\nsequence",
    "arrange_largest_number": "Arrange largest\nnumber*",
    "pack_objects_into_box": "Pack objects\ninto box*",
    "classify_objects": "Classify\nobjects",
    "build_tower": "Build\ntower",
    "make_kong": "Make kong",
    "fold_clothes": "Fold\nclothes*",
    "put_bottles_into_dustbin": "Put bottles\ninto dustbin",
}
MODEL_KEYS = {
    "DM0.5": "OpenDM05",
    "GalaxeaVLA (G0.5)": "G05",
    "Xiaomi-Robotics-1": "Xiaomi_Robotics_1",
    "Pi-05": "Pi_05",
}
BASELINE = "Pi-05"
HYBRID = "Pi-05 + GPT (v3)"
COLORS = ("#416CA6", "#57A49A", "#A891BD", "#AAB1BA", "#E67541")
CERTIFICATE_SHA = "40701bd29b81c7e2f391ec2c33743184ef387daf4b7d9824901896ad95e6083e"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    if urllib.parse.urlparse(url).netloc != "robodojo-benchmark.com":
        raise ValueError("Only the public RoboDojo host is permitted")
    with urllib.request.urlopen(url, timeout=45) as response:
        if urllib.parse.urlparse(response.url).netloc != "robodojo-benchmark.com":
            raise ValueError("Unexpected external redirect")
        return response.read()


def parse_official(script: str) -> dict:
    """Fail closed if the published data format changes; no JavaScript eval."""
    pattern = r'\{model:"([^"]+)",team:"([^"]*)",generalizationStd:o\((.*?)average:o\(([\d.]+),([\d.]+)\)'
    ranking = [{"model": m[1], "team": m[2], "overall_score": float(m[4]),
                "overall_success_rate": float(m[5])}
               for m in re.finditer(pattern, script)]
    if len(ranking) < 3 or len({r["model"] for r in ranking}) != len(ranking):
        raise ValueError("Could not uniquely parse simulation overall ranking")
    ranking.sort(key=lambda x: (-x["overall_score"], -x["overall_success_rate"], x["model"]))
    names = [r["model"] for r in ranking[:3]]
    if BASELINE in names:
        raise ValueError("Pi-05 is now in the top three: review the figure layout")
    decoder = json.JSONDecoder()
    models = {}
    for name in names + [BASELINE]:
        if name not in MODEL_KEYS:
            raise ValueError(f"New top-three model requires a reviewed data mapping: {name}")
        marker = json.dumps(MODEL_KEYS[name]) + ":"
        if script.count(marker) != 1:
            raise ValueError(f"Ambiguous model task map: {name}")
        data, _ = decoder.raw_decode(script.split(marker, 1)[1])
        if not isinstance(data, dict):
            raise ValueError("Invalid model task map")
        models[name] = data
    date = re.search(r'fc="([^"]+)"', script)
    return {"ranking": ranking, "models": models,
            "published_data_date": date[1] if date else None}


def parse_hybrid(payload: bytes) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    if len(rows) != 50 or len({r["case_id"] for r in rows}) != 50:
        raise ValueError("Expected exactly 50 unique complete hybrid cases")
    if Counter(r["task"] for r in rows) != Counter({task: 5 for task in TASKS}):
        raise ValueError("The ten-task panel does not match")
    for row in rows:
        if row["context_version"] != "v3" or row["native_success"] not in ("true", "false"):
            raise ValueError("Not an eligible v3 native terminal result")
        score = float(row["native_score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid native score")
        row["native_score"] = score
    expected_generalization = {"arrange_largest_number", "pack_objects_into_box", "fold_clothes"}
    for task in TASKS:
        counts = Counter(r["variant"] for r in rows if r["task"] == task)
        expected = {"standard": 2, "random": 3} if task in expected_generalization else {"standard": 5}
        if counts != Counter(expected):
            raise ValueError(f"Unexpected variant composition for {task}: {counts}")
    return rows


def aggregate(official: dict, cases: list[dict]) -> dict:
    names = [r["model"] for r in official["ranking"][:3]] + [BASELINE, HYBRID]
    task_rows, variant_rows = [], []
    for task in TASKS:
        local = [row for row in cases if row["task"] == task]
        counts = Counter(row["variant"] for row in local)
        values, baseline_sr = {}, 0.0
        for model in names[:-1]:
            weighted = 0.0
            for variant, count in sorted(counts.items()):
                key = task + ("_random" if variant == "random" else "")
                record = official["models"][model][key]  # Missing is NOT zero.
                score, sr = float(record["score"]), float(record["successRate"])
                if not all(math.isfinite(v) and 0 <= v <= 100 for v in (score, sr)):
                    raise ValueError(f"Invalid official score: {model}/{key}")
                weighted += count * score / len(local)
                if model == BASELINE:
                    baseline_sr += count * sr / len(local)
                variant_rows.append({"task": task, "variant": variant, "model": model,
                                     "score_100": score, "success_rate_percent": sr,
                                     "panel_weight": count / len(local), "local_n": None,
                                     "source": "official_published_aggregate"})
            values[model] = weighted
        values[HYBRID] = statistics.mean(row["native_score"] for row in local) * 100
        for variant, count in sorted(counts.items()):
            selected = [r for r in local if r["variant"] == variant]
            variant_rows.append({"task": task, "variant": variant, "model": HYBRID,
                                 "score_100": statistics.mean(r["native_score"] for r in selected) * 100,
                                 "success_rate_percent": sum(r["native_success"] == "true" for r in selected) / count * 100,
                                 "panel_weight": count / len(local), "local_n": count,
                                 "source": "hybrid_v3_50_complete_cases"})
        task_rows.append({"task": task, "scores": values, "pi05_panel_weighted_sr": baseline_sr,
                          "local_n": len(local), "variant_counts": dict(counts)})
    means = {name: statistics.mean(row["scores"][name] for row in task_rows) for name in names}
    return {"models": names, "tasks": task_rows, "ten_task_mean": means,
            "variants": variant_rows,
            "lowest_five": [r["task"] for r in sorted(task_rows, key=lambda r: r["pi05_panel_weighted_sr"])[:5]]}


def verify_selection(cases: list[dict], certificate: dict) -> None:
    selection = certificate["selection"]
    if not selection["complete"] or not selection["platform_verified"] or selection["errors"] or selection["missing"]:
        raise ValueError("The selected hybrid evaluation is not verified complete")
    selected = {row["case_id"]: row for row in selection["cases"]}
    if len(selected) != 50 or set(selected) != {row["case_id"] for row in cases}:
        raise ValueError("CSV case membership differs from the completion certificate")
    for row in cases:
        original = selected[row["case_id"]]
        for key in ("task", "variant", "context_version", "archive", "job_id",
                    "result_sha256", "outcome_sha256", "artifact_manifest_sha256"):
            if row[key] != original[key]:
                raise ValueError(f"CSV/certificate mismatch in {key}: {row['case_id']}")
        if row["native_score"] != original["native_score"] or (row["native_success"] == "true") != original["native_success"]:
            raise ValueError("CSV terminal score/success mismatch")
        for key in ("eval_seed", "layout_id", "reset_seed", "simulator_initial_seed", "policy_rng_seed"):
            if int(row[key]) != original["evaluation_case"][key]:
                raise ValueError(f"CSV seed mismatch: {key}")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(data: dict, destination: Path, compact: bool = False) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.labelcolor": "#344256", "text.color": "#1E2B3D",
        "xtick.color": "#344256", "ytick.color": "#6D7888",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    tasks = [t for t in data["tasks"] if not compact or t["task"] in data["lowest_five"]]
    width = 16.6 if compact else 25.4
    fig = plt.figure(figsize=(width, 6.9))
    # Keep a dedicated right-hand panel, with exactly the same 0–100 scale.
    grid = fig.add_gridspec(1, 2, width_ratios=[len(tasks), 1.25],
                           left=.047 if not compact else .063, right=.983,
                           top=.71, bottom=.225, wspace=.09)
    ax = fig.add_subplot(grid[0, 0]); mean_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    positions = np.arange(len(tasks)) * 1.15
    bar_width = .153
    for j, (model, color) in enumerate(zip(data["models"], COLORS)):
        offsets = positions + (j - 2) * bar_width
        values = [t["scores"][model] for t in tasks]
        bars = ax.bar(offsets, values, width=bar_width * .9, color=color,
                      edgecolor="white", linewidth=.35, zorder=3)
        for bar, value in zip(bars, values):
            label = "100" if math.isclose(value, 100) else f"{value:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.6, label,
                    ha="center", va="bottom", fontsize=8.2,
                    color="#AD4F22" if model == HYBRID else "#586575",
                    fontweight="bold" if model == HYBRID else "normal")
        value = data["ten_task_mean"][model]
        mean_ax.bar(j, value, width=.69, color=color, edgecolor="white", linewidth=.4, zorder=3)
        mean_ax.text(j, value + 2.0, f"{value:.1f}", ha="center", va="bottom",
                     fontsize=11.2, fontweight="bold",
                     color="#AD4F22" if model == HYBRID else "#344256")
    for plot_ax in (ax, mean_ax):
        plot_ax.set_ylim(0, 109)
        plot_ax.set_yticks(np.arange(0, 101, 20))
        plot_ax.grid(axis="y", color="#E7EBF0", linewidth=.65, zorder=0)
        plot_ax.tick_params(axis="both", length=0, pad=11)
        for side in ("top", "right", "left"):
            plot_ax.spines[side].set_visible(False)
        plot_ax.spines["bottom"].set_color("#D6DDE5")
        plot_ax.spines["bottom"].set_linewidth(.8)
    ax.set_xlim(positions[0] - .61, positions[-1] + .61)
    ax.set_xticks(positions, [LABELS[t["task"]] for t in tasks], fontsize=10.5, linespacing=1.6)
    ax.set_ylabel("Score (0–100)", fontsize=11.5, labelpad=12)
    mean_ax.set_facecolor("#F4F6F9")
    mean_ax.tick_params(axis="y", labelleft=False)
    mean_ax.set_xlim(-.7, 4.7)
    mean_ax.set_xticks([2], ["Mean · all 10 tasks"], fontsize=11.5, fontweight="bold")
    mean_ax.set_title("TEN-TASK AVERAGE", fontsize=10, fontweight="bold", color="#637186", pad=12)
    left = .047 if not compact else .063
    title = "RoboDojo  |  Task scores" if not compact else "RoboDojo  |  Five low-success tasks"
    subtitle = ("Official simulation top 3 + π0.5, compared with π0.5 + GPT (v3)"
                if not compact else "Lowest π0.5 success rates in our panel • Right panel always averages all 10 tasks")
    fig.text(left, .928, title, fontsize=24, fontweight="bold", ha="left")
    fig.text(left, .874, subtitle, fontsize=12.6, color="#657286", ha="left")
    legend_labels = ["DM0.5  ·  #1", "GalaxeaVLA (G0.5)  ·  #2", "Xiaomi-Robotics-1  ·  #3",
                     "π0.5", "π0.5 + GPT  ·  v3"]
    fig.legend([Patch(facecolor=c, edgecolor="none") for c in COLORS], legend_labels,
               loc="upper left", bbox_to_anchor=(left - .006, .829), ncol=5,
               frameon=False, fontsize=11.5 if compact else 12,
               handlelength=1.35, handleheight=.85, columnspacing=2.0, labelspacing=1)
    fig.text(.983, .934, "COMPLETE HYBRID PANEL   50 / 50", ha="right", fontsize=10,
             color="#AD4F22", fontweight="bold")
    fig.text(left, .113, "* Generalization tasks: 40% standard + 60% random. Official scores are reweighted to this mix; all task means are equally weighted.",
             fontsize=9.4, color="#637186")
    fig.text(left, .075, "Hybrid v3: 5 complete rollouts/task, including native failures. Official aggregates use different samples/seeds: reference comparison, not a paired baseline.",
             fontsize=9.4, color="#637186")
    fig.text(left, .037, "Source: robodojo-benchmark.com/leaderboard  •  Ranking by official overall simulation Score  •  Data release: 2026-09-10  |  Retrieved: 2026-09-13",
             fontsize=8.8, color="#8791A0")
    stem = "scores_lowest5_plus_mean10" if compact else "scores_all10_plus_mean"
    for extension in ("png", "svg", "pdf"):
        fig.savefig(destination / f"{stem}.{extension}", dpi=230)
    # Smaller actual render for visual inspection and inline display.
    fig.savefig(destination / f"{stem}_preview.png", dpi=100)
    plt.close(fig)
    return {"stem": stem, "displayed_tasks": [t["task"] for t in tasks],
            "mean_task_count": 10, "formats": ["png", "svg", "pdf"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hybrid-csv", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--official-script", type=Path, help="Offline reproduction using the saved public JS asset")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing figure directory")
    certificate_bytes = args.certificate.read_bytes()
    if sha(certificate_bytes) != CERTIFICATE_SHA:
        raise ValueError("Hybrid completion certificate is not the frozen v3 panel")
    csv_bytes = args.hybrid_csv.read_bytes()
    cases = parse_hybrid(csv_bytes)
    verify_selection(cases, json.loads(certificate_bytes))
    fetched_utc = datetime.now(timezone.utc).isoformat()
    if args.official_script:
        script_bytes = args.official_script.read_bytes()
        homepage_bytes = None
        script_url = urllib.parse.urljoin(SITE, "assets/" + args.official_script.name)
    else:
        homepage_bytes = fetch(SITE)
        assets = re.findall(r'<script[^>]*src="([^"]+)"', homepage_bytes.decode())
        scripts = [asset for asset in assets if asset.startswith("/assets/index-") and asset.endswith(".js")]
        if len(scripts) != 1:
            raise ValueError("Could not identify the current website data bundle")
        script_url = urllib.parse.urljoin(SITE, scripts[0])
        script_bytes = fetch(script_url)
    official = parse_official(script_bytes.decode())
    data = aggregate(official, cases)
    data["provenance"] = {
        "retrieved_utc": fetched_utc, "leaderboard_url": SITE + "leaderboard",
        "script_url": script_url, "script_sha256": sha(script_bytes),
        "homepage_sha256": sha(homepage_bytes) if homepage_bytes else None,
        "published_data_date": official["published_data_date"],
        "official_simulation_ranking": official["ranking"],
        "hybrid_csv": str(args.hybrid_csv.resolve()), "hybrid_csv_sha256": sha(csv_bytes),
        "completion_certificate": str(args.certificate.resolve()), "completion_certificate_sha256": sha(certificate_bytes),
        "hybrid_case_count": len(cases), "hybrid_successes": sum(r["native_success"] == "true" for r in cases),
        "context_version": "v3", "model": "gpt-6-astra", "reasoning_effort": "xhigh",
        "official_reference_is_same_seed_baseline": False,
        "mean_definition": "Unweighted mean of all ten task scores; each official task is reweighted to the panel variant proportions.",
        "uncertainty_note": "Five local rollouts per task; no official per-episode uncertainty available; no significance claim.",
    }
    args.output.mkdir(parents=True)
    sources = args.output / "sources"; sources.mkdir()
    (sources / Path(urllib.parse.urlparse(script_url).path).name).write_bytes(script_bytes)
    if homepage_bytes:
        (sources / "homepage.html").write_bytes(homepage_bytes)
    (sources / "hybrid50_cases.csv").write_bytes(csv_bytes)
    (sources / "hybrid50_completion.json").write_bytes(certificate_bytes)
    (args.output / "data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    write_csv(args.output / "task_scores.csv",
              [{"task": row["task"], **row["scores"]} for row in data["tasks"]]
              + [{"task": "MEAN_ALL_10", **data["ten_task_mean"]}], ["task"] + data["models"])
    write_csv(args.output / "variant_scores.csv", data["variants"], list(data["variants"][0]))
    rendered = [plot(data, args.output), plot(data, args.output, compact=True)]
    shutil.copyfile(__file__, args.output / "plot_score_comparison.py")
    products = {str(p.relative_to(args.output)): sha(p.read_bytes())
                for p in args.output.rglob("*") if p.is_file()}
    (args.output / "manifest.json").write_text(json.dumps({"created_utc": fetched_utc,
                  "figures": rendered, "sha256": products}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "ten_task_mean": data["ten_task_mean"],
                      "source_sha256": sha(script_bytes), "figures": rendered}, indent=2))


if __name__ == "__main__":
    main()
