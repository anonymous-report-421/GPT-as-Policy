"""Data-only checks; no network, model calls, or rollout mutations."""
import csv
import io
import json
import unittest

from hybrid_rollout.robodojo.plot_score_comparison import (
    BASELINE, HYBRID, MODEL_KEYS, TASKS, aggregate, parse_hybrid, parse_official, verify_selection,
)


def cases():
    generalization = {"arrange_largest_number", "pack_objects_into_box", "fold_clothes"}
    return [{"case_id": f"{task}-{i}", "task": task,
             "variant": "random" if task in generalization and i >= 2 else "standard",
             "context_version": "v3", "native_score": i / 4,
             "native_success": "true" if i == 4 else "false"}
            for task in TASKS for i in range(5)]


def official():
    ranking = [{"model": name, "overall_score": 4-i} for i, name in enumerate(MODEL_KEYS)]
    return {"ranking": ranking, "models": {
        name: {task + suffix: {"score": 10 if not suffix else 90, "successRate": 5}
               for task in TASKS for suffix in ("", "_random")}
        for name in MODEL_KEYS}}


def csv_bytes(rows):
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


class ScoreComparisonTests(unittest.TestCase):
    def test_generalization_uses_2_3_not_unweighted_or_standard_only(self):
        data = aggregate(official(), cases())
        self.assertEqual(data["tasks"][3]["scores"][BASELINE], 58)
        self.assertEqual(data["tasks"][0]["scores"][BASELINE], 10)

    def test_ten_task_mean_and_lowest_five_do_not_change_mean(self):
        data = aggregate(official(), cases())
        self.assertEqual(data["ten_task_mean"][BASELINE], (10 * 7 + 58 * 3) / 10)
        self.assertEqual(data["ten_task_mean"][HYBRID], 50)
        self.assertEqual(len(data["lowest_five"]), 5)

    def test_native_failures_keep_their_partial_scores(self):
        data = aggregate(official(), cases())
        self.assertEqual(data["tasks"][0]["scores"][HYBRID], 50)  # Success rate is only 20%.

    def test_missing_official_score_is_not_zero(self):
        source = official(); del source["models"][BASELINE]["fold_clothes_random"]
        with self.assertRaises(KeyError):
            aggregate(source, cases())

    def test_fifty_unique_v3_cases_required(self):
        rows = cases()
        self.assertEqual(len(parse_hybrid(csv_bytes(rows))), 50)
        rows[-1] = rows[0]
        with self.assertRaises(ValueError):
            parse_hybrid(csv_bytes(rows))

    def test_variant_count_and_version_checked(self):
        for field, value in [("context_version", "v2"), ("variant", "random"), ("native_score", "nan")]:
            rows = cases(); rows[0][field] = value
            with self.assertRaises(ValueError):
                parse_hybrid(csv_bytes(rows))

    def test_ranking_sorted_by_overall_score(self):
        source = official()
        rows = []
        for i, name in enumerate(MODEL_KEYS):
            rows.append('{model:' + json.dumps(name) + ',team:"Example",generalizationStd:o(0,0),average:o(' + str(4-i) + ',0)}')
        tasks = {MODEL_KEYS[name]: data for name, data in source["models"].items()}
        script = 'fc="2026-09-10";[' + ','.join(reversed(rows)) + '];' + json.dumps(tasks, separators=(",", ":"))
        parsed = parse_official(script)
        self.assertEqual([r["model"] for r in parsed["ranking"][:3]], list(MODEL_KEYS)[:3])
        self.assertEqual(parsed["models"][BASELINE]["organize_table"]["score"], 10)

    def test_certificate_checks_scores_and_seeds(self):
        rows = cases(); originals = []
        seeds = ("eval_seed", "layout_id", "reset_seed", "simulator_initial_seed", "policy_rng_seed")
        for row in rows:
            row.update({k: "0" for k in seeds})
            row.update({k: "fixture" for k in ("archive", "job_id", "result_sha256", "outcome_sha256", "artifact_manifest_sha256")})
            originals.append({**row, "native_success": row["native_success"] == "true",
                              "evaluation_case": {k: 0 for k in seeds}})
        certificate = {"selection": {"complete": True, "platform_verified": True,
                                      "errors": [], "missing": [], "cases": originals}}
        verify_selection(rows, certificate)
        rows[0]["native_score"] = 1
        with self.assertRaises(ValueError):
            verify_selection(rows, certificate)
        rows[0]["native_score"] = 0
        rows[0]["reset_seed"] = "1"
        with self.assertRaises(ValueError):
            verify_selection(rows, certificate)

    def test_certificate_requires_full_native_completion(self):
        with self.assertRaises(ValueError):
            verify_selection(cases(), {"selection": {"complete": False}})


if __name__ == "__main__":
    unittest.main()
