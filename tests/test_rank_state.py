"""Tests for the /rank state query and atomic patch helper."""

import importlib.util
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "rank_state.py"
COMMAND = REPO / ".claude" / "commands" / "rank.md"
TODAY = "2026-09-03"


def entry(**overrides):
    value = {
        "title": "SOC Analyst",
        "company": "Acme",
        "url": "https://example.com/jobs/1",
        "first_seen": "2026-09-01",
        "status": "new",
        "deadline": None,
        "portal": "linkedin-search",
    }
    value.update(overrides)
    return value


def sections(text: str) -> dict[str, str]:
    result = {}
    for part in text.split("\n## ")[1:]:
        heading, _, body = part.partition("\n")
        result[heading.strip()] = body
    return result


def documented_rank_fields() -> set[str]:
    """Derive the write contract from Step 4 instead of duplicating it here."""
    step4 = sections(COMMAND.read_text(encoding="utf-8"))["Step 4: Update State"]
    ranked = step4.partition("- Ranked jobs:")[2].partition("\n- Dead or past-deadline jobs:")[0]
    return set(re.findall(r'"([a-z_]+)"\s*:', ranked))


class RankStateCase(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.tmp = Path(self._temporary.name)
        self.state = self.tmp / "seen_jobs.json"
        self.tracker = self.tmp / "job_search_tracker.csv"
        self.addCleanup(self._temporary.cleanup)

    def write_state(self, seen, *, wrapped=True):
        document = {"seen": seen, "schema_version": 1} if wrapped else seen
        self.state.write_text(json.dumps(document), encoding="utf-8")

    def read_document(self):
        return json.loads(self.state.read_text(encoding="utf-8"))

    def read_seen(self):
        document = self.read_document()
        return document.get("seen", document)

    def results_file(self, results):
        path = self.tmp / "results.json"
        path.write_text(json.dumps(results), encoding="utf-8")
        return path

    def run_tool(self, command, *arguments, expected=0):
        process = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                command,
                *map(str, arguments),
                "--state",
                str(self.state),
                "--today",
                TODAY,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(process.returncode, expected, process.stderr or process.stdout)
        return json.loads(process.stdout)

    def scored_result(self, key="a", **overrides):
        result = {
            "key": key,
            "status": "scored",
            "scores": {
                "technical": 80,
                "experience": 60,
                "behavioral": 70,
                "career": 75,
            },
            "location_verdict": "PASS",
            "language_gate": "PASS",
            "language_note": None,
            "deadline": None,
            "strengths": ["Direct SOC experience"],
            "gaps": ["No named cloud platform"],
        }
        result.update(overrides)
        return result


class Candidates(RankStateCase):
    def test_projects_only_the_four_agreed_fields(self):
        self.write_state(
            {
                "a": entry(
                    strengths=["secret fit note"],
                    gaps=["secret gap"],
                    unrelated={"large": "payload"},
                ),
                "b": entry(status="ranked"),
                "c": entry(status="skipped"),
            }
        )
        output = self.run_tool("candidates", "--tracker", self.tracker)
        self.assertEqual([row["key"] for row in output["selected"]], ["a"])
        self.assertEqual(
            set(output["selected"][0]),
            {"key", "title", "company", "url"},
        )
        self.assertNotIn("secret fit note", json.dumps(output))

    def test_limit_is_applied_after_filters_and_reports_deferral(self):
        seen = {f"job-{index}": entry(title=f"Role {index}") for index in range(14)}
        seen["not-new"] = entry(status="ranked")
        self.write_state(seen)
        output = self.run_tool(
            "candidates", "--limit", 10, "--tracker", self.tracker
        )
        self.assertEqual(len(output["selected"]), 10)
        self.assertEqual(output["eligible"], 14)
        self.assertEqual(output["deferred"], 4)

    def test_tracker_exclusion_is_case_insensitive_and_tracker_is_read_only(self):
        self.write_state(
            {
                "a": entry(company="Acme", title="SOC Analyst"),
                "b": entry(company="Other", title="SOC Analyst"),
            }
        )
        self.tracker.write_text(
            "date,company,role\n2026-09-01, ACME , soc analyst \n",
            encoding="utf-8",
        )
        before = self.tracker.read_bytes()
        output = self.run_tool("candidates", "--tracker", self.tracker)
        self.assertEqual([row["key"] for row in output["selected"]], ["b"])
        self.assertEqual(output["excluded_by_tracker"], 1)
        self.assertEqual(self.tracker.read_bytes(), before)

    def test_focus_matches_title_and_stored_fit_notes(self):
        self.write_state(
            {
                "title": entry(title="Detection Engineer"),
                "strength": entry(strengths=["Detection engineering ownership"]),
                "gap": entry(gaps=["Needs detection engineering depth"]),
                "miss": entry(title="Accountant"),
            }
        )
        output = self.run_tool(
            "candidates", "--focus", "detection engineer", "--tracker", self.tracker
        )
        self.assertEqual(
            sorted(row["key"] for row in output["selected"]),
            ["gap", "strength", "title"],
        )

    def test_all_includes_non_new_entries_but_tracker_still_excludes(self):
        self.write_state(
            {
                "ranked": entry(status="ranked"),
                "expired": entry(status="expired", company="Old"),
            }
        )
        output = self.run_tool("candidates", "--all", "--tracker", self.tracker)
        self.assertEqual(
            sorted(row["key"] for row in output["selected"]),
            ["expired", "ranked"],
        )

    def test_limit_must_be_positive(self):
        self.write_state({"a": entry()})
        process = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "candidates",
                "--limit",
                "0",
                "--state",
                str(self.state),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(process.returncode, 0)


class Sweep(RankStateCase):
    def test_sweep_writes_only_past_ranked_deadlines(self):
        self.write_state(
            {
                "past": entry(status="ranked", deadline="2026-09-02"),
                "soon": entry(status="ranked", deadline="2026-09-06"),
                "future": entry(status="ranked", deadline="2026-12-01"),
                "new": entry(deadline="2026-09-02"),
            }
        )
        output = self.run_tool("sweep", "--write")
        self.assertEqual([row["key"] for row in output["newly_expired"]], ["past"])
        self.assertEqual([row["key"] for row in output["closing_soon"]], ["soon"])
        self.assertEqual(self.read_seen()["past"]["status"], "expired")
        self.assertEqual(self.read_seen()["new"]["status"], "new")

    def test_sweep_excludes_rescored_keys_and_reports_bad_dates(self):
        self.write_state(
            {
                "rescored": entry(status="ranked", deadline="2026-09-01"),
                "bad": entry(status="ranked", deadline="ASAP", portal="jobindex-search"),
            }
        )
        output = self.run_tool("sweep", "--write", "--exclude", "rescored")
        self.assertEqual(output["newly_expired"], [])
        self.assertEqual(output["unparseable_deadlines"][0]["portal"], "jobindex-search")
        self.assertEqual(self.read_seen()["rescored"]["status"], "ranked")

    def test_sweep_without_write_does_not_modify_state(self):
        self.write_state({"past": entry(status="ranked", deadline="2026-09-01")})
        before = self.state.read_bytes()
        output = self.run_tool("sweep")
        self.assertFalse(output["written"])
        self.assertEqual(self.state.read_bytes(), before)


class Apply(RankStateCase):
    def test_persists_every_ranked_field_documented_by_step4(self):
        self.write_state({"a": entry()})
        result = self.scored_result(
            language_gate="FLAG",
            language_note="Danish preferred",
            deadline="2026-09-08",
        )
        output = self.run_tool("apply", "--results", self.results_file([result]))
        stored = self.read_seen()["a"]
        self.assertLessEqual(documented_rank_fields(), set(stored))
        self.assertEqual(stored["rank_score"], 72)
        self.assertEqual(stored["rank_verdict"], "Good Fit")
        self.assertTrue(output["ranked"][0]["urgent"])

    def test_patch_preserves_unknown_and_wrapper_fields(self):
        metadata = {"source": "portal", "nested": [1, {"two": 2}]}
        self.write_state({"a": entry(custom_metadata=metadata)})
        self.run_tool(
            "apply", "--results", self.results_file([self.scored_result()])
        )
        document = self.read_document()
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["seen"]["a"]["custom_metadata"], metadata)

    def test_legacy_location_verdict_is_migrated_without_losing_a_real_place(self):
        self.write_state(
            {
                "legacy": entry(location="FLAG"),
                "place": entry(location="Athens, Greece"),
            }
        )
        self.run_tool(
            "apply",
            "--results",
            self.results_file(
                [
                    self.scored_result("legacy", location_verdict="PASS"),
                    self.scored_result("place", location_verdict="PASS"),
                ]
            ),
        )
        stored = self.read_seen()
        self.assertNotIn("location", stored["legacy"])
        self.assertEqual(stored["legacy"]["location_verdict"], "PASS")
        self.assertEqual(stored["place"]["location"], "Athens, Greece")

    def test_null_deadline_is_not_a_correction(self):
        self.write_state({"a": entry(deadline="2026-10-01")})
        self.run_tool(
            "apply", "--results", self.results_file([self.scored_result(deadline=None)])
        )
        self.assertEqual(self.read_seen()["a"]["deadline"], "2026-10-01")

    def test_veto_fields_and_arrays_are_persisted_verbatim(self):
        strengths = ["Uses C++", "Handles naïve input"]
        gaps = ["No AWS", {"source-shaped": "data remains data"}]
        self.write_state({"a": entry(language_note="obsolete")})
        result = self.scored_result(
            location_verdict="FLAG",
            language_gate="FAIL",
            language_note="Requires fluent Polish",
            strengths=strengths,
            gaps=gaps,
        )
        output = self.run_tool("apply", "--results", self.results_file([result]))
        stored = self.read_seen()["a"]
        self.assertEqual(stored["strengths"], strengths)
        self.assertEqual(stored["gaps"], gaps)
        self.assertEqual(stored["location_verdict"], "FLAG")
        self.assertEqual(stored["language_gate"], "FAIL")
        self.assertEqual(stored["language_note"], "Requires fluent Polish")
        self.assertEqual([row["key"] for row in output["vetoed"]], ["a"])

    def test_pass_removes_an_old_language_note(self):
        self.write_state({"a": entry(language_note="old warning")})
        self.run_tool(
            "apply", "--results", self.results_file([self.scored_result()])
        )
        self.assertNotIn("language_note", self.read_seen()["a"])

    def test_expired_result_and_past_deadline_both_expire(self):
        self.write_state({"closed": entry(), "past": entry()})
        output = self.run_tool(
            "apply",
            "--results",
            self.results_file(
                [
                    {"key": "closed", "status": "expired"},
                    self.scored_result("past", deadline="2026-09-01"),
                ]
            ),
        )
        self.assertEqual(
            sorted(row["key"] for row in output["expired"]),
            ["closed", "past"],
        )
        self.assertEqual(self.read_seen()["closed"]["status"], "expired")
        self.assertEqual(self.read_seen()["past"]["status"], "expired")

    def test_staleness_is_derived_without_becoming_a_veto(self):
        self.write_state({"a": entry(posted_date="2026-07-01")})
        output = self.run_tool(
            "apply", "--results", self.results_file([self.scored_result()])
        )
        self.assertTrue(output["ranked"][0]["stale"])
        self.assertGreater(output["ranked"][0]["age_days"], 30)

    def test_bad_posted_date_is_reported_with_its_portal(self):
        self.write_state({"a": entry(posted_date="last month", portal="freehire-search")})
        output = self.run_tool(
            "apply", "--results", self.results_file([self.scored_result()])
        )
        self.assertEqual(
            output["unparseable_posted_dates"][0]["portal"], "freehire-search"
        )

    def test_stored_bad_deadline_survives_null_and_is_reported(self):
        self.write_state({"a": entry(deadline="ASAP", portal="jobindex-search")})
        output = self.run_tool(
            "apply", "--results", self.results_file([self.scored_result(deadline=None)])
        )
        self.assertEqual(self.read_seen()["a"]["deadline"], "ASAP")
        self.assertEqual(
            output["unparseable_deadlines"][0]["portal"], "jobindex-search"
        )

    def test_validation_error_never_partially_writes(self):
        self.write_state({"a": entry(), "b": entry()})
        before = self.state.read_bytes()
        invalid = self.scored_result("b")
        invalid["scores"].pop("career")
        output = self.run_tool(
            "apply",
            "--results",
            self.results_file([self.scored_result("a"), invalid]),
            expected=1,
        )
        self.assertFalse(output["written"])
        self.assertEqual(self.state.read_bytes(), before)

    def test_supports_the_legacy_unwrapped_state_shape(self):
        self.write_state({"a": entry()}, wrapped=False)
        self.run_tool(
            "apply", "--results", self.results_file([self.scored_result()])
        )
        self.assertEqual(self.read_seen()["a"]["status"], "ranked")


class AtomicWrite(RankStateCase):
    def test_save_uses_replace_from_a_temporary_file_in_the_same_directory(self):
        specification = importlib.util.spec_from_file_location("rank_state", TOOL)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        real_replace = module.os.replace
        with mock.patch.object(module.os, "replace", wraps=real_replace) as replace:
            module.save_state(self.state, {"seen": {"a": entry()}})
        source, destination = replace.call_args.args
        self.assertEqual(Path(source).parent, self.state.parent)
        self.assertEqual(Path(destination), self.state)
        self.assertTrue(self.state.is_file())


if __name__ == "__main__":
    unittest.main()
