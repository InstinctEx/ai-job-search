#!/usr/bin/env python3
"""Query and atomically update /rank state without loading it into context.

Usage:
  python3 tools/rank_state.py candidates [--all] [--focus TEXT] [--limit N]
  python3 tools/rank_state.py sweep --write [--exclude KEY,KEY]
  python3 tools/rank_state.py apply --results RESULTS_JSON

All commands print JSON. The tracker is read-only; only ``apply`` and
``sweep --write`` may replace the state file.
"""

import argparse
import copy
import csv
import json
import os
import re
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "job_scraper" / "seen_jobs.json"
TRACKER = ROOT / "job_search_tracker.csv"

DEFAULT_LIMIT = 10
URGENT_DAYS = 7
STALE_DAYS = 30
VERDICTS = {"PASS", "FAIL", "FLAG"}
WEIGHTS = {
    "technical": 0.30,
    "experience": 0.25,
    "behavioral": 0.15,
    "career": 0.30,
}
BANDS = (
    (75, "Strong Fit"),
    (60, "Good Fit"),
    (45, "Moderate Fit"),
    (30, "Weak Fit"),
    (0, "Poor Fit"),
)
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_state(path: Path) -> tuple[dict, dict]:
    """Return the full document and its mutable seen-entry mapping."""
    if not path.is_file():
        raise ValueError(f"{path} not found - run /scrape first")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    seen = document.get("seen") if isinstance(document, dict) and "seen" in document else document
    if not isinstance(seen, dict):
        raise ValueError(f"{path}: expected an object of job entries")
    if not all(isinstance(value, dict) for value in seen.values()):
        raise ValueError(f"{path}: every job entry must be an object")
    return document, seen


def save_state(path: Path, document: dict) -> None:
    """Write beside the destination and atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=".seen_jobs.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def parse_iso(value) -> date | None:
    if not isinstance(value, str) or not ISO_DATE.fullmatch(value.strip()):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def normalize(value) -> str:
    return " ".join(str(value or "").casefold().split())


def tracker_pairs(path: Path) -> set[tuple[str, str]]:
    """Read company+role exclusions without ever modifying the tracker."""
    if not path.is_file():
        return set()
    pairs = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            company = normalize(row.get("company"))
            role = normalize(row.get("role"))
            if company and role:
                pairs.add((company, role))
    return pairs


def focus_text(entry: dict) -> str:
    notes = []
    for field in ("strengths", "gaps"):
        value = entry.get(field)
        if isinstance(value, list):
            notes.extend(str(item) for item in value)
    return " ".join([str(entry.get("title") or ""), *notes]).casefold()


def cmd_candidates(args) -> dict:
    _, seen = load_state(args.state)
    exclusions = tracker_pairs(args.tracker)
    eligible = []
    excluded_by_tracker = 0

    for key, entry in seen.items():
        if not args.all and entry.get("status") != "new":
            continue
        pair = (normalize(entry.get("company")), normalize(entry.get("title")))
        if pair in exclusions:
            excluded_by_tracker += 1
            continue
        if args.focus and args.focus.casefold() not in focus_text(entry):
            continue
        eligible.append(
            {
                "key": key,
                "title": entry.get("title"),
                "company": entry.get("company"),
                "url": entry.get("url"),
            }
        )

    selected = eligible[: args.limit]
    return {
        "eligible": len(eligible),
        "selected": selected,
        "deferred": len(eligible) - len(selected),
        "excluded_by_tracker": excluded_by_tracker,
        "total_entries": len(seen),
    }


def cmd_sweep(args) -> dict:
    document, seen = load_state(args.state)
    excluded = {key for key in (args.exclude or "").split(",") if key}
    expired = []
    closing_soon = []
    unparseable = []
    swept = 0

    for key, entry in seen.items():
        if entry.get("status") != "ranked" or key in excluded:
            continue
        swept += 1
        raw = entry.get("deadline")
        if raw in (None, ""):
            continue
        parsed = parse_iso(raw)
        if parsed is None:
            unparseable.append(
                {"key": key, "portal": entry.get("portal"), "deadline": raw}
            )
            continue
        row = {
            "key": key,
            "title": entry.get("title"),
            "company": entry.get("company"),
            "url": entry.get("url"),
            "deadline": raw,
        }
        if parsed < args.today:
            expired.append(row)
        elif parsed <= args.today + timedelta(days=URGENT_DAYS):
            closing_soon.append(row)

    if args.write and expired:
        updated = copy.deepcopy(document)
        updated_seen = updated["seen"] if "seen" in updated else updated
        for row in expired:
            updated_seen[row["key"]]["status"] = "expired"
        save_state(args.state, updated)

    return {
        "swept": swept,
        "newly_expired": expired,
        "closing_soon": sorted(closing_soon, key=lambda row: row["deadline"]),
        "unparseable_deadlines": unparseable,
        "written": bool(args.write and expired),
    }


def overall_score(scores: dict) -> int:
    total = 0.0
    for dimension, weight in WEIGHTS.items():
        value = scores.get(dimension)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"missing or non-numeric score '{dimension}'")
        if not 0 <= value <= 100:
            raise ValueError(f"score '{dimension}' must be between 0 and 100")
        total += value * weight
    return int(total + 0.5)


def score_band(score: int) -> str:
    return next(name for floor, name in BANDS if score >= floor)


def validate_results(results, seen: dict) -> list[dict]:
    if not isinstance(results, list):
        raise ValueError("results file must be a JSON array of scoring objects")
    errors = []
    keys = set()
    for result in results:
        if not isinstance(result, dict):
            errors.append({"key": None, "error": "each result must be an object"})
            continue
        key = result.get("key")
        if key not in seen:
            errors.append({"key": key, "error": "no such key in seen_jobs.json"})
            continue
        if key in keys:
            errors.append({"key": key, "error": "duplicate result key"})
            continue
        keys.add(key)
        status = result.get("status", "scored")
        if status not in {"scored", "expired"}:
            errors.append({"key": key, "error": f"unsupported status '{status}'"})
            continue
        if status == "expired":
            continue
        try:
            overall_score(result.get("scores") or {})
        except ValueError as exc:
            errors.append({"key": key, "error": str(exc)})
        for field in ("location_verdict", "language_gate"):
            if result.get(field) not in VERDICTS:
                errors.append({"key": key, "error": f"invalid {field}"})
        for field in ("strengths", "gaps"):
            if not isinstance(result.get(field), list):
                errors.append({"key": key, "error": f"{field} must be an array"})
        if "deadline" not in result:
            errors.append({"key": key, "error": "deadline is required (use null when absent)"})
        elif result["deadline"] is not None and parse_iso(result["deadline"]) is None:
            errors.append({"key": key, "error": "deadline must be YYYY-MM-DD or null"})
    return errors


def result_row(
    key: str, entry: dict, today: date
) -> tuple[dict, dict | None, dict | None]:
    deadline = parse_iso(entry.get("deadline"))
    posted = parse_iso(entry.get("posted_date"))
    age_days = (today - posted).days if posted and posted <= today else None
    row = {
        "key": key,
        "title": entry.get("title"),
        "company": entry.get("company"),
        "location": entry.get("location"),
        "url": entry.get("url"),
        "score": entry["rank_score"],
        "verdict": entry["rank_verdict"],
        "location_verdict": entry["location_verdict"],
        "language_gate": entry["language_gate"],
        "language_note": entry.get("language_note"),
        "deadline": entry.get("deadline"),
        "urgent": bool(deadline and today <= deadline <= today + timedelta(days=URGENT_DAYS)),
        "posted_date": entry.get("posted_date"),
        "stale": bool(age_days is not None and age_days > STALE_DAYS),
        "age_days": age_days,
        "strengths": entry["strengths"],
        "gaps": entry["gaps"],
    }
    bad_posted = None
    if entry.get("posted_date") not in (None, "") and posted is None:
        bad_posted = {
            "key": key,
            "portal": entry.get("portal"),
            "posted_date": entry.get("posted_date"),
        }
    bad_deadline = None
    if entry.get("deadline") not in (None, "") and deadline is None:
        bad_deadline = {
            "key": key,
            "portal": entry.get("portal"),
            "deadline": entry.get("deadline"),
        }
    return row, bad_posted, bad_deadline


def cmd_apply(args) -> tuple[dict, int]:
    document, seen = load_state(args.state)
    try:
        results = json.loads(args.results.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read results file {args.results}: {exc}") from exc
    if isinstance(results, dict):
        results = results.get("results")

    errors = validate_results(results, seen)
    if errors:
        return {
            "ranked": [],
            "vetoed": [],
            "expired": [],
            "errors": errors,
            "written": False,
        }, 1

    updated = copy.deepcopy(document)
    updated_seen = updated["seen"] if "seen" in updated else updated
    rows = []
    expired = []
    unparseable_deadlines = []
    unparseable_posted_dates = []

    for result in results:
        key = result["key"]
        entry = updated_seen[key]
        if result.get("status", "scored") == "expired":
            entry["status"] = "expired"
            expired.append(
                {
                    "key": key,
                    "title": entry.get("title"),
                    "company": entry.get("company"),
                    "url": entry.get("url"),
                }
            )
            continue

        if entry.get("location") in VERDICTS:
            entry.pop("location")
        score = overall_score(result["scores"])
        entry["status"] = "ranked"
        entry["rank_score"] = score
        entry["rank_verdict"] = score_band(score)
        entry["rank_date"] = args.today.isoformat()
        entry["location_verdict"] = result["location_verdict"]
        entry["language_gate"] = result["language_gate"]
        if result["language_gate"] == "PASS":
            entry.pop("language_note", None)
        else:
            entry["language_note"] = result.get("language_note")
        if result["deadline"] is not None:
            entry["deadline"] = result["deadline"]
        entry["strengths"] = copy.deepcopy(result["strengths"])
        entry["gaps"] = copy.deepcopy(result["gaps"])

        deadline = parse_iso(entry.get("deadline"))
        if deadline and deadline < args.today:
            entry["status"] = "expired"
            expired.append(
                {
                    "key": key,
                    "title": entry.get("title"),
                    "company": entry.get("company"),
                    "url": entry.get("url"),
                    "deadline": entry.get("deadline"),
                }
            )
            continue

        row, bad_posted, bad_deadline = result_row(key, entry, args.today)
        rows.append(row)
        if bad_posted:
            unparseable_posted_dates.append(bad_posted)
        if bad_deadline:
            unparseable_deadlines.append(bad_deadline)

    if not args.dry_run:
        save_state(args.state, updated)

    rows.sort(key=lambda row: (row["score"], row["urgent"]), reverse=True)
    vetoed = [
        row
        for row in rows
        if row["location_verdict"] == "FAIL" or row["language_gate"] == "FAIL"
    ]
    vetoed_keys = {row["key"] for row in vetoed}
    ranked = [row for row in rows if row["key"] not in vetoed_keys]
    return {
        "ranked": ranked,
        "vetoed": vetoed,
        "expired": expired,
        "unparseable_deadlines": unparseable_deadlines,
        "unparseable_posted_dates": unparseable_posted_dates,
        "errors": [],
        "written": not args.dry_run,
    }, 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state", type=Path, default=STATE)
    common.add_argument("--today", type=date.fromisoformat, default=date.today())

    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="command", required=True)

    candidates = commands.add_parser("candidates", parents=[common])
    candidates.add_argument("--tracker", type=Path, default=TRACKER)
    candidates.add_argument("--all", action="store_true")
    candidates.add_argument("--focus")
    candidates.add_argument("--limit", type=positive_int, default=DEFAULT_LIMIT)

    sweep = commands.add_parser("sweep", parents=[common])
    sweep.add_argument("--exclude", default="")
    sweep.add_argument("--write", action="store_true")

    apply = commands.add_parser("apply", parents=[common])
    apply.add_argument("--results", type=Path, required=True)
    apply.add_argument("--dry-run", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "candidates":
            output, code = cmd_candidates(args), 0
        elif args.command == "sweep":
            output, code = cmd_sweep(args), 0
        else:
            output, code = cmd_apply(args)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
