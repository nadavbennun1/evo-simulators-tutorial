#!/usr/bin/env python3
"""Aggregate an authorized assessment export without printing individual rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        records = list(csv.DictReader(text.splitlines()))
    elif text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    for record in records:
        if isinstance(record.get("payload"), str):
            record["payload"] = json.loads(record["payload"])
    return records


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def median_or_none(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def received_key(record: dict[str, Any]) -> tuple[str, str]:
    value = str(record.get("received_at") or "")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        parsed = value
    return parsed, str(record.get("id") or "")


def answer_map(record: dict[str, Any], family: str) -> dict[str, Any]:
    answers = record["payload"].get("answers", {}).get(family, [])
    return {answer["question_id"]: answer.get("response") for answer in answers}


def score_record(record: dict[str, Any], question_bank: dict[str, Any]) -> tuple[int, dict[str, bool]]:
    actual = answer_map(record, "knowledge")
    item_scores = {}
    for question in question_bank["knowledge_questions"]:
        variant = question.get("variants", {}).get(record["phase"], {})
        expected = variant["correct_response"] if "correct_response" in variant else question["correct_response"]
        item_scores[question["question_id"]] = canonical(actual.get(question["question_id"])) == canonical(expected)
    return sum(item_scores.values()), item_scores


def pairing_key(record: dict[str, Any]) -> str:
    code = record["payload"].get("pairing_code")
    if code:
        return "code:" + str(code).upper()
    return "participant:" + str(record.get("participant_id"))


def summarize(records: list[dict[str, Any]], question_bank: dict[str, Any], group_by_venue: bool = False) -> dict[str, Any]:
    version = question_bank["assessment_version"]
    completed = [
        record for record in records
        if record.get("event_type") == "completed"
        and record.get("assessment_version") == version
        and record.get("phase") in {"pre", "post"}
        and isinstance(record.get("payload"), dict)
    ]
    unique = {str(record.get("id")): record for record in completed}
    completed = list(unique.values())
    scored: dict[str, list[tuple[dict[str, Any], int, dict[str, bool]]]] = {"pre": [], "post": []}
    for record in completed:
        score, items = score_record(record, question_bank)
        scored[record["phase"]].append((record, score, items))

    latest: dict[str, dict[str, tuple[dict[str, Any], int, dict[str, bool]]]] = defaultdict(dict)
    for phase in ("pre", "post"):
        for row in scored[phase]:
            key = pairing_key(row[0])
            prior = latest[key].get(phase)
            if prior is None or received_key(row[0]) > received_key(prior[0]):
                latest[key][phase] = row
    matched = [pair for pair in latest.values() if "pre" in pair and "post" in pair]
    pre_scores = [row[1] for row in scored["pre"]]
    post_scores = [row[1] for row in scored["post"]]
    changes = [pair["post"][1] - pair["pre"][1] for pair in matched]

    item_results: dict[str, dict[str, Any]] = {}
    for question in question_bank["knowledge_questions"]:
        item_id = question["question_id"]
        item_results[item_id] = {}
        for phase in ("pre", "post"):
            values = [int(row[2][item_id]) for row in scored[phase]]
            item_results[item_id][phase + "_correct_percent"] = round(100 * statistics.fmean(values), 1) if values else None

    confidence: dict[str, dict[str, float | None]] = {}
    for question in question_bank["confidence_questions"]:
        item_id = question["question_id"]
        confidence[item_id] = {}
        for phase in ("pre", "post"):
            values = [answer_map(row[0], "confidence").get(item_id) for row in scored[phase]]
            confidence[item_id][phase + "_mean"] = mean_or_none([float(value) for value in values if isinstance(value, (int, float))])

    summary: dict[str, Any] = {
        "assessment_version": version,
        "completed_n": {phase: len(scored[phase]) for phase in ("pre", "post")},
        "matched_n": len(matched),
        "knowledge_score_0_to_6": {
            "pre_mean": mean_or_none(pre_scores), "pre_median": median_or_none(pre_scores),
            "post_mean": mean_or_none(post_scores), "post_median": median_or_none(post_scores),
            "matched_change_mean": mean_or_none(changes), "matched_change_median": median_or_none(changes),
        },
        "item_correct_percent": item_results,
        "confidence": confidence,
    }
    if group_by_venue:
        venues: dict[str, Any] = {}
        for venue in ("NYU", "UMN", "TAU", "ONLINE"):
            venues[venue] = {}
            for phase in ("pre", "post"):
                values = [row[1] for row in scored[phase] if row[0].get("venue") == venue]
                venues[venue][phase] = {"n": len(values), "mean": mean_or_none(values), "median": median_or_none(values)}
        summary["by_venue"] = venues
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="Google Sheet CSV, JSON array, or JSONL export")
    parser.add_argument("--question-bank", type=Path, default=Path(__file__).parents[1] / "questions" / "v1.1.0.json")
    parser.add_argument("--group-by-venue", action="store_true")
    args = parser.parse_args()
    bank_bytes = args.question_bank.read_bytes()
    bank = json.loads(bank_bytes)
    records = load_records(args.export)
    expected_hash = hashlib.sha256(bank_bytes).hexdigest()
    mismatches = [
        record for record in records
        if record.get("event_type") == "completed"
        and record.get("assessment_version") == bank["assessment_version"]
        and record.get("payload", {}).get("question_bank_hash") not in {None, expected_hash}
    ]
    if mismatches:
        raise SystemExit(
            f"Refusing to score {len(mismatches)} completed event(s): question-bank hash does not match {args.question_bank}"
        )
    print(json.dumps(summarize(records, bank, args.group_by_venue), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
