"""Source-bound shot/rally ledger, reviewed outcomes and reproducible factual reports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil

from badminton_proto import sha256_file
from player_identity import source_snapshot, _local

SCHEMA = "2.0.0"
FAMILIES = ("unknown", "serve", "clear", "drop", "smash", "drive", "net", "lift", "block", "push", "kill", "other")
STATUSES = ("unknown", "proposed", "accepted", "rejected")
FINISHES = ("undetermined", "direct_winner", "forced_error", "unforced_error", "other_fault", "not_applicable")
CAUSES = ("unknown", "ground_in", "ground_out", "net_failure", "body_fault", "service_fault", "other_fault", "let")
LINKS = ("last_contact_event_id", "direct_winner_event_id", "terminal_error_event_id", "preceding_pressure_event_id")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, names, label):
    require(type(value) is dict and set(value) == set(names.split()), f"Invalid {label} fields")


def identifier(value):
    require(type(value) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value), "IDs must be 1..80 letters, digits, underscores or hyphens")


def integer(value, low, high, label):
    require(type(value) is int and low <= value <= high, f"Invalid {label}")


def pair(value, label):
    require(type(value) is list and len(value) == 2 and all(type(v) is int and 0 <= v <= 10000 for v in value), f"Invalid {label}")


def evidence(value, frames):
    require(type(value) is list, "Evidence must be a list")
    for item in value:
        fields(item, "kind start_frame end_frame note", "evidence")
        require(item["kind"] in ("video", "manual", "official", "scoreboard"), "Invalid evidence type")
        require(type(item["note"]) is str and bool(item["note"].strip()), "Evidence needs a note")
        if item["start_frame"] is None:
            require(item["end_frame"] is None and item["kind"] in ("manual", "official"), "Video/scoreboard evidence needs frame bounds")
        else:
            integer(item["start_frame"], 0, frames-1, "evidence start")
            integer(item["end_frame"], item["start_frame"], frames-1, "evidence end")


def empty_document():
    return {"match": {"teams": {"A": [], "B": []}, "rules": {"profile_id": "unconfigured",
                "target_points": None, "win_by": None, "cap_points": None, "best_of_games": None}, "initial_scores": {}},
            "rallies": [], "events": [], "outcomes": [], "score_events": [],
            "analysis_rubric":{"id":"badmintonai-error-rubric-v1-draft","skill_context":"unspecified"},"notes": ""}


def new_rally(name, start, end):
    return {"id": name, "game": 1, "start_frame": start, "end_frame": end, "kind": "play",
            "coverage": "complete", "status": "proposed", "server": None, "receiver": None, "notes": ""}


def new_event(name, rally, frame):
    return {"id": name, "rally_id": rally, "start_frame": frame, "end_frame": frame, "kind": "contact",
            "player": None, "side": None, "status": "proposed", "shot_family": "unknown", "shot_status": "unknown",
            "direction": "unknown", "execution": "unknown", "evidence": [], "notes": ""}


def new_outcome(rally):
    return {"rally_id": rally, "status": "unknown", "winner": None, "terminal_cause": "unknown", "terminal_status": "unknown",
            "finish": "undetermined", "finish_status": "unknown", "error_player": None,
            **{key: None for key in LINKS}, "setup_sequence_event_ids": [], "score_before": None, "score_after": None,
            "evidence": [], "notes": ""}


def validate_document(document, source):
    fields(document, "match rallies events outcomes score_events analysis_rubric notes", "document")
    fields(document['analysis_rubric'],'id skill_context','analysis rubric')
    identifier(document['analysis_rubric']['id'])
    require(type(document['analysis_rubric']['skill_context']) is str and bool(document['analysis_rubric']['skill_context'].strip()),'Error rubric needs a skill context')
    require(type(document["notes"]) is str, "Notes must be text")
    match = document["match"]
    fields(match, "teams rules initial_scores", "match")
    fields(match["teams"], "A B", "teams")
    slots = {"P1", "P2", "P3", "P4"}
    owners = {}
    for side, members in match["teams"].items():
        require(type(members) is list and len(members) <= source["expected_players"]//2, "Team has too many players")
        for player in members:
            require(type(player) is str and player in slots and int(player[1]) <= source["expected_players"] and player not in owners,
                    "Invalid, duplicate or cross-team player slot")
            owners[player] = side
    rules = match["rules"]
    fields(rules, "profile_id target_points win_by cap_points best_of_games", "rules")
    identifier(rules["profile_id"])
    if rules["profile_id"] == "unconfigured":
        require(all(rules[key] is None for key in ("target_points", "win_by", "cap_points", "best_of_games")), "Unconfigured rules must not imply a scoring format")
    else:
        integer(rules["target_points"], 1, 100, "target points")
        integer(rules["win_by"], 1, 10, "winning margin")
        if rules["cap_points"] is not None:
            integer(rules["cap_points"], rules["target_points"], 1000, "point cap")
        integer(rules["best_of_games"], 1, 9, "best of games")
        require(rules["best_of_games"] % 2 == 1, "Best of games must be odd")
    require(type(match["initial_scores"]) is dict, "Initial scores must be a dictionary")
    for game, score in match["initial_scores"].items():
        require(type(game) is str and re.fullmatch(r"[1-9][0-9]*", game), "Invalid initial-score game")
        pair(score, "initial score")
    frames = source["frames"]
    for key in ("rallies", "events", "outcomes", "score_events"):
        require(type(document[key]) is list, f"{key} must be a list")
    rallies, events, outcomes = {}, {}, {}
    last_end, last_game = -1, 0
    for rally in document["rallies"]:
        fields(rally, "id game start_frame end_frame kind coverage status server receiver notes", "rally")
        identifier(rally["id"]); require(rally["id"] not in rallies, "Duplicate rally ID")
        integer(rally["game"], 1, rules["best_of_games"] or 99, "game index")
        integer(rally["start_frame"], 0, frames-1, "rally start")
        integer(rally["end_frame"], rally["start_frame"], frames-1, "rally end")
        require(rally["start_frame"] > last_end and rally["game"] >= last_game, "Rallies must be chronological and not overlap")
        require(rally["kind"] in ("play", "let", "replay", "practice") and rally["coverage"] in ("complete", "partial_start", "partial_end", "fragment"), "Invalid rally kind/coverage")
        require(rally["status"] in STATUSES and type(rally["notes"]) is str, "Invalid rally status/notes")
        for role in ("server", "receiver"):
            require(rally[role] is None or rally[role] in owners, "Server/receiver needs known side membership")
        if rally["server"] and rally["receiver"]:
            require(owners[rally["server"]] != owners[rally["receiver"]], "Server and receiver must be opponents")
        rallies[rally["id"]] = rally
        last_end, last_game = rally["end_frame"], rally["game"]
    for event in document["events"]:
        fields(event, "id rally_id start_frame end_frame kind player side status shot_family shot_status direction execution evidence notes", "event")
        identifier(event["id"]); require(event["id"] not in events, "Duplicate event ID")
        require(event["rally_id"] in rallies, "Event refers to missing rally")
        rally = rallies[event["rally_id"]]
        integer(event["start_frame"], rally["start_frame"], rally["end_frame"], "event start")
        integer(event["end_frame"], event["start_frame"], rally["end_frame"], "event end")
        require(event["kind"] in ("contact", "attempt"), "Invalid event kind")
        require(not any(event["start_frame"] < cut <= event["end_frame"] for cut in source.get("scene_boundaries",[])), "Event uncertainty crosses a sampled camera boundary")
        require(event["status"] in STATUSES and event["shot_status"] in STATUSES, "Invalid event/shot status")
        require(event["shot_family"] in FAMILIES, "Unknown shot family")
        require(event["shot_status"] != "accepted" or event["shot_family"] != "unknown", "Accepted shot class cannot be unknown")
        require(event["side"] in (None, "A", "B"), "Invalid event side")
        require(event["player"] is None or event["player"] in owners, "Hitter needs known side membership")
        if event["player"]:
            require(event["side"] == owners[event["player"]], "Hitter and side disagree")
        require(event["direction"] in ("unknown", "straight", "cross", "body", "middle"), "Invalid shot direction")
        require(event["execution"] in ("unknown", "continued", "net", "out", "miss", "body_fault", "service_fault"), "Invalid execution")
        require(event["kind"] != "contact" or event["execution"] != "miss", "A failed-contact miss must be an attempt")
        evidence(event["evidence"], frames)
        require(event["status"] != "accepted" or bool(event["evidence"]), "Accepted contact/attempt needs evidence")
        require(type(event["notes"]) is str, "Event notes must be text")
        events[event["id"]] = event
    by_rally = defaultdict(list)
    for event in events.values():
        by_rally[event["rally_id"]].append(event)
    for group in by_rally.values():
        accepted = sorted((e for e in group if e["kind"] == "contact" and e["status"] == "accepted"), key=lambda e:e["start_frame"])
        require(all(a["end_frame"] < b["start_frame"] for a,b in zip(accepted, accepted[1:])), "Accepted contacts have ambiguous temporal order")
    for outcome in document["outcomes"]:
        fields(outcome, "rally_id status winner terminal_cause terminal_status finish finish_status error_player " + " ".join(LINKS) + " setup_sequence_event_ids score_before score_after evidence notes", "outcome")
        rally_id = outcome["rally_id"]
        require(rally_id in rallies and rally_id not in outcomes, "Missing rally or duplicate outcome")
        rally = rallies[rally_id]
        require(all(outcome[key] in STATUSES for key in ("status", "terminal_status", "finish_status")), "Invalid outcome review status")
        require(outcome["winner"] in (None,"A","B") and outcome["terminal_cause"] in CAUSES and outcome["finish"] in FINISHES, "Invalid outcome labels")
        require(outcome["error_player"] is None or outcome["error_player"] in owners, "Error player needs side membership")
        evidence(outcome["evidence"], frames)
        require(type(outcome["notes"]) is str, "Outcome notes must be text")
        scoring = outcome["status"] == "accepted" and outcome["winner"] is not None
        if scoring:
            require(rally["kind"] == "play" and rally["status"] != "rejected" and rally["coverage"] not in ("partial_end","fragment") and bool(outcome["evidence"]), "Accepted point needs a scored rally with ending evidence")
        if rally["kind"] in ("let","replay","practice"):
            require(outcome["winner"] is None and outcome["finish"] in ("not_applicable","undetermined"), "Unscored rally cannot have a point winner")
        if outcome["terminal_cause"] == "let":
            require(outcome["winner"] is None, "A let awards no point")
        for key in ("score_before","score_after"):
            if outcome[key] is not None:
                pair(outcome[key], key)
                require(outcome["status"] == "accepted", "Score anchors require accepted outcome review")
        references = {}
        for key in LINKS:
            ref = outcome[key]
            require(ref is None or (ref in events and events[ref]["rally_id"] == rally_id and events[ref]["status"] != "rejected"), "Outcome link crosses rallies or refers to a missing/rejected event")
            references[key] = events.get(ref)
        contact = references["last_contact_event_id"]
        contacts = [e for e in by_rally[rally_id] if e["kind"] == "contact" and e["status"] != "rejected"]
        if contact:
            require(contact["kind"] == "contact" and contact == max(contacts,key=lambda e:e["end_frame"]), "Last-contact link is not the final contact")
        finish_accepted = outcome["finish_status"] == "accepted"
        if finish_accepted:
            require(scoring or outcome["finish"] == "not_applicable", "Accepted finish attribution needs an accepted point")
            require(bool(outcome["evidence"]), "Accepted finish judgment needs evidence")
        winner_event = references["direct_winner_event_id"]
        error_event = references["terminal_error_event_id"]
        pressure = references["preceding_pressure_event_id"]
        if winner_event:
            require(winner_event["kind"] == "contact" and winner_event["side"] == outcome["winner"] and outcome["finish"] == "direct_winner", "Direct-winner link is inconsistent")
        if error_event and outcome["winner"]:
            require(error_event["side"] is not None and error_event["side"] != outcome["winner"], "Terminal error must belong to losing side")
            if outcome["error_player"]:
                require(error_event["player"] == outcome["error_player"], "Error player conflicts with terminal event")
            later = [e for e in by_rally[rally_id] if e["status"] != "rejected" and e["start_frame"] > error_event["end_frame"]]
            require(not later, "Terminal error precedes later rally events")
        if outcome["error_player"] and outcome["winner"]:
            require(owners[outcome["error_player"]] != outcome["winner"], "Error cannot be blamed on winning side")
        if pressure:
            require(error_event is not None and pressure["kind"] == "contact" and pressure["side"] == outcome["winner"]
                    and pressure["end_frame"] < error_event["start_frame"], "Pressure link has wrong side or temporal order")
            preceding = [e for e in by_rally[rally_id] if e["kind"] == "contact" and e["status"] != "rejected" and e["end_frame"] < error_event["start_frame"]]
            require(pressure == max(preceding,key=lambda e:e["end_frame"]), "Pressure link must be the preceding observed shot")
        if finish_accepted and outcome["finish"] == "direct_winner":
            require(winner_event is not None and error_event is None and winner_event == contact, "Direct winner needs final-contact credit, not a second terminal error")
        if outcome["finish"] == "unforced_error":
            require(pressure is None, "Unforced error must not also claim a forced-pressure link")
        if finish_accepted and outcome["finish"] in ("forced_error","unforced_error"):
            require(document['analysis_rubric']['skill_context'].strip().casefold() not in ('unspecified','unknown'),'Accepted forced/unforced judgment needs an explicit skill/context rubric')
            require(error_event is not None and winner_event is None, "Accepted error attribution needs terminal error evidence")
            if outcome["finish"] == "forced_error":
                require(pressure is not None, "Accepted forced error needs a preceding pressure link")
        if finish_accepted:
            require(all(e is None or e["status"] == "accepted" for e in references.values()), "Accepted finish uses unaccepted event evidence")
        setup = outcome["setup_sequence_event_ids"]
        require(type(setup) is list and len(set(setup)) == len(setup), "Setup links must be unique")
        terminal = error_event or winner_event or contact
        for ref in setup:
            require(ref in events and events[ref]["rally_id"] == rally_id and terminal is not None
                    and events[ref]["end_frame"] < terminal["start_frame"], "Setup link crosses rally or follows the terminal event")
        outcomes[rally_id] = outcome
    ids = set()
    for item in document["score_events"]:
        fields(item, "id game frame side status reason evidence", "administrative score event")
        identifier(item["id"]);require(item["id"] not in ids and item["id"] not in events and item["id"] not in rallies, "Duplicate administrative event ID");ids.add(item["id"])
        integer(item["game"], 1, rules["best_of_games"] or 99, "score game")
        integer(item["frame"], 0, frames-1, "score-event frame")
        require(item["side"] in ("A","B") and item["status"] in STATUSES, "Invalid administrative score event")
        require(type(item["reason"]) is str and bool(item["reason"].strip()), "Administrative point needs a reason")
        evidence(item["evidence"], frames)
        require(item["status"] != "accepted" or bool(item["evidence"]), "Administrative point needs evidence")
    score_timeline(document)
    return document


def game_over(score, rules):
    if rules["profile_id"] == "unconfigured":
        return False
    return (max(score) >= rules["target_points"] and abs(score[0]-score[1]) >= rules["win_by"]) or (
        rules["cap_points"] is not None and max(score) >= rules["cap_points"])


def score_timeline(document):
    """Unknown points break score continuity; they are never counted as zero."""
    rallies = {r["id"]: r for r in document["rallies"]}
    outcomes = {o["rally_id"]:o for o in document["outcomes"]}
    actions = []
    for rally in document["rallies"]:
        if rally["kind"] == "play" and rally["status"] != "rejected":
            actions.append((rally["game"], rally["end_frame"], rally["id"], outcomes.get(rally["id"]), False))
        elif rally["status"] != "rejected" and (rally["kind"] == "let" or rally["status"] != "accepted"):
            actions.append((rally["game"], rally["end_frame"], rally["id"], outcomes.get(rally["id"]), False))
    for item in document["score_events"]:
        if item["status"] != "rejected":
            actions.append((item["game"],item["frame"],item["id"],item,True))
    actions.sort(key=lambda a:(a[0],a[1]))
    keys = [(a[0],a[1]) for a in actions]
    require(len(set(keys)) == len(keys), "Score events share a time; duplicate/ambiguous point awards must be resolved")
    states = {int(g):list(s) for g,s in document["match"]["initial_scores"].items()}
    timeline, games_finished = [], {}
    match_winner = None
    rules = document["match"]["rules"]
    for game, frame, name, outcome, admin in actions:
        require(match_winner is None, "A point/rally follows an already completed match")
        state = states.get(game)
        if game in games_finished:
            raise ValueError("A point/rally follows an already completed game")
        before_anchor = outcome.get("score_before") if outcome and not admin else None
        if before_anchor is not None:
            require(state is None or state == before_anchor, "Score-before conflicts with preceding accepted score")
            state = list(before_anchor)
        before = list(state) if state is not None else None
        winner = (outcome["side"] if outcome["status"] == "accepted" else None) if admin else (outcome["winner"] if outcome and outcome["status"] == "accepted" else None)
        no_point = not admin and rallies[name]["kind"] == "let" and (rallies[name]["status"] == "accepted" or (outcome is not None and outcome["status"] == "accepted"))
        if not no_point and winner is None:
            state = None
        elif winner:
            if state is not None:
                require(not game_over(state,rules), "A point follows a score that already ends the game")
                state[0 if winner == "A" else 1] += 1
        after_anchor = outcome.get("score_after") if outcome and not admin else None
        if after_anchor is not None:
            require(state is not None and state == after_anchor, "Score-after is inconsistent; add a score-before anchor if prior score is unknown")
        states[game] = state
        if state is not None and game_over(state,rules):
            games_finished[game] = "A" if state[0] > state[1] else "B"
            wins=Counter(games_finished.values())
            needed=(rules['best_of_games']//2+1) if rules['best_of_games'] else None
            if needed and max(wins.values())>=needed:match_winner=max(wins,key=wins.get)
        timeline.append({"game":game,"frame":frame,"event_id":name,"administrative":admin,"winner":winner,
                         "score_before":before,"score_after":deepcopy(state),"status":"known" if state is not None else "unknown"})
    return {"timeline":timeline,"latest_by_game":{str(g):s for g,s in states.items()},"completed_games":{str(g):s for g,s in games_finished.items()},
            "rules_profile":rules["profile_id"],"match_winner_from_annotated_games":match_winner,
            "service_sequence_status":"not_validated","reconciliation_status":"annotation_chain_only; recording completeness not established"}


def summarize(document, source, reviewer_kind):
    validate_document(document,source)
    families, finishes, errors = Counter(), Counter(), Counter()
    side_points, direct_winners, induced = Counter(), Counter(), Counter()
    rallies = {r["id"]:r for r in document["rallies"]}
    accepted = [e for e in document["events"] if e["status"] == "accepted" and rallies[e["rally_id"]]["kind"] != "replay" and rallies[e["rally_id"]]["status"] != "rejected"]
    classified = [e for e in accepted if e["shot_status"] == "accepted"]
    events = {e["id"]:e for e in document["events"]}
    player_reports = {p:{"accepted_contacts":0,"accepted_attempts":0,"shot_counts":{},"direct_winners":0,
                         "intended_attempt_counts":{},"forced_errors_committed":0,"unforced_errors_committed":0,"forced_errors_induced":0}
                      for members in document["match"]["teams"].values() for p in members}
    for event in accepted:
        if event["player"]:
            player = player_reports[event["player"]]
            player["accepted_contacts" if event["kind"] == "contact" else "accepted_attempts"] += 1
            if event["shot_status"] == "accepted":
                counter=player["shot_counts"] if event["kind"]=='contact' else player["intended_attempt_counts"]
                counter[event["shot_family"]] = counter.get(event["shot_family"],0)+1
    for event in classified:
        if event['kind']=='contact':families[event["shot_family"]] += 1
    played, won = Counter(), Counter()
    for outcome in document["outcomes"]:
        if outcome["status"] != "accepted" or outcome["winner"] is None:
            continue
        side_points[outcome["winner"]] += 1
        rally_events = [e for e in classified if e["rally_id"] == outcome["rally_id"]]
        for side,family in {(e["side"],e["shot_family"]) for e in rally_events if e["side"] is not None}:
            played[(side,family)] += 1
            won[(side,family)] += side == outcome["winner"]
        if outcome["finish_status"] != "accepted":
            continue
        finishes[outcome["finish"]] += 1
        direct = events.get(outcome["direct_winner_event_id"])
        error = events.get(outcome["terminal_error_event_id"])
        pressure = events.get(outcome["preceding_pressure_event_id"])
        if direct and direct in classified:
            direct_winners[direct["shot_family"]] += 1
        if direct and direct["player"]:
            player_reports[direct["player"]]["direct_winners"] += 1
        if error and error in classified:
            errors[error["shot_family"]] += 1
        if pressure and outcome["finish"] == "forced_error":
            induced[pressure["id"]] += 1
            if pressure["player"]:
                player_reports[pressure["player"]]["forced_errors_induced"] += 1
        if outcome["error_player"] and outcome["finish"] in ("forced_error","unforced_error"):
            key = "forced_errors_committed" if outcome["finish"] == "forced_error" else "unforced_errors_committed"
            player_reports[outcome["error_player"]][key] += 1
    denominators = Counter(e["shot_family"] for e in classified)
    contact_denominators = Counter(e["shot_family"] for e in classified if e["kind"] == "contact")
    def event_summary(ref):
        event=events.get(ref)
        if event is None or event['status']!='accepted':return None
        return {'event_id':ref,'player':event['player'],'side':event['side'],'kind':event['kind'],
                'shot_family':event['shot_family'] if event['shot_status']=='accepted' else 'unknown',
                'source_seconds':event['start_frame']/source['fps']}
    outcome_by_rally={o['rally_id']:o for o in document['outcomes']}
    rally_results=[]
    for rally in document['rallies']:
        outcome=outcome_by_rally.get(rally['id'],new_outcome(rally['id']))
        finish_accepted=outcome['finish_status']=='accepted'
        rally_results.append({'rally_id':rally['id'],'game':rally['game'],'kind':rally['kind'],'coverage':rally['coverage'],
            'point_status':outcome['status'],'point_winner':outcome['winner'] if outcome['status']=='accepted' else None,
            'finish':outcome['finish'] if finish_accepted else 'undetermined','error_player':outcome['error_player'] if finish_accepted else None,
            'last_contact':event_summary(outcome['last_contact_event_id']),
            'direct_winning_shot':event_summary(outcome['direct_winner_event_id']) if finish_accepted and outcome['finish']=='direct_winner' else None,
            'terminal_error':event_summary(outcome['terminal_error_event_id']) if finish_accepted and outcome['finish'] in ('forced_error','unforced_error') else None,
            'preceding_pressure_shot':event_summary(outcome['preceding_pressure_event_id']) if finish_accepted and outcome['finish']=='forced_error' else None,
            'start_seconds':rally['start_frame']/source['fps'],'end_seconds':rally['end_frame']/source['fps']})
    return {"schema_version":SCHEMA,"status":"reviewed_annotation_summary" if accepted else "no_accepted_event_annotations",
            "analysis_rubric":document['analysis_rubric'],
            "reviewer_kind":reviewer_kind,"labels_are_authenticated":False,
            "coverage":{"annotated_rallies":len(document["rallies"]),"accepted_events":len(accepted),
                        "proposed_events":sum(e["status"] == "proposed" for e in document["events"]),
                        "accepted_contacts":sum(e["kind"] == "contact" for e in accepted),
                        "accepted_attempts":sum(e["kind"] == "attempt" for e in accepted),
                        "accepted_shot_classes":len(classified),"events_without_accepted_class":len(accepted)-len(classified),
                        "rallies_without_confirmed_point":sum(r["kind"] == "play" and r["status"] != "rejected" and not any(
                            o["rally_id"] == r["id"] and o["status"] == "accepted" and o["winner"] for o in document["outcomes"]) for r in document["rallies"])},
            "shot_counts":dict(families),"accepted_rally_points":dict(side_points),"finish_counts":dict(finishes),
            "intended_attempt_counts":dict(Counter(e['shot_family'] for e in classified if e['kind']=='attempt')),
            "terminal_errors_without_accepted_shot_class":sum(o['status']=='accepted' and o['finish_status']=='accepted' and o['finish'] in ('forced_error','unforced_error')
                    and events.get(o['terminal_error_event_id']) not in classified for o in document['outcomes']),
            "player_reports":player_reports,
            "rally_results":rally_results,
            "unknown_error_responsibility":sum(o["status"] == "accepted" and o["finish_status"] == "accepted" and o["finish"] in ("forced_error","unforced_error") and o["error_player"] is None for o in document["outcomes"]),
            "terminal_error_rates":{f:{"numerator":errors[f],"denominator":n,"rate":errors[f]/n} for f,n in denominators.items()},
            "direct_winner_rates":{f:{"numerator":direct_winners[f],"denominator":n,"rate":direct_winners[f]/n} for f,n in contact_denominators.items()},
            "rally_win_rates_after_shot":[{"side":s,"shot_family":f,"numerator":won[(s,f)],"denominator":n,"rate":won[(s,f)]/n}
                                         for (s,f),n in sorted(played.items())],
            "induced_forced_error_event_ids":dict(induced),"score":score_timeline(document),
            "accepted_administrative_points":dict(Counter(e["side"] for e in document["score_events"] if e["status"] == "accepted")),
            "playlists":[{"event_id":e["id"],"rally_id":e["rally_id"],"shot_family":e["shot_family"],"player":e["player"],
                          "start_seconds":max(rallies[e["rally_id"]]["start_frame"],e["start_frame"]-round(source["fps"]))/source["fps"],
                          "end_seconds":min(rallies[e["rally_id"]]["end_frame"],e["end_frame"]+round(source["fps"]))/source["fps"]} for e in accepted],
            "coaching_metrics":{"status":"not_available"},
            "limitations":["Accepted labels are explicit reviewer annotations, not proof of automatic model accuracy.",
                           "Shot associations do not establish causation; no automatic forcedness inference.",
                           "Unknown points break score continuity; service order and physical measurements are not validated."]}


def _json_bytes(value):
    return (json.dumps(value,indent=2,allow_nan=False)+"\n").encode()


def publish(output, packet, preview, parent=None, patch=None):
    require(not output.exists(),"Choose a new analysis output folder")
    validate_document(packet["document"],packet["source"])
    report = summarize(packet["document"],packet["source"],packet["reviewer_kind"])
    encoded = _json_bytes(packet)
    data = {"packet":packet,"base_sha256":sha256(encoded).hexdigest(),"families":FAMILIES,"statuses":STATUSES,"finishes":FINISHES,"causes":CAUSES,"report":report,
            "templates":{"rally":new_rally('rally',0,0),"event":new_event('event','rally',0),"outcome":new_outcome('rally')}}
    payload=json.dumps(data,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    output.mkdir(parents=True)
    (output/'analysis.json').write_bytes(encoded)
    (output/'report.json').write_bytes(_json_bytes(report))
    # Immutable media may be shared across revisions on the same filesystem.
    # Fall back to a normal copy when hard links are unavailable.
    try: os.link(preview,output/'preview.mp4')
    except OSError: shutil.copyfile(preview,output/'preview.mp4')
    require(sha256_file(output/'preview.mp4')==packet['source']['preview_sha256'],'Preview changed while publishing analysis')
    (output/'index.html').write_text(template.replace('/*MATCH_DATA*/',payload),encoding='utf-8')
    if parent is not None:(output/'parent_analysis.json').write_bytes(parent)
    if patch is not None:(output/'corrections.json').write_bytes(patch)
    artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}
    (output/'analysis_manifest.json').write_bytes(_json_bytes({'schema_version':SCHEMA,'source_binding':packet['source_binding'],'artifacts':artifacts}))
    return output/'analysis.json'


def initial_packet(job_path):
    job,result,summary,rows,binding=source_snapshot(job_path)
    preview=_local(job_path,result['artifacts']['preview'])
    source={'fps':result['preview']['fps'],'frames':result['preview']['frames'],
            'expected_players':job['binding']['configuration']['expected_players'],'preview_sha256':sha256_file(preview),
            'scene_boundaries':[math.ceil(row['frame_index']/summary['sampling']['fps']*result['preview']['fps']-1e-8) for row in rows if row['scene_cut']]}
    render=job['stages']['render'];render_root=_local(job_path,render['directory'])
    require(source['preview_sha256']==render['artifacts'][str(preview.relative_to(render_root))],'Preview differs from completed job')
    packet={'schema_version':SCHEMA,'source_binding':binding,'source':source,'revision':0,'parent_sha256':None,
            'reviewer':None,'reviewer_kind':None,'created_utc':datetime.now(timezone.utc).isoformat(),
            'history':[],'document':empty_document()}
    require(source_snapshot(job_path)[4]==binding,'Source changed while preparing analysis')
    return packet,preview


def prepare(job_path,output):
    packet,preview=initial_packet(job_path)
    return publish(output,packet,preview)


def verify(job_path,folder):
    job,result,summary,rows,binding=source_snapshot(job_path)
    manifest=json.loads((folder/'analysis_manifest.json').read_text(encoding='utf-8'))
    require(manifest['schema_version']==SCHEMA and manifest['source_binding']==binding,'Analysis source is stale')
    require({'analysis.json','report.json','preview.mp4','index.html'} <= set(manifest['artifacts']),'Analysis manifest is incomplete')
    for name,expected in manifest['artifacts'].items():
        require(type(name) is str and Path(name).name==name and '/' not in name and '\\' not in name,'Invalid manifest artifact path')
        require(sha256_file(folder/name)==expected,'Analysis artifact modified: '+name)
    packet=json.loads((folder/'analysis.json').read_text(encoding='utf-8'))
    require(packet['schema_version']==SCHEMA and packet['source_binding']==binding,'Analysis packet source is stale')
    require(sha256_file(folder/'preview.mp4')==packet['source']['preview_sha256'],'Analysis preview differs')
    require(packet['source']['frames']==result['preview']['frames'] and packet['source']['fps']==result['preview']['fps']
            and packet['source']['expected_players']==job['binding']['configuration']['expected_players'],'Analysis geometry differs from source job')
    boundaries=[math.ceil(row['frame_index']/summary['sampling']['fps']*result['preview']['fps']-1e-8) for row in rows if row['scene_cut']]
    require(packet['source'].get('scene_boundaries',[])==boundaries,'Analysis camera boundaries differ from source job')
    validate_document(packet['document'],packet['source'])
    saved=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    require(saved==summarize(packet['document'],packet['source'],packet['reviewer_kind']),'Analysis summary is stale')
    return packet


def apply(job_path,base,patch_path,output):
    packet=verify(job_path,base)
    original=(base/'analysis.json').read_bytes();patch_bytes=patch_path.read_bytes()
    patch=json.loads(patch_bytes.decode('utf-8'))
    fields(patch,'schema_version base_sha256 reviewer reviewer_kind document','correction')
    require(patch['schema_version']==SCHEMA and patch['base_sha256']==sha256(original).hexdigest(),'Correction base is stale')
    require(type(patch['reviewer']) is str and bool(patch['reviewer'].strip()) and patch['reviewer_kind'] in ('human','automated'),'Explicit reviewer and reviewer kind required')
    validate_document(patch['document'],packet['source'])
    next_packet={**packet,'revision':packet['revision']+1,'parent_sha256':sha256(original).hexdigest(),
        'reviewer':patch['reviewer'].strip(),'reviewer_kind':patch['reviewer_kind'],
        'created_utc':datetime.now(timezone.utc).isoformat(),'document':patch['document'],
        'history':packet['history']+[{'revision':packet['revision'],'sha256':sha256(original).hexdigest(),
                                   'reviewer':packet['reviewer'],'reviewer_kind':packet['reviewer_kind']}]}
    require(source_snapshot(job_path)[4]==packet['source_binding'],'Source changed during correction')
    return publish(output,next_packet,base/'preview.mp4',parent=original,patch=patch_bytes)


def suggest_outcomes(document):
    """Propose the opposing side from an explicit terminal execution tag only.

    This is not an automated line/net detector. Never infer forcedness, a clean
    winner, or a score from disappearance, posture, or shot name.
    """
    document=deepcopy(document);suggestions=[]
    causes={'net':'net_failure','out':'ground_out','body_fault':'body_fault','service_fault':'service_fault'}
    existing={o['rally_id']:o for o in document['outcomes']}
    for rally in document['rallies']:
        if rally['kind']!='play' or rally['status']=='rejected' or rally['coverage'] in ('fragment','partial_end'):continue
        old=existing.get(rally['id'])
        if old and (old['status']!='unknown' or old['evidence'] or old['winner'] is not None):continue
        events=sorted((e for e in document['events'] if e['rally_id']==rally['id'] and e['status']!='rejected'),key=lambda e:e['end_frame'])
        if not events:continue
        last=events[-1]
        if last['status']!='accepted' or last['side'] is None or last['execution'] not in causes:continue
        outcome=new_outcome(rally['id']);outcome.update(status='proposed',winner='B' if last['side']=='A' else 'A',
            terminal_cause=causes[last['execution']],terminal_status='proposed',terminal_error_event_id=last['id'],
            notes='Derived from an explicitly annotated terminal execution. Verify that play ended here and no earlier fault changed the result.')
        contacts=[e for e in events if e['kind']=='contact']
        outcome['last_contact_event_id']=contacts[-1]['id'] if contacts else None
        outcome['evidence']=deepcopy(last['evidence'])
        document['outcomes']=[o for o in document['outcomes'] if o['rally_id']!=rally['id']]+[outcome]
        suggestions.append({'rally_id':rally['id'],'source_event_id':last['id'],'winner_proposal':outcome['winner'],
                            'finish_attribution':'undetermined; no forced/unforced inference'})
    return document,suggestions


def propose_outcomes(job_path,base,output):
    packet=deepcopy(verify(job_path,base));original=(base/'analysis.json').read_bytes()
    document,suggestions=suggest_outcomes(packet['document'])
    packet.update(document=document,revision=packet['revision']+1,parent_sha256=sha256(original).hexdigest(),
                  created_utc=datetime.now(timezone.utc).isoformat(),history=packet['history']+[{'revision':packet['revision'],
                      'sha256':sha256(original).hexdigest(),'reviewer':packet['reviewer'],'reviewer_kind':packet['reviewer_kind'],'operation':'terminal_outcome_suggestions'}])
    packet['outcome_suggestions']={'status':'proposed' if suggestions else 'no_eligible_terminal_annotations',
                                  'suggestions':suggestions,'accepted_points_added':0,'forcedness_inferred':False}
    require(source_snapshot(job_path)[4]==packet['source_binding'],'Source changed during outcome suggestions')
    return publish(output,packet,base/'preview.mp4',parent=original)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','apply','verify','suggest-outcomes'))
    parser.add_argument('--job',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--base',type=Path);parser.add_argument('--patch',type=Path)
    args=parser.parse_args()
    try:
        if args.command=='prepare':print(prepare(args.job,args.out))
        elif args.command=='verify':print(json.dumps({'status':'valid','revision':verify(args.job,args.out)['revision']}))
        elif args.command=='suggest-outcomes':
            require(args.base is not None,'--base is required')
            print(propose_outcomes(args.job,args.base,args.out))
        else:
            require(args.base is not None and args.patch is not None,'--base and --patch are required')
            print(apply(args.job,args.base,args.patch,args.out))
    except (ValueError,OSError,KeyError,TypeError) as exc:parser.exit(1,f'Analysis error: {exc}\n')


if __name__=='__main__':main()
