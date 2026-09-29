"""Download Cricsheet JSON zips and convert them to the raw Parquet layout.

The output has the same columns as the TIGZIG ``ball_by_ball`` / ``match_info`` /
``match_players`` tables, saved as ``cricsheet_<table>.parquet`` next to them, so
``load_db`` can union both sources without any feature code changing.
"""

import json
import zipfile
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import CRICSHEET_LEAGUES, CRICSHEET_ZIP_URL, RAW_DIR
from src.ingestion.download import download_file

MAX_FIELDERS = 3

BALL_COLUMNS = {
    "match_id": "int64", "season": "string", "start_date": "datetime64[ns]",
    "venue": "string", "innings": "int64", "ball": "float64", "batting_team": "string",
    "bowling_team": "string", "striker": "string", "non_striker": "string",
    "bowler": "string", "runs_off_bat": "int64", "extras": "int64", "wides": "Int64",
    "noballs": "Int64", "byes": "Int64", "legbyes": "Int64", "penalty": "Int64",
    "wicket_type": "string", "player_dismissed": "string", "other_wicket_type": "string",
    "other_player_dismissed": "string", "match_type": "string", "over_no": "int64",
    "delivery_in_over": "int64", "actual_delivery": "float64", "non_boundary": "boolean",
    "fielder_1": "string", "fielder_2": "string", "fielder_3": "string",
    "gender": "string", "team_type": "string",
}


def _parse_match(match_id: int, data: dict) -> tuple[dict, list[dict], list[dict]]:
    """Convert one Cricsheet match into (match_info row, ball rows, player rows)."""
    info = data["info"]
    teams = info["teams"]
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    toss = info.get("toss", {})
    potm = info.get("player_of_match", [])
    start_date = pd.Timestamp(info["dates"][0])

    match_row = {
        "match_id": match_id, "match_type": info["match_type"], "team1": teams[0],
        "team2": teams[1], "gender": info["gender"], "season": str(info["season"]),
        "start_date": start_date, "venue": info.get("venue"), "city": info.get("city"),
        "event": info.get("event", {}).get("name"),
        "match_number": info.get("event", {}).get("match_number"),
        "toss_winner": toss.get("winner"), "toss_decision": toss.get("decision"),
        "winner": outcome.get("winner"), "winner_runs": by.get("runs"),
        "winner_wickets": by.get("wickets"),
        "player_of_match1": potm[0] if len(potm) > 0 else None,
        "player_of_match2": potm[1] if len(potm) > 1 else None,
        "outcome": outcome.get("result"), "method": outcome.get("method"),
        "team_type": info.get("team_type"), "balls_per_over": info.get("balls_per_over"),
        "overs": info.get("overs"),
    }

    ball_rows = []
    for innings_no, inn in enumerate(data.get("innings", []), start=1):
        if inn.get("super_over"):  # super overs are not part of normal T20 play
            continue
        batting = inn["team"]
        bowling = teams[1] if batting == teams[0] else teams[0]
        for over in inn.get("overs", []):
            for j, dl in enumerate(over["deliveries"], start=1):
                extras = dl.get("extras", {})
                runs = dl["runs"]
                wickets = dl.get("wickets", [])
                first = wickets[0] if wickets else {}
                second = wickets[1] if len(wickets) > 1 else {}
                fielders = [f.get("name") for f in first.get("fielders", [])][:MAX_FIELDERS]
                fielders += [None] * (MAX_FIELDERS - len(fielders))
                actual = dl.get("actual_delivery")
                actual = float(actual) if actual is not None else over["over"] + j / 10
                ball_rows.append({
                    "match_id": match_id, "season": match_row["season"],
                    "start_date": start_date, "venue": match_row["venue"],
                    "innings": innings_no, "ball": actual, "batting_team": batting,
                    "bowling_team": bowling, "striker": dl["batter"],
                    "non_striker": dl["non_striker"], "bowler": dl["bowler"],
                    "runs_off_bat": runs["batter"], "extras": runs["extras"],
                    "wides": extras.get("wides"), "noballs": extras.get("noballs"),
                    "byes": extras.get("byes"), "legbyes": extras.get("legbyes"),
                    "penalty": extras.get("penalty"),
                    "wicket_type": first.get("kind"),
                    "player_dismissed": first.get("player_out"),
                    "other_wicket_type": second.get("kind"),
                    "other_player_dismissed": second.get("player_out"),
                    "match_type": info["match_type"], "over_no": over["over"],
                    "delivery_in_over": j, "actual_delivery": actual,
                    "non_boundary": bool(runs.get("non_boundary", False)),
                    "fielder_1": fielders[0], "fielder_2": fielders[1],
                    "fielder_3": fielders[2], "gender": info["gender"],
                    "team_type": info.get("team_type"),
                })

    registry = info.get("registry", {}).get("people", {})
    player_rows = [
        {"match_id": match_id, "person_name": name, "identifier": registry.get(name),
         "team": team, "role": "player"}
        for team, names in info.get("players", {}).items()
        for name in names
    ]
    return match_row, ball_rows, player_rows


def parse_zip(zip_path: Path) -> dict[str, pd.DataFrame]:
    """Parse every match JSON in a Cricsheet zip into raw-layout DataFrames."""
    matches, balls, players = [], [], []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            m, b, p = _parse_match(int(Path(name).stem), json.loads(z.read(name)))
            matches.append(m)
            balls += b
            players += p

    ball_df = pd.DataFrame(balls, columns=list(BALL_COLUMNS)).astype(BALL_COLUMNS)
    return {
        "match_info": pd.DataFrame(matches),
        "ball_by_ball": ball_df,
        "match_players": pd.DataFrame(players),
    }


def ingest_cricsheet(
    leagues: list[str] | None = None, force: bool = False, raw_dir: Path | None = None
) -> dict[str, Path]:
    """Download the league zips and write ``cricsheet_<table>.parquet`` files."""
    raw = Path(raw_dir or RAW_DIR)
    zip_dir = raw / "cricsheet"
    outputs = {t: raw / f"cricsheet_{t}.parquet" for t in ("match_info", "ball_by_ball", "match_players")}

    if not force and all(p.exists() for p in outputs.values()):
        logger.info("Skipping Cricsheet JSON ingest - parquet already exists")
        return outputs

    frames: dict[str, list[pd.DataFrame]] = {t: [] for t in outputs}
    for key in leagues or CRICSHEET_LEAGUES:
        dest = zip_dir / f"{key}_json.zip"
        if force or not dest.exists():
            logger.info(f"Downloading Cricsheet {key}")
            download_file(CRICSHEET_ZIP_URL.format(key=key), dest)
        parsed = parse_zip(dest)
        logger.info(
            f"  {key}: {len(parsed['match_info']):,} matches, "
            f"{len(parsed['ball_by_ball']):,} deliveries"
        )
        for table, df in parsed.items():
            frames[table].append(df)

    for table, path in outputs.items():
        pd.concat(frames[table], ignore_index=True).to_parquet(path, index=False)
    logger.success(f"Cricsheet JSON converted to {raw}")
    return outputs


if __name__ == "__main__":
    ingest_cricsheet()
