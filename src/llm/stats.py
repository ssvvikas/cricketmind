"""Plain counting statistics (career / season lines, leaderboards, team records, venues).

These are facts, not model outputs, so they use cricket's standard definitions:
  * batting average = runs / dismissals (None when never dismissed); an innings counts if the
    batter faced a ball or was dismissed; 50s are 50-99, 100s are 100+; highest score carries a
    not-out flag
  * bowling: only legal balls count towards overs; runs are bat runs + wides + no-balls (byes and
    leg-byes are not the bowler's); run-outs are not credited to the bowler
  * super overs are excluded (the loader keeps innings 1-2 only)

Every function takes a read-only DuckDB connection to the analytics database and returns a
JSON-friendly dict.  Filters: ``league`` (IPL/BBL/CPL/PSL/T20I) and ``season`` (as stored, e.g.
"2016" or "2023/24").
"""

from typing import Any

LEAGUES = ("IPL", "BBL", "CPL", "PSL", "T20I")
LEADERBOARD_STATS = ("runs", "wickets", "sixes", "fours", "strike_rate", "economy",
                     "highest_score", "batting_average")
COVERAGE = ("Covers only IPL, BBL, CPL, PSL and men's T20 internationals in the data "
            "(Afghanistan is absent; other leagues such as the Hundred or SA20 are not included).")
RAW = "Raw counting stats: not adjusted for era, phase or opposition."


class NoData(ValueError):
    """Raised when a filter matches nothing (message lists what is available)."""


def _where(league: str | None, seasons: list[str] | None, alias: str = "m") -> tuple[str, list]:
    """WHERE clause for a league and a list of resolved season labels."""
    clauses, params = [], []
    if league:
        clauses.append(f"{alias}.league = ?")
        params.append(league)
    if seasons:
        clauses.append(f"{alias}.season IN ({','.join('?' for _ in seasons)})")
        params += list(seasons)
    return (" AND ".join(clauses) if clauses else "TRUE"), params


def seasons_available(con, league: str | None = None) -> list[str]:
    q = "SELECT DISTINCT season FROM match_info" + (" WHERE league = ?" if league else "")
    rows = con.execute(q, [league] if league else []).fetchall()
    return sorted(r[0] for r in rows)


def resolve_seasons(con, league: str | None, season: str | None) -> list[str] | None:
    """Turn a user's season into stored labels.

    Cricsheet labels are inconsistent (IPL 2008 is '2007/08', IPL 2010 is '2009/10', IPL 2020
    is '2020/21', IPL 2009 is plain '2009').  An exact label always matches; a four-digit year
    also matches every edition that *concluded* in that year, so "IPL 2010" finds '2009/10'.
    """
    if not season:
        return None
    season = str(season).strip()
    rows = con.execute(
        "SELECT season, MAX(date) FROM match_info" + (" WHERE league = ?" if league else "")
        + " GROUP BY season", [league] if league else []).fetchall()
    labels = {r[0]: r[1] for r in rows}
    picks = {season} & set(labels)
    if season.isdigit() and len(season) == 4:
        picks |= {lab for lab, last in labels.items() if last.year == int(season)}
    if not picks:
        raise NoData(f"No matches for season {season!r}" + (f" in {league}" if league else "")
                     + f". Available seasons: {sorted(labels)}")
    return sorted(picks)


def _prepare(con, league, season) -> list[str] | None:
    """Validate the league and resolve the season (None when no season filter)."""
    if league and league not in LEAGUES:
        raise ValueError(f"league must be one of {LEAGUES}")
    return resolve_seasons(con, league, season)


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def _overs(balls: int) -> str:
    return f"{balls // 6}.{balls % 6}"


def _caveats(league: str | None, extra: list[str] | None = None) -> list[str]:
    out = [COVERAGE, RAW]
    if league in (None, "T20I"):
        out.append("T20I totals include matches against associate nations, so they are "
                   "inflated relative to full-member cricket.")
    return out + (extra or [])


# ── batting ─────────────────────────────────────────────────────────────
_BAT_INNINGS = """
    WITH faced AS (
        SELECT b.match_id, b.innings, b.batter AS player,
               SUM(b.runs_batter) AS runs, SUM(b.is_faced::INT) AS balls,
               SUM((b.runs_batter = 4)::INT) AS fours, SUM((b.runs_batter = 6)::INT) AS sixes
        FROM ball_by_ball b JOIN match_info m USING (match_id)
        WHERE b.is_faced AND {where}
        GROUP BY ALL
    ),
    outs AS (
        SELECT DISTINCT b.match_id, b.innings, b.player_out AS player
        FROM ball_by_ball b JOIN match_info m USING (match_id)
        WHERE b.is_wicket AND b.player_out IS NOT NULL AND {where}
    ),
    inn AS (
        SELECT COALESCE(f.match_id, o.match_id) AS match_id,
               COALESCE(f.innings, o.innings) AS innings,
               COALESCE(f.player, o.player) AS player,
               COALESCE(f.runs, 0) AS runs, COALESCE(f.balls, 0) AS balls,
               COALESCE(f.fours, 0) AS fours, COALESCE(f.sixes, 0) AS sixes,
               (o.player IS NOT NULL) AS out
        FROM faced f FULL OUTER JOIN outs o
          ON f.match_id = o.match_id AND f.innings = o.innings AND f.player = o.player
    )
"""


def _batting_innings(con, player: str, league, seasons) -> Any:
    where, params = _where(league, seasons)
    # season / league come from match_info (a dismissed-without-facing row has no ball row)
    return con.execute(
        _BAT_INNINGS.format(where=where) + """
        SELECT inn.*, mi.season, mi.league FROM inn JOIN match_info mi USING (match_id)
        WHERE player = ?""", params + params + [player]).df()


def _bat_line(df) -> dict:
    n = len(df)
    outs = int(df["out"].sum())
    runs, balls = int(df["runs"].sum()), int(df["balls"].sum())
    hs = df.sort_values(["runs", "out"], ascending=[False, True]).iloc[0]
    return {
        "innings": n, "runs": runs, "balls_faced": balls, "dismissals": outs,
        "not_outs": n - outs,
        "average": _r(runs / outs) if outs else None,
        "strike_rate": _r(runs * 100 / balls) if balls else None,
        "highest_score": f"{int(hs['runs'])}{'' if hs['out'] else '*'}",
        "fifties": int(((df["runs"] >= 50) & (df["runs"] < 100)).sum()),
        "hundreds": int((df["runs"] >= 100).sum()),
        "fours": int(df["fours"].sum()), "sixes": int(df["sixes"].sum()),
        "ducks": int(((df["runs"] == 0) & df["out"]).sum()),
    }


def batting_stats(con, player: str, league=None, season=None, breakdown=None) -> dict:
    seasons = _prepare(con, league, season)
    if breakdown not in (None, "season", "league"):
        raise ValueError("breakdown must be 'season', 'league' or omitted")
    df = _batting_innings(con, player, league, seasons)
    if df.empty:
        raise NoData(f"{player!r} has no batting innings for these filters")
    out: dict[str, Any] = {"player": player, "league": league or "all",
                           "season": seasons or "all", **_bat_line(df)}
    if breakdown:
        key = "season" if breakdown == "season" else "league"
        out["breakdown"] = [{key: k, **_bat_line(g)} for k, g in df.groupby(key, sort=True)]
    out["caveats"] = _caveats(league)
    return out


# ── bowling ─────────────────────────────────────────────────────────────
def _bowling_spells(con, player: str, league, seasons):
    where, params = _where(league, seasons)
    return con.execute(f"""
        SELECT b.match_id, m.season, m.league,
               SUM(b.is_legal::INT) AS balls, SUM(b.bowler_runs) AS runs,
               SUM(b.is_bowler_wicket::INT) AS wkts,
               SUM((b.is_legal AND b.runs_batter = 0 AND b.runs_extras = 0)::INT) AS dots
        FROM ball_by_ball b JOIN match_info m USING (match_id)
        WHERE b.bowler = ? AND {where}
        GROUP BY ALL""", [player] + params).df()


def _bowl_line(df) -> dict:
    balls, runs, wkts = int(df["balls"].sum()), int(df["runs"].sum()), int(df["wkts"].sum())
    best = df.sort_values(["wkts", "runs"], ascending=[False, True]).iloc[0]
    return {
        "matches": len(df), "overs": _overs(balls), "balls": balls, "runs_conceded": runs,
        "wickets": wkts,
        "economy": _r(runs * 6 / balls) if balls else None,
        "average": _r(runs / wkts) if wkts else None,
        "strike_rate": _r(balls / wkts) if wkts else None,
        "best_figures": f"{int(best['wkts'])}/{int(best['runs'])}",
        "three_wicket_hauls": int(((df["wkts"] >= 3) & (df["wkts"] < 5)).sum()),
        "five_wicket_hauls": int((df["wkts"] >= 5).sum()),
        "dot_ball_percent": _r(df["dots"].sum() * 100 / balls, 1) if balls else None,
    }


def bowling_stats(con, player: str, league=None, season=None, breakdown=None) -> dict:
    seasons = _prepare(con, league, season)
    if breakdown not in (None, "season", "league"):
        raise ValueError("breakdown must be 'season', 'league' or omitted")
    df = _bowling_spells(con, player, league, seasons)
    df = df[df["balls"] > 0]
    if df.empty:
        raise NoData(f"{player!r} has no bowling for these filters")
    out: dict[str, Any] = {"player": player, "league": league or "all",
                           "season": seasons or "all", **_bowl_line(df)}
    if breakdown:
        key = "season" if breakdown == "season" else "league"
        out["breakdown"] = [{key: k, **_bowl_line(g)} for k, g in df.groupby(key, sort=True)]
    out["caveats"] = _caveats(league, [
        "Wickets exclude run-outs; runs conceded exclude byes and leg-byes."])
    return out


# ── leaderboards ────────────────────────────────────────────────────────
def leaderboard(con, stat: str, league=None, season=None, top: int = 10,
                min_balls: int = 120) -> dict:
    """Top players for a stat.  Rate stats need ``min_balls`` (faced / bowled) to qualify."""
    if stat not in LEADERBOARD_STATS:
        raise ValueError(f"stat must be one of {LEADERBOARD_STATS}")
    seasons = _prepare(con, league, season)
    top = max(1, min(int(top), 50))
    where, params = _where(league, seasons)
    if stat in ("runs", "sixes", "fours", "strike_rate", "highest_score", "batting_average"):
        sql = _BAT_INNINGS.format(where=where)
        agg = {
            "runs": ("SUM(runs)", "DESC"), "sixes": ("SUM(sixes)", "DESC"),
            "fours": ("SUM(fours)", "DESC"),
            "strike_rate": ("SUM(runs) * 100.0 / NULLIF(SUM(balls), 0)", "DESC"),
            "batting_average": ("SUM(runs) * 1.0 / NULLIF(SUM(out::INT), 0)", "DESC"),
            "highest_score": ("MAX(runs)", "DESC"),
        }[stat]
        having = ""
        if stat == "strike_rate":
            having = f"HAVING SUM(balls) >= {int(min_balls)}"
        if stat == "batting_average":
            having = f"HAVING SUM(balls) >= {int(min_balls)} AND SUM(out::INT) > 0"
        rows = con.execute(f"""{sql}
            SELECT player, {agg[0]} AS value, COUNT(*) AS innings, SUM(runs) AS runs,
                   SUM(balls) AS balls
            FROM inn
            GROUP BY player {having} ORDER BY value {agg[1]}, runs DESC, player LIMIT {top}""",
                           params + params).df()
    else:
        agg = {"wickets": "SUM(is_bowler_wicket::INT)",
               "economy": "SUM(bowler_runs) * 6.0 / NULLIF(SUM(is_legal::INT), 0)"}[stat]
        having = f"HAVING SUM(is_legal::INT) >= {int(min_balls)}" if stat == "economy" else ""
        order = "ASC" if stat == "economy" else "DESC"
        rows = con.execute(f"""
            SELECT bowler AS player, {agg} AS value, COUNT(DISTINCT b.match_id) AS matches,
                   SUM(is_legal::INT) AS balls
            FROM ball_by_ball b JOIN match_info m USING (match_id)
            WHERE {where} GROUP BY bowler {having} ORDER BY value {order}, balls DESC, bowler
            LIMIT {top}""", params).df()
    if rows.empty:
        raise NoData("No players qualify for these filters")
    rows["value"] = rows["value"].astype(float).round(2)
    rows.insert(0, "rank", range(1, len(rows) + 1))
    extra = []
    if stat in ("strike_rate", "batting_average", "economy"):
        extra.append(f"Only players with at least {min_balls} balls "
                     f"({'bowled' if stat == 'economy' else 'faced'}) qualify.")
    return {"stat": stat, "league": league or "all", "season": seasons or "all",
            "rows": rows.to_dict("records"), "caveats": _caveats(league, extra)}


# ── teams ───────────────────────────────────────────────────────────────
def _team_rows(con, team, league, seasons, opponent=None):
    where, params = _where(league, seasons)
    opp = "AND (m.team1 = ? OR m.team2 = ?)" if opponent else ""
    p = [team, team] + params + ([opponent, opponent] if opponent else [])
    return con.execute(f"""
        SELECT m.match_id, m.date, m.league, m.season, m.winner, m.toss_winner,
               CASE WHEN m.team1 = ? THEN m.team2 ELSE m.team1 END AS opponent
        FROM match_info m
        WHERE (m.team1 = ? OR m.team2 = ?) AND {where} {opp}""", [team] + p).df()


def team_record(con, team: str, league=None, season=None, opponent=None) -> dict:
    """Win/loss record; with ``opponent`` it is the head-to-head between two teams."""
    seasons = _prepare(con, league, season)
    df = _team_rows(con, team, league, seasons, opponent)
    if df.empty:
        raise NoData(f"No matches for {team!r}" + (f" against {opponent!r}" if opponent else "")
                     + " with these filters")
    won = int((df["winner"] == team).sum())
    lost = int(((df["winner"].notna()) & (df["winner"] != team)).sum())
    nr = len(df) - won - lost
    # scores: first-innings totals when batting first vs chasing
    ids = ",".join(str(int(i)) for i in df["match_id"])
    inn = con.execute(f"""
        SELECT b.match_id, b.innings, b.batting_team,
               SUM(b.runs_batter + b.runs_extras) AS runs
        FROM ball_by_ball b WHERE b.match_id IN ({ids}) GROUP BY ALL""").df()
    mine = inn[inn["batting_team"] == team]
    first = mine[mine["innings"] == 1]
    chase = mine[mine["innings"] == 2]
    won_ids = set(df.loc[df["winner"] == team, "match_id"])
    won_first = sum(1 for mid in first["match_id"] if mid in won_ids)
    won_chase = sum(1 for mid in chase["match_id"] if mid in won_ids)
    out = {
        "team": team, "opponent": opponent or "all", "league": league or "all",
        "season": seasons or "all", "played": len(df), "won": won, "lost": lost,
        "no_result_or_tie": nr,
        "win_percent": _r(won * 100 / (won + lost), 1) if (won + lost) else None,
        "avg_first_innings_score_when_batting_first": _r(first["runs"].mean(), 1)
        if len(first) else None,
        "matches_batting_first": len(first), "won_batting_first": won_first,
        "batting_first_win_percent": _r(won_first * 100 / len(first), 1) if len(first) else None,
        "matches_chasing": len(chase), "won_chasing": won_chase,
        "chasing_win_percent": _r(won_chase * 100 / len(chase), 1) if len(chase) else None,
        "toss_wins": int((df["toss_winner"] == team).sum()),
    }
    out["caveats"] = _caveats(league, [
        "Ties, no-results and super-over decisions are not counted as wins or losses."])
    return out


def standings(con, league: str, season: str) -> dict:
    """Table for one league season: played / won / lost / no-result / win %."""
    if not league or not season:
        raise ValueError("standings needs both league and season")
    seasons = _prepare(con, league, season)
    ph = ",".join("?" for _ in seasons)
    df = con.execute(f"""
        SELECT team, COUNT(*) AS played,
               SUM((winner = team)::INT) AS won,
               SUM((winner IS NOT NULL AND winner <> team)::INT) AS lost,
               SUM((winner IS NULL)::INT) AS no_result_or_tie
        FROM (SELECT team1 AS team, winner FROM match_info
              WHERE league = ? AND season IN ({ph})
              UNION ALL
              SELECT team2, winner FROM match_info WHERE league = ? AND season IN ({ph}))
        GROUP BY team""", [league, *seasons, league, *seasons]).df()
    df["win_percent"] = (df["won"] * 100 / (df["won"] + df["lost"]).replace(0, float("nan"))
                         ).round(1)
    # team name last: tied sides must come out in the same order on every call
    df = df.sort_values(["won", "win_percent", "team"], ascending=[False, False, True]
                        ).reset_index(drop=True)
    df.insert(0, "position_by_wins", range(1, len(df) + 1))
    return {"league": league, "season": seasons, "rows": df.to_dict("records"),
            "caveats": _caveats(league, [
                "Ordered by wins, not the official table (no points, net run rate or "
                "playoff stage). The data does not say who won the final."])}


# ── venues ──────────────────────────────────────────────────────────────
def venue_stats(con, venue: str, league=None) -> dict:
    """Scoring and chasing record at venues whose name contains ``venue``."""
    if league and league not in LEAGUES:
        raise ValueError(f"league must be one of {LEAGUES}")
    where, params = _where(league, None)
    names = [r[0] for r in con.execute(
        f"SELECT DISTINCT venue FROM match_info m WHERE venue ILIKE ? AND {where} ORDER BY 1",
        [f"%{venue}%"] + params).fetchall()]
    if not names:
        raise NoData(f"No venue matching {venue!r}")
    ph = ",".join("?" for _ in names)
    inn = con.execute(f"""
        SELECT b.match_id, b.innings, SUM(b.runs_batter + b.runs_extras) AS runs,
               SUM(b.is_legal::INT) AS balls
        FROM ball_by_ball b JOIN match_info m USING (match_id)
        WHERE m.venue IN ({ph}) AND {where} GROUP BY ALL""", names + params).df()
    m = con.execute(f"""
        SELECT match_id, winner, team1, team2 FROM match_info m
        WHERE venue IN ({ph}) AND {where}""", names + params).df()
    first = inn[inn["innings"] == 1].set_index("match_id")["runs"]
    second = inn[inn["innings"] == 2].set_index("match_id")["runs"]
    first_team = con.execute(f"""
        SELECT DISTINCT b.match_id, b.batting_team FROM ball_by_ball b JOIN match_info m
        USING (match_id) WHERE b.innings = 1 AND m.venue IN ({ph}) AND {where}""",
                             names + params).df().set_index("match_id")["batting_team"]
    decided = m[m["winner"].notna() & m["match_id"].isin(first_team.index)]
    bat_first_wins = sum(1 for r in decided.itertuples() if r.winner == first_team[r.match_id])
    return {
        "venue_query": venue, "league": league or "all", "matched_venues": names,
        "matches": len(m),
        "avg_first_innings_score": _r(first.mean(), 1) if len(first) else None,
        "avg_second_innings_score": _r(second.mean(), 1) if len(second) else None,
        "highest_total": int(inn["runs"].max()),
        "lowest_total_full_innings": int(inn[inn["balls"] >= 114]["runs"].min())
        if (inn["balls"] >= 114).any() else None,
        "batting_first_win_percent": _r(bat_first_wins * 100 / len(decided), 1)
        if len(decided) else None,
        "chasing_win_percent": _r((len(decided) - bat_first_wins) * 100 / len(decided), 1)
        if len(decided) else None,
        "decided_matches": len(decided),
        "caveats": _caveats(league, [
            "All venue names containing the query are combined (see matched_venues); "
            "average totals ignore conditions, era and opposition."]),
    }
