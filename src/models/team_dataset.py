"""Leak-free training data for the team-strength model.

Matches are processed strictly in chronological order.  For every match the
features are read from a running ``TeamState`` *before* the result is applied,
so a row never contains information from its own match or any later one.

Each decided match yields two rows - one from each team's point of view - with
"A minus B" difference features.  Averaging the two predictions at inference
time makes the model symmetric, so the arbitrary team1/team2 order in the source
data cannot bias it.

All matches on the same date are scored *before* any of that date's results are
applied, so a row can never see a same-day match (double-headers).

Lineup features rate the actual playing XI from each player's earlier matches
(pooled across every league, so ratings follow players between teams).
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

from src.config import DB_PATH

ELO_START = 1500.0
ELO_K = 24.0
FORM_WINDOW = 10
LEAGUES = ["IPL", "BBL", "CPL", "PSL", "T20I"]

FEATURES_TEAM = [
    "elo_diff",
    "form_diff",
    "bat_rr_diff",
    "bowl_econ_diff",
    "h2h_rate",
    "h2h_n",
    "venue_rate_diff",
    "exp_diff",
    "min_matches",
    "toss_won",
    "league_code",
]
LINEUP_FEATURES = [
    "lineup_bat_diff",  # runs above par per innings from the XI's batting
    "lineup_bowl_diff",  # runs saved per innings from the XI's bowling
    "lineup_wkt_diff",  # wickets above par per innings from the XI's bowling
    "lineup_net_diff",  # batting + bowling runs
    "xi_known_min",  # min over both sides of XI players with >= KNOWN_MATCHES history
]
FEATURES = FEATURES_TEAM + LINEUP_FEATURES

# player-rating shrinkage: pseudo-observations pulling small samples to the average
BAT_PRIOR_BALLS = 120
BOWL_PRIOR_BALLS = 120
APPEARANCE_PRIOR = 4  # matches
KNOWN_MATCHES = 10
# league averages used until enough data has accumulated to estimate them
DEFAULT_AVGS = {"sr": 1.28, "bpm": 10.5, "rpb": 1.30, "wr": 0.045, "bowlpm": 10.5}
PAR_WINDOW_MATCHES = 400  # era par = the pool's most recent matches
MIN_WINDOW_MATCHES = 30


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B under the Elo model."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


# module-level factories (not lambdas) so TeamState can be pickled
def _elo_start() -> float:
    return ELO_START


def _window() -> deque:
    return deque(maxlen=FORM_WINDOW)


def _pair() -> list:
    return [0, 0]


def rating_pool(league: str) -> str:
    """Club leagues and internationals are rated separately: associate-nation bowlers
    and batters look elite against weak opposition and would distort club ratings."""
    return "intl" if league == "T20I" else "club"


class PlayerState:
    """Running per-player batting/bowling record for one rating pool.

    Every performance is credited *against the par of its era* (a rolling window of the
    pool's most recent matches), so rising scoring over the years doesn't make later
    players look better than earlier ones.  Each record is
    [matches, bat_balls, bowl_balls, bat_above, bowl_above, wkt_above] where
      bat_above  = sum(runs - balls_faced * par_runs_per_ball)
      bowl_above = sum(balls_bowled * par_runs_per_ball - runs_conceded)
      wkt_above  = sum(wickets - balls_bowled * par_wickets_per_ball)
    plus raw bat/bowl ball counts for shrinkage and workload estimates.
    """

    def __init__(self) -> None:
        self.p: dict[str, list] = {}
        self.window: deque = deque(maxlen=PAR_WINDOW_MATCHES)
        self.win = [0, 0, 0, 0, 0, 0]  # players, bat_runs, bat_balls, bowl_runs, bowl_balls, wkts

    def averages(self) -> dict:
        n, bruns, bballs, wruns, wballs, wkts = self.win
        if len(self.window) < MIN_WINDOW_MATCHES or bballs == 0 or wballs == 0:
            return dict(DEFAULT_AVGS)
        return {"sr": bruns / bballs, "bpm": bballs / n, "rpb": wruns / wballs,
                "wr": wkts / wballs, "bowlpm": wballs / n}

    def add_match(self, player_stats: dict) -> None:
        """Credit one finished match: {name: [bat_runs, bat_balls, bowl_balls, bowl_runs, wkts]}."""
        av = self.averages()  # par as of BEFORE this match
        totals = [len(player_stats), 0, 0, 0, 0, 0]
        for name, (bruns, bballs, wballs, wruns, wkts) in player_stats.items():
            rec = self.p.setdefault(name, [0, 0, 0, 0.0, 0.0, 0.0])
            rec[0] += 1
            rec[1] += bballs
            rec[2] += wballs
            rec[3] += bruns - bballs * av["sr"]
            rec[4] += wballs * av["rpb"] - wruns
            rec[5] += wkts - wballs * av["wr"]
            for i, v in ((1, bruns), (2, bballs), (3, wruns), (4, wballs), (5, wkts)):
                totals[i] += v
        if len(self.window) == self.window.maxlen:
            old = self.window[0]
            self.win = [w - o for w, o in zip(self.win, old, strict=True)]
        self.window.append(totals)
        self.win = [w + t for w, t in zip(self.win, totals, strict=True)]

    def strength(self, xi: list[str]) -> dict:
        """Sum of per-player contributions over the XI (unknown players contribute 0)."""
        av = self.averages()
        bat = bowl = wkt = 0.0
        known = 0
        for name in xi:
            rec = self.p.get(name)
            if rec is None:
                continue
            n, bballs, wballs, bat_above, bowl_above, wkt_above = rec
            known += n >= KNOWN_MATCHES
            bpm = (bballs + APPEARANCE_PRIOR * av["bpm"]) / (n + APPEARANCE_PRIOR)
            bat += bpm * bat_above / (bballs + BAT_PRIOR_BALLS)
            bowlpm = (wballs + APPEARANCE_PRIOR * av["bowlpm"]) / (n + APPEARANCE_PRIOR)
            bowl += bowlpm * bowl_above / (wballs + BOWL_PRIOR_BALLS)
            wkt += bowlpm * wkt_above / (wballs + BOWL_PRIOR_BALLS)
        return {"bat": bat, "bowl": bowl, "wkt": wkt, "known": known}


@dataclass
class TeamState:
    """Running per-team history.  Keys are (league, team) so leagues never mix."""

    elo: dict = field(default_factory=lambda: defaultdict(_elo_start))
    played: dict = field(default_factory=lambda: defaultdict(int))
    results: dict = field(default_factory=lambda: defaultdict(_window))
    bat_rr: dict = field(default_factory=lambda: defaultdict(_window))
    bowl_econ: dict = field(default_factory=lambda: defaultdict(_window))
    h2h: dict = field(default_factory=lambda: defaultdict(_pair))  # sorted pair -> wins
    venue: dict = field(default_factory=lambda: defaultdict(_pair))  # (team, venue)
    players: dict = field(default_factory=lambda: {"club": PlayerState(), "intl": PlayerState()})
    last_xi: dict = field(default_factory=dict)  # (league, team) -> most recent playing XI

    # ── read ───────────────────────────────────────────────────────
    def _form(self, key) -> float:
        r = self.results[key]
        return (sum(r) + 1) / (len(r) + 2)  # Laplace-smoothed win rate

    @staticmethod
    def _mean(dq) -> float:
        return float(np.mean(dq)) if dq else np.nan

    def _venue_rate(self, team, venue) -> float:
        wins, n = self.venue[(team, venue)]
        return (wins + 1) / (n + 2)

    def _h2h(self, league, a, b) -> tuple[float, int]:
        pair = tuple(sorted((a, b)))
        wins = self.h2h[(league, pair)]
        a_wins = wins[0] if pair[0] == a else wins[1]
        n = wins[0] + wins[1]
        return (a_wins + 1) / (n + 2), n

    def features(self, league, a, b, venue, toss_winner, xi_a=None, xi_b=None) -> dict:
        """Features for the match from A's point of view, using history only.

        ``xi_a`` / ``xi_b`` are the announced playing XIs (lists of names); when either
        is missing the lineup features are NaN.
        """
        ka, kb = (league, a), (league, b)
        h2h_rate, h2h_n = self._h2h(league, a, b)
        lineup = dict.fromkeys(LINEUP_FEATURES, np.nan)
        if xi_a and xi_b:
            pool = self.players[rating_pool(league)]
            sa, sb = pool.strength(xi_a), pool.strength(xi_b)
            lineup = {
                "lineup_bat_diff": sa["bat"] - sb["bat"],
                "lineup_bowl_diff": sa["bowl"] - sb["bowl"],
                "lineup_wkt_diff": sa["wkt"] - sb["wkt"],
                "lineup_net_diff": (sa["bat"] + sa["bowl"]) - (sb["bat"] + sb["bowl"]),
                "xi_known_min": min(sa["known"], sb["known"]),
            }
        return {
            **lineup,
            "elo_diff": self.elo[ka] - self.elo[kb],
            "form_diff": self._form(ka) - self._form(kb),
            "bat_rr_diff": self._mean(self.bat_rr[ka]) - self._mean(self.bat_rr[kb]),
            # positive = A concedes fewer runs per over than B
            "bowl_econ_diff": self._mean(self.bowl_econ[kb]) - self._mean(self.bowl_econ[ka]),
            "h2h_rate": h2h_rate,
            "h2h_n": h2h_n,
            "venue_rate_diff": self._venue_rate(ka, venue) - self._venue_rate(kb, venue),
            "exp_diff": np.log1p(self.played[ka]) - np.log1p(self.played[kb]),
            "min_matches": min(self.played[ka], self.played[kb]),
            "toss_won": 1.0 if toss_winner == a else (0.0 if toss_winner == b else 0.5),
            "league_code": LEAGUES.index(league) if league in LEAGUES else len(LEAGUES),
        }

    # ── write (call only AFTER the row for this match has been emitted) ──
    def update(self, league, t1, t2, winner, venue, stats: dict, player_stats=None,
               xis: dict | None = None) -> None:
        """Apply a finished match.  ``player_stats`` maps name -> [bat_runs, bat_balls,
        bowl_balls, bowl_runs, bowl_wkts] for everyone who took part."""
        if player_stats:
            self.players[rating_pool(league)].add_match(player_stats)
        for team, xi in (xis or {}).items():
            if xi:
                self.last_xi[(league, team)] = list(xi)
        k1, k2 = (league, t1), (league, t2)
        e1 = elo_expected(self.elo[k1], self.elo[k2])
        s1 = 1.0 if winner == t1 else 0.0
        self.elo[k1] += ELO_K * (s1 - e1)
        self.elo[k2] += ELO_K * ((1 - s1) - (1 - e1))

        for k, won in ((k1, s1), (k2, 1 - s1)):
            self.played[k] += 1
            self.results[k].append(won)
        pair = tuple(sorted((t1, t2)))
        self.h2h[(league, pair)][0 if winner == pair[0] else 1] += 1
        for k, team, won in ((k1, t1, s1), (k2, t2, 1 - s1)):
            v = self.venue[(k, venue)]
            v[0] += int(won)
            v[1] += 1

        for team, key in ((t1, k1), (t2, k2)):
            s = stats.get(team)
            if s is not None:
                self.bat_rr[key].append(s["bat_rr"])
                self.bowl_econ[key].append(s["bowl_econ"])


def load_matches(db_path: str | None = None):
    """Decided matches in date order plus everything needed to build features.

    Returns (matches, team_stats, lineups, player_stats) where
      team_stats[match_id][team]      = {"bat_rr", "bowl_econ"}
      lineups[(match_id, team)]       = announced XI (list of names)
      player_stats[(match_id, team)]  = {name: [bat_runs, bat_balls, bowl_balls,
                                                bowl_runs, bowl_wkts]} for all who took part
    """
    con = duckdb.connect(db_path or str(DB_PATH), read_only=True)
    matches = con.execute("""
        SELECT match_id, date, league, venue, team1, team2, winner, toss_winner
        FROM match_info
        WHERE winner IS NOT NULL
        ORDER BY date, match_id
    """).df()
    bat = con.execute("""
        SELECT match_id, batting_team AS team,
               SUM(runs_batter + runs_extras) AS runs, SUM(is_legal::INT) AS balls
        FROM ball_by_ball GROUP BY match_id, batting_team
    """).df()
    batters = con.execute("""
        SELECT match_id, batting_team AS team, batter AS name,
               SUM(runs_batter) AS runs, SUM(is_faced::INT) AS balls
        FROM ball_by_ball GROUP BY ALL
    """).fetchall()
    bowlers = con.execute("""
        SELECT match_id, bowling_team AS team, bowler AS name,
               SUM(is_legal::INT) AS balls, SUM(bowler_runs) AS runs,
               SUM(is_bowler_wicket::INT) AS wkts
        FROM ball_by_ball GROUP BY ALL
    """).fetchall()
    xi_rows = con.execute("""
        SELECT match_id, team, person_name FROM match_players WHERE role = 'player'
    """).fetchall()
    con.close()

    bat["rr"] = bat["runs"] * 6.0 / bat["balls"].where(bat["balls"] > 0)
    rr = {(r.match_id, r.team): r.rr for r in bat.itertuples()}
    stats: dict = {}
    for m in matches.itertuples():
        per_team = {}
        for team, opp in ((m.team1, m.team2), (m.team2, m.team1)):
            own, conceded = rr.get((m.match_id, team)), rr.get((m.match_id, opp))
            if own is not None and conceded is not None and not (
                np.isnan(own) or np.isnan(conceded)
            ):
                per_team[team] = {"bat_rr": own, "bowl_econ": conceded}
        stats[m.match_id] = per_team

    lineups: dict = defaultdict(list)
    for match_id, team, name in xi_rows:
        lineups[(match_id, team)].append(name)

    # everyone who was in the XI, batted or bowled counts as having played the match
    player_stats: dict = defaultdict(dict)
    for key, names in lineups.items():
        for name in names:
            player_stats[key].setdefault(name, [0, 0, 0, 0, 0])
    for match_id, team, name, runs, balls in batters:
        rec = player_stats[(match_id, team)].setdefault(name, [0, 0, 0, 0, 0])
        rec[0], rec[1] = int(runs), int(balls)
    for match_id, team, name, balls, runs, wkts in bowlers:
        rec = player_stats[(match_id, team)].setdefault(name, [0, 0, 0, 0, 0])
        rec[2], rec[3], rec[4] = int(balls), int(runs), int(wkts)
    return matches, stats, dict(lineups), dict(player_stats)


def build_dataset(db_path: str | None = None) -> tuple[pd.DataFrame, TeamState]:
    """Return the two-rows-per-match feature table and the final TeamState."""
    matches, stats, lineups, player_stats = load_matches(db_path)
    state = TeamState()
    rows = []
    for _, day in matches.groupby("date", sort=True):
        # 1) emit every row for this date from the state as of the START of the date
        for m in day.itertuples():
            for a, b in ((m.team1, m.team2), (m.team2, m.team1)):
                feats = state.features(
                    m.league, a, b, m.venue, m.toss_winner,
                    lineups.get((m.match_id, a)), lineups.get((m.match_id, b)),
                )
                rows.append({
                    "match_id": m.match_id, "date": m.date, "league": m.league,
                    "team_a": a, "team_b": b, "team1_is_a": a == m.team1,
                    "y": int(m.winner == a), **feats,
                })
        # 2) only now may this date's results influence later dates
        for m in day.itertuples():
            pstats = {**player_stats.get((m.match_id, m.team1), {}),
                      **player_stats.get((m.match_id, m.team2), {})}
            xis = {t: lineups.get((m.match_id, t)) for t in (m.team1, m.team2)}
            state.update(m.league, m.team1, m.team2, m.winner, m.venue,
                         stats[m.match_id], pstats, xis)

    df = pd.DataFrame(rows)
    logger.info(f"Team dataset: {df.match_id.nunique():,} matches, {len(df):,} rows")
    return df, state
