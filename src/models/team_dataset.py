"""Leak-free training data for the team-strength model.

Matches are processed strictly in chronological order.  For every match the
features are read from a running ``TeamState`` *before* the result is applied,
so a row never contains information from its own match or any later one.

Each decided match yields two rows - one from each team's point of view - with
"A minus B" difference features.  Averaging the two predictions at inference
time makes the model symmetric, so the arbitrary team1/team2 order in the source
data cannot bias it.
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

FEATURES = [
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

    def features(self, league, a, b, venue, toss_winner) -> dict:
        """Features for the match from A's point of view, using history only."""
        ka, kb = (league, a), (league, b)
        h2h_rate, h2h_n = self._h2h(league, a, b)
        return {
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
    def update(self, league, t1, t2, winner, venue, stats: dict) -> None:
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


def load_matches(db_path: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Decided matches in chronological order + per-match team scoring stats."""
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
    return matches, stats


def build_dataset(db_path: str | None = None) -> tuple[pd.DataFrame, TeamState]:
    """Return the two-rows-per-match feature table and the final TeamState."""
    matches, stats = load_matches(db_path)
    state = TeamState()
    rows = []
    for m in matches.itertuples():
        for a, b in ((m.team1, m.team2), (m.team2, m.team1)):
            feats = state.features(m.league, a, b, m.venue, m.toss_winner)
            rows.append({
                "match_id": m.match_id, "date": m.date, "league": m.league,
                "team_a": a, "team_b": b, "team1_is_a": a == m.team1,
                "y": int(m.winner == a), **feats,
            })
        # only now is the result allowed to influence future rows
        state.update(m.league, m.team1, m.team2, m.winner, m.venue, stats[m.match_id])

    df = pd.DataFrame(rows)
    logger.info(f"Team dataset: {df.match_id.nunique():,} matches, {len(df):,} rows")
    return df, state
