# Models and backtests

What was tested, how, and what it showed, including the ideas that did not survive. Back to the [README](../README.md).

## Team-strength model

`python -m src.models.team_strength` trains an XGBoost classifier for P(team A beats team B)
and writes the model, state, calibration and a metrics/SHAP report to `data/processed/models/`.

**Features** (all computed only from earlier matches; all matches on one date are scored
before any of that date's results are applied):
- Team: Elo per league, recent form and scoring rate, head-to-head, venue record, toss
- **Lineup:** the announced XI is rated from each player's earlier matches: batting runs above
  par, bowling runs saved and wickets above par, weighted by workload and shrunk toward average
  for small samples. Ratings are measured against the par of their era (rolling 400-match
  window) and kept in separate club / international pools, because associate-nation players
  look elite against weak opposition
- Each match is scored from both teams' viewpoints and averaged, so team order can't bias it

**Evaluation:** expanding window, train on everything before year Y and test on Y for
2021-2026, pooled, with bootstrap 95% intervals. Leakage tests flip a result (or edit a match's
ball data) and check no earlier feature changes; they were verified to fail on a deliberate leak.

| Rolling test 2021-26 | Matches | Best model | Accuracy [95% CI] | Log loss (coin flip 0.693) |
|---|---:|---|---|---|
| T20I | 2,587 | XGBoost, team + lineup | 70.1% [68.4, 71.8] | 0.563 (Elo alone: 0.636) |
| Club leagues | 1,105 | any | 50-53% [±3] | 0.69-0.70 (no better than a coin flip) |

**Honest reading**
- **Internationals:** useful. Lineup features improve on team-only features (log loss 0.572 to
  0.563). A plain logistic regression does about as well as XGBoost
- **Club leagues (IPL, BBL, CPL, PSL): no usable win prediction.** The lineup ratings are real:
  a team's rating edge predicts the runs it scores with a slope of about 0.6 runs per point
  (after removing league-year trends). But that edge moves the predicted match margin by only
  ~4 runs against a match-to-match spread of ~43 runs, capping accuracy near 53%. Single T20
  matches between club sides are mostly luck. Reported club probabilities are therefore
  shrunk toward 50% (factor 0.15, fitted out of sample), so they stay honest
- Team history is a poor guide in club cricket because squads turn over (only ~36% of an IPL
  squad returns the next season; season-to-season win% correlation is 0.09)

```python
from src.models.team_strength import predict_match
predict_match("India", "Australia", "T20I")
predict_match("Mumbai Indians", "Chennai Super Kings", "IPL", xi_a=[...])  # optional XI
```

## Player ratings, rankings and matchups

`src/models/player_ratings.py` (`python -m src.models.player_ratings`) rates every batter and
bowler and answers matchup questions. Each ball is compared with the par for its
(pool, year, phase), so a powerplay ball in 2025 isn't scored like a death-over ball in 2012.
A rating is the shrunk (K = 150 balls), recency-weighted (2-year half-life) average of runs above
par. Three pools keep contexts apart: `club` leagues, `intl` (two ICC Full Members) and `assoc`
(any match involving an associate nation, where players look elite against weak opposition).

```python
from src.models.player_ratings import RatingBook, load_balls
book = RatingBook(load_balls())
book.rank_batters("club", top=10)
book.recommend_bowlers("V Kohli", ["JJ Bumrah", "SP Narine"], pool="club", phase="powerplay")
```

**Backtests** (rate using data before a cut date, score on the next year; 2019-2025 cuts;
Spearman rank correlation with next-year above-par performance):

| Question | Result |
|---|---|
| Batting, club | New rating **0.30** vs Phase 1 composite 0.20 vs raw strike rate 0.23. Raw batting *average*: -0.03 (useless) |
| Batting, Full-Member internationals | 0.34, no better than raw strike rate (0.36); small sample |
| Bowling: runs saved | Predictable (0.25-0.41), but raw economy is as good as the adjusted rating |
| Bowling: wickets | Barely persists (0.06-0.11): don't rank bowlers by wickets |
| Partnerships | Phase 1 chemistry score **0.03** (club) / 0.04 (intl). Individual ratings of the two batters: **0.22 / 0.30**. Adding the pair's own history made it worse at every shrinkage tried |
| Fielding | Phase 1 fielding score 0.40 vs plain dismissals per match played **0.59**. It also divided by matches *with a dismissal* (~3x too high). Catching persists (~0.5 even excluding keepers) but reflects role and opportunity, not measurable skill |
| Matchups on future balls | Additive ratings beat par (+0.7% MSE); **adding head-to-head history made it worse** at every shrinkage tried, and raw head-to-head (Phase 1's approach) was 4.8% worse than par |

So the adjusted ratings mainly buy comparability across eras and phases, not better rankings than
raw strike rate or economy. The matchup model therefore uses no head-to-head history; raw
head-to-head is exposed only as a fact (`head_to_head`). The Phase 1 `rank_score` (now
`legacy_score`), `vulnerability_score`, `chemistry_score` and `fielding_score` were removed or
demoted because of these results.
