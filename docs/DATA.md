# Data

Where the numbers come from, what is in the database, and its known gaps. Back to the [README](../README.md).

## Data

Built on [Cricsheet](https://cricsheet.org/) ball-by-ball data, from two sources:

- **TIGZIG Parquet export** of Cricsheet: IPL + men's T20 internationals
- **Direct Cricsheet JSON** (`src/ingestion/cricsheet_json.py`): BBL, CPL, PSL, converted to the same raw layout. The parser was cross-checked against TIGZIG on all 1,243 IPL matches (deliveries, runs, extras, legal balls, wickets, fielders: zero mismatches)

Filtered to men's T20 cricket, innings 1-2 only (super overs are dropped):

| League | Matches | Seasons |
|--------|--------:|---------|
| T20I | 3,558 | 2005-2026 |
| IPL | 1,243 | 2008-2026 |
| BBL | 662 | 2011-2026 |
| CPL | 442 | 2013-2026 |
| PSL | 357 | 2016-2026 |

6,262 matches, ~1.44M deliveries, stored in DuckDB.

**Known limitations**
- Player names are strings, not IDs, so two players sharing a name are merged (Cricsheet's JSON has a name→ID registry that could fix this)
- Renamed franchises appear as separate teams (e.g. Royal Challengers Bangalore/Bengaluru)
- T20I includes associate nations, which skews unadjusted team win rates
- Women's cricket is out of scope
- **Afghanistan has no matches in Cricsheet** (any format), so it and its players are absent
- Ratings are per name string, and a player rated in several pools has a separate rating in each
