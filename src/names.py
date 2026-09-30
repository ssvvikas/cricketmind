"""Name matching shared by the assistant's tools and the entity search index.

Cricsheet abbreviates given names ("V Kohli", "MS Dhoni"), while users type "Virat Kohli",
"Kohli" or "RCB".  These pure functions map what a user typed onto the names in the data.
"""

import difflib
import re

DOMINANCE = 4  # a name match with 4x the data beats look-alikes

TEAM_ABBREVIATIONS = {
    "csk": "Chennai Super Kings", "mi": "Mumbai Indians", "rcb": "Royal Challengers Bengaluru",
    "kkr": "Kolkata Knight Riders", "srh": "Sunrisers Hyderabad", "dc": "Delhi Capitals",
    "pbks": "Punjab Kings", "rr": "Rajasthan Royals", "lsg": "Lucknow Super Giants",
    "gt": "Gujarat Titans", "aus": "Australia", "ind": "India", "eng": "England",
    "sa": "South Africa", "nz": "New Zealand", "pak": "Pakistan", "sl": "Sri Lanka",
    "ban": "Bangladesh", "wi": "West Indies", "zim": "Zimbabwe", "ire": "Ireland",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower().replace(".", " ").replace("-", " ")).strip()


def score_name(query: str, candidate: str) -> float:
    """How well a user-typed name matches a Cricsheet name such as 'V Kohli' (0-1).

    Cricsheet abbreviates given names ('V Kohli', 'MS Dhoni'), so a surname match plus a
    compatible initial is a strong match, e.g. 'Virat Kohli' -> 'V Kohli'.
    """
    q, c = _norm(query).split(), _norm(candidate).split()
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if q[-1] == c[-1]:  # same surname
        q_given, c_given = q[:-1], "".join(c[:-1])
        if not q_given:
            return 0.9  # surname only
        initials = "".join(t[0] for t in q_given)
        if c_given and (c_given == initials or initials.startswith(c_given)
                        or c_given.startswith(initials)):
            return 0.97
        if c_given and c_given[0] == initials[0]:
            return 0.85
        return 0.55  # same surname, different person
    return round(0.8 * difflib.SequenceMatcher(None, " ".join(q), " ".join(c)).ratio(), 3)


def resolve(query: str, candidates: dict[str, int], limit: int = 5) -> dict:
    """Rank candidate names for a query.  ``candidates`` maps name -> sample size (balls)."""
    scored = sorted(((score_name(query, n), b, n) for n, b in candidates.items()), reverse=True)
    top = [{"name": n, "match": round(s, 2), "balls": int(b)} for s, b, n in scored[:limit]
           if s >= 0.6]
    if not top:
        return {"query": query, "candidates": [], "ambiguous": False, "resolved": None}
    # several people matching about equally well (e.g. plain "Sharma") are ambiguous, unless one
    # has far more data than the rest (an obscure 19-ball "T Kohli" must not block "Kohli")
    tied = [c for c in top if top[0]["match"] - c["match"] < 0.05]
    leader = max(tied, key=lambda c: c["balls"])
    others = [c for c in tied if c is not leader]
    out = {"query": query, "candidates": top, "ambiguous": False, "resolved": leader["name"]}
    if others:
        if leader["balls"] >= DOMINANCE * max(c["balls"] for c in others):
            out["note"] = (f"Assumed {leader['name']}; also matches "
                           f"{', '.join(c['name'] for c in others)} (far less data).")
        else:
            out.update(ambiguous=True, resolved=None)
    return out


