"""FPL Lab live gameweek: your live points, provisional bonus and auto-subs, written to site/live.json.

Runs every 15 minutes while matches are on (see .github/workflows/live.yml). Uses only public FPL endpoints.
    python engine/live.py --team-id 659140 --out site/live.json
"""
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl_engine import API, fpl_session   # noqa: E402

POS_LIMITS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}


def provisional_bonus(fixture):
    """3, 2 and 1 bonus points from the BPS table, shared on ties, as FPL awards them."""
    rows = []
    for st in fixture.get("stats", []):
        if st.get("identifier") == "bps":
            rows = [(r["element"], r["value"]) for side in ("h", "a") for r in st.get(side, [])]
    rows.sort(key=lambda r: -r[1])
    out, awarded, i = {}, 0, 0
    values = sorted({v for _, v in rows}, reverse=True)
    for v in values:
        if awarded >= 3:
            break
        tied = [e for e, val in rows if val == v]
        pts = 3 - awarded
        for e in tied:
            out[e] = pts
        awarded += len(tied)
    return out


def valid_xi(xi, pos_of):
    count = {1: 0, 2: 0, 3: 0, 4: 0}
    for e in xi:
        count[pos_of[e]] += 1
    return all(lo <= count[k] for k, (lo, hi) in POS_LIMITS.items()) and len(xi) == 11


def auto_subs(picks, played, finished_team, pos_of, team_of):
    """FPL's auto-sub rules: a starter who didn't play, once his match is over, is replaced by the first
    bench player (in order) who did play and keeps a valid formation. The goalkeeper only swaps with the reserve keeper."""
    xi = [p["element"] for p in picks if p["position"] <= 11]
    bench = [p["element"] for p in sorted(picks, key=lambda p: p["position"]) if p["position"] > 11]
    subs = []
    for out in list(xi):
        if played.get(out) or not finished_team.get(team_of.get(out)):
            continue
        for b in list(bench):
            if not played.get(b):
                continue
            if (pos_of[out] == 1) != (pos_of[b] == 1):
                continue
            trial = [b if e == out else e for e in xi]
            if pos_of[out] == 1 or valid_xi(trial, pos_of):
                xi, subs = trial, subs + [dict(out=out, **{"in": b})]
                bench.remove(b)
                break
    return xi, subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-id", default=os.environ.get("FPL_TEAM_ID"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "site" / "live.json"))
    a = ap.parse_args()
    s = fpl_session()
    boot = s.get(f"{API}/bootstrap-static/", timeout=20).json()
    cur = next((e for e in boot["events"] if e.get("is_current")), None)
    if not cur:
        print("No gameweek in progress"); return
    gw = cur["id"]
    pos_of = {e["id"]: e["element_type"] for e in boot["elements"]}
    team_of = {e["id"]: e["team"] for e in boot["elements"]}
    live = s.get(f"{API}/event/{gw}/live/", timeout=20).json()
    fixtures = s.get(f"{API}/fixtures/?event={gw}", timeout=20).json()
    stats = {e["id"]: e["stats"] for e in live.get("elements", [])}
    bonus = {}
    finished_team = {}
    for f in fixtures:
        done = f.get("finished") or f.get("finished_provisional")
        for t in (f["team_h"], f["team_a"]):
            finished_team[t] = finished_team.get(t, True) and bool(done) if t in finished_team else bool(done)
        if f.get("started") and not f.get("finished"):
            for e, b in provisional_bonus(f).items():
                if not stats.get(e, {}).get("bonus"):
                    bonus[e] = bonus.get(e, 0) + b
    out = dict(gw=gw, updated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               done=all(f.get("finished") for f in fixtures),
               players={e: dict(pts=st["total_points"] + bonus.get(e, 0), mins=st["minutes"]) for e, st in stats.items() if st["minutes"] or st["total_points"]})
    if a.team_id:
        pk = s.get(f"{API}/entry/{a.team_id}/event/{gw}/picks/", timeout=20).json()
        picks = pk.get("picks", [])
        played = {e: stats.get(e, {}).get("minutes", 0) > 0 for e in pos_of}
        chip = pk.get("active_chip")
        xi, subs = (auto_subs(picks, played, finished_team, pos_of, team_of) if chip != "bboost" else ([p["element"] for p in picks], []))
        mult = {p["element"]: p["multiplier"] for p in picks}
        cap = next((p["element"] for p in picks if p.get("is_captain")), None)
        vice = next((p["element"] for p in picks if p.get("is_vice_captain")), None)
        if cap and not played.get(cap) and finished_team.get(team_of.get(cap)) and vice:
            mult[vice], mult[cap], cap = mult.get(cap, 2), 0, vice          # vice-captain takes over
        pts = lambda e: stats.get(e, {}).get("total_points", 0) + bonus.get(e, 0)
        players = [dict(id=e, pts=pts(e), mult=(mult.get(e, 1) or 1) if e in xi else 0, played=played.get(e, False)) for e in [p["element"] for p in picks]]
        hits = pk.get("entry_history", {}).get("event_transfers_cost", 0)
        total = sum(x["pts"] * x["mult"] for x in players) - hits
        out["my"] = dict(total=total, hits=hits, captain=cap, captain_pts=pts(cap) if cap else 0, mult=mult.get(cap, 2) if cap else 2,
                         bonus=sum(bonus.get(e, 0) for e in xi), subs=subs, players=players, chip=chip)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {a.out}: GW{gw}" + (f", {out['my']['total']} points" if "my" in out else ""))


if __name__ == "__main__":
    main()
