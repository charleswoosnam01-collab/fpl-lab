#!/usr/bin/env python3
"""
FPL Lab engine, version 2.

  python engine/fpl_engine.py                 # fresh data, writes site/data.json
  python engine/fpl_engine.py --cache .cache  # keep downloaded files between runs

Data
  Match, player and shot-level data come from FPL Core Insights
  (github.com/olbauday/FPL-Core-Insights), which merges the official FPL API
  with detailed match stats and refreshes at 07:30 and 17:30 UTC. If the
  official FPL API is reachable, prices, ownership and injury news are then
  overwritten with its live values, so team news is as fresh as possible.

Model, in short
  1. Team ratings: recency-weighted expected goals for and against (blended
     with real goals), adjusted for the opponents faced.
  2. Situational profiles: every shot is tagged by zone (six-yard box,
     central box, wide box, outside the box), body part and situation (open
     play, fast break, set piece). Each team gets a "concedes" and "creates"
     profile per category versus the league average.
  3. Player model: share of team xG and xA, expected minutes, DEFCON hit
     rate, bonus rate, saves, cards.
  4. Style matchup: a player's shot profile is crossed with the opponent's
     concession profile, nudging his expected goals up or down (capped).
  5. Points: Poisson-based expected points per fixture, plus plain-English
     explanations of the main drivers.
  6. Accuracy: the model is re-run as it would have been before each past
     gameweek and scored against actual points and against FPL's own
     expected points.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import hashlib
import json
import sys
import math
import re
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SEASON = "2026-2027"
CORE = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/{season}"
TOURNAMENT = "By%20Tournament/Premier%20League"
API = "https://fantasy.premierleague.com/api"

# ---------------------------------------------------------------- scoring
GOAL = {1: 10, 2: 6, 3: 5, 4: 4}
ASSIST = 3
CLEAN_SHEET = {1: 4, 2: 4, 3: 1, 4: 0}
DEFCON_POINTS = 2
DEFCON_THRESHOLD = {2: 10, 3: 12, 4: 12}

# ---------------------------------------------------------------- tuning
HORIZON = 6               # gameweeks shown in most views
BOOK_WEIGHT = 0.6         # share of a fixture's expected goals taken from bookmaker odds when available
FD_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
ODDS_API = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
TEAM_ALIASES = {   # bookmaker spellings -> FPL names
    "man united": "Man Utd", "manchester united": "Man Utd", "manchester utd": "Man Utd",
    "manchester city": "Man City", "tottenham": "Spurs", "tottenham hotspur": "Spurs",
    "nottingham forest": "Nott'm Forest", "nottm forest": "Nott'm Forest", "newcastle united": "Newcastle",
    "brighton and hove albion": "Brighton", "brighton hove albion": "Brighton", "wolverhampton wanderers": "Wolves",
    "west ham united": "West Ham", "leeds united": "Leeds", "ipswich": "Ipswich Town", "hull": "Hull City",
    "coventry": "Coventry City", "afc bournemouth": "Bournemouth", "leicester city": "Leicester",
    "sheffield utd": "Sheffield Utd", "sheffield united": "Sheffield Utd", "luton town": "Luton",
}
DECAY = 0.92                 # recency weight for player rates
TEAM_DECAY = 0.95            # recency weight for team ratings (0.95 predicted goals best; heavier recency did worse)
HAUL_TEST = dict(n=1516, fpl_proj=5.77, ours=3.81, actual=3.55, best_w=0.0)   # players with a 10+ score in their last three
BREAKOUT_TEST = dict(n=123, beat_again=0.38, avg_next=-0.07, threshold=0.4)   # surprise runs: two games well above forecast
# Availability over time: how much less often a player who's in the team today is still playing 60+ minutes
# each week further ahead (2025/26, expected starters, share playing 60+ minutes by week ahead; nailed players
# declined at the same rate). Applied to weeks two onwards only; next week is unchanged.
AVAIL_DECAY = {1: 0.03, 2: 0.04, 3: 0.045, 4: 0.035}
# Switched off (owner's decision, November 2026): future weeks are projected as if the player is fit, and news
# changes them when it arrives. The decay made later weeks look worse than nearer ones, which skewed chip timing
# and when-to-transfer decisions; for comparing players over the same weeks it changed little.
USE_AVAIL_DECAY = False
AVAIL_FLOOR = 0.78
FUTURE_TEST = dict(windows=540, seasons="2025/26 and 2026/27", mae_model=0.287, mae_last4=0.386, mae_season=0.300,
                   rise_hit=0.89, rise_n=93, rise_avg=0.46, fall_hit=0.87, fall_n=94, fall_avg=-0.51,
                   big_rise_hit=1.0, big_rise_n=17, hot_change=-0.43, hot_stayed=0.48, hot_n=123,
                   cold_change=0.50, cold_stayed=0.41, cold_n=69, box_signal=0.028)   # next-4 team xG forecasts, tested
FORM_TEST = dict(matches=370, loglik={"no extra form": -1.4489, "25% form": -1.4506, "50% form": -1.4537, "100% form": -1.4629},
                 decay={"0.80": -1.4527, "0.88": -1.4500, "0.92": -1.4489, "0.95": -1.4484}, verdict="no extra form; team decay 0.95")
TEAM_PRIOR_MATCHES = 4
FORM_N = 4                   # recent-form window, in matches
FORM_PRIOR_MATCHES = 2       # shrinks a team's recent-form ratio towards its overall rating
FORM_W_ATT = 0.0             # how much recent attacking form moves expected goals (set by form_test)
FORM_W_DEF = 0.0             # how much recent defensive form moves expected goals against
PEN_PRIOR_MATCHES = 8        # shrinks each team's penalty rate towards the league average
VENUE_PRIOR_MATCHES = 8      # shrinkage for each team's home/away split (shown as context)
VENUE_IN_MODEL = False       # team-specific splits made predictions worse when tested (see VENUE_TEST)
VENUE_TEST = dict(matches=369, seasons="2025/26 and 2026/27 to GW9",
                  loglik={"off": -1.4495, "light (32)": -1.4500, "medium (8)": -1.4537, "strong (2)": -1.4670},
                  verdict="off")   # average log-likelihood of actual goals per team-match; higher is better
GOAL_BLEND = 0.25
SHARE_PRIOR = {"g": {1: 0.0, 2: 0.04, 3: 0.10, 4: 0.22}, "a": {1: 0.005, 2: 0.06, 3: 0.12, 4: 0.10}}
SHARE_PRIOR_WEIGHT = 5.0
DEFCON_PRIOR = {1: 0.0, 2: 0.30, 3: 0.15, 4: 0.05}
BONUS_PRIOR = {1: 0.20, 2: 0.20, 3: 0.25, 4: 0.30}
MINUTE_WEIGHTS = [1.0, 0.8, 0.64, 0.51, 0.41]
PROFILE_PRIOR_XG = 1.0      # xG of league-average style mixed into each player profile
STYLE_WEIGHT = 0.5          # how much of the raw style matchup is applied
STYLE_CAP = (0.85, 1.15)
MIN_MINS_FOR_RANKS = 360

SIM_N = 3000               # simulated gameweeks per player for haul chance and ranges
OTHER_COMPS = ["Champions%20League", "Europa%20League", "Conference%20League", "EFL%20Cup", "FA%20Cup"]
CONGESTED_BEFORE = pd.Timedelta(days=4)   # a cup or European match this close before a league game
CONGESTED_AFTER = pd.Timedelta(days=3)    # ...or this close after it

POSITION = {"Goalkeeper": 1, "Defender": 2, "Midfielder": 3, "Forward": 4, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
POS_NAME = {1: "goalkeepers", 2: "defenders", 3: "midfielders", 4: "forwards"}

# Shot categories. x is distance from the goal line (pitch units), y is 0-100 across.
CATS = {
    "six": "the six-yard box", "central": "central areas of the box", "wide": "wide areas of the box",
    "outside": "outside the box", "head": "headers", "fast": "fast breaks", "setpiece": "set pieces",
}
CAT_SHORT = {"six": "Six-yard box", "central": "Central box", "wide": "Wide box", "outside": "Outside box",
             "head": "Headers", "fast": "Fast breaks", "setpiece": "Set pieces"}
ZONES = ["six", "central", "wide", "outside"]
SET_PIECES = {"corner", "set-piece", "throw-in-set-piece", "free-kick"}


def zone_of(x, y):
    central = 36.5 <= y <= 63.5
    if x <= 6 and central:
        return "six"
    if x <= 17:
        return "central" if central else "wide"
    return "outside"


# =========================================================== data loading
def _read(url: str, cache: Path | None, max_age: float | None = 6 * 3600) -> pd.DataFrame:
    path = None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (hashlib.md5(url.encode()).hexdigest() + ".csv")
        if path.exists() and (max_age is None or time.time() - path.stat().st_mtime < max_age):
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
    try:
        df = pd.read_csv(url)
    except (urllib.error.HTTPError, urllib.error.URLError, pd.errors.EmptyDataError):
        df = pd.DataFrame()
    if path is not None:
        df.to_csv(path, index=False)
    return df


def _bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def load_core(season: str = SEASON, cache: Path | None = None, cups: bool = True):
    base = CORE.format(season=season)
    age = None if season != SEASON else 6 * 3600          # finished seasons never change
    rd = lambda url: _read(url, cache, age)
    teams = rd(f"{base}/teams.csv")
    roster = rd(f"{base}/players.csv")
    snaps = rd(f"{base}/playerstats.csv")

    matches, pgs, pms, shots = [], [], [], []
    for gw in range(1, 39):
        folder = f"{base}/{TOURNAMENT}/GW{gw}"
        m = rd(f"{folder}/matches.csv")
        if m.empty:
            continue
        m["gameweek"] = gw
        matches.append(m)
        if _bool(m["finished"]).any():
            for name, bucket in (("player_gameweek_stats", pgs), ("playermatchstats", pms), ("shots", shots)):
                d = rd(f"{folder}/{name}.csv")
                if not d.empty:
                    d["gw"] = gw
                    bucket.append(d)

    code2id = dict(zip(teams["code"].astype(int), teams["id"].astype(int)))
    m = pd.concat(matches, ignore_index=True)
    fixtures = pd.DataFrame({
        "id": m["match_id"].astype(str), "event": m["gameweek"].astype(int),
        "team_h": m["home_team"].map(lambda c: code2id.get(int(c)) if pd.notna(c) else None),
        "team_a": m["away_team"].map(lambda c: code2id.get(int(c)) if pd.notna(c) else None),
        "kickoff": pd.to_datetime(m["kickoff_time"], utc=True, errors="coerce"),
        "finished": _bool(m["finished"]),
        "hs": pd.to_numeric(m["home_score"], errors="coerce"), "as": pd.to_numeric(m["away_score"], errors="coerce"),
        "xg_h": pd.to_numeric(m["home_expected_goals_xg"], errors="coerce"),
        "xg_a": pd.to_numeric(m["away_expected_goals_xg"], errors="coerce"),
    }).dropna(subset=["team_h", "team_a"]).drop_duplicates("id")
    fixtures[["team_h", "team_a"]] = fixtures[["team_h", "team_a"]].astype(int)

    # current players: roster plus the latest snapshot of FPL fields
    snaps = snaps.copy()
    snaps["gw"] = pd.to_numeric(snaps["gw"], errors="coerce")
    latest = snaps.sort_values("gw").groupby("id").tail(1)
    players = roster.rename(columns={"player_id": "id"}).merge(
        latest.drop(columns=[c for c in ("first_name", "second_name", "web_name") if c in latest.columns]),
        on="id", how="left")
    players["team"] = players["team_code"].map(code2id)
    players["element_type"] = players["position"].map(POSITION)
    players = players.dropna(subset=["team", "element_type"])
    players[["team", "element_type"]] = players[["team", "element_type"]].astype(int)

    pg = pd.concat(pgs, ignore_index=True) if pgs else pd.DataFrame()
    pm = pd.concat(pms, ignore_index=True) if pms else pd.DataFrame()
    sh = pd.concat(shots, ignore_index=True) if shots else pd.DataFrame()
    hist = build_history(pg, pm, players, fixtures)
    shots_df = attribute_shots(build_shots(sh, fixtures), hist)
    if cups:
        cups, cup_mins = load_cups(base, cache, code2id)
    else:
        cups = pd.DataFrame(columns=["match", "comp", "team", "kickoff", "finished"])
        cup_mins = pd.DataFrame(columns=["player", "match", "comp", "kickoff", "minutes"])
    return SimpleNamespace(players=players, teams=teams, fixtures=fixtures, hist=hist, shots=shots_df,
                           snaps=snaps, cups=cups, cup_mins=cup_mins, source="FPL Core Insights", season=season,
                           code2id=code2id, team_prior={}, player_prior={})


def load_cups(base, cache, code2id):
    """Every cup and European match involving a Premier League side, plus player minutes in them."""
    rows, mins = [], []
    for comp in OTHER_COMPS:
        name = comp.replace("%20", " ")
        for gw in range(1, 39):
            folder = f"{base}/By%20Tournament/{comp}/GW{gw}"
            m = _read(f"{folder}/matches.csv", cache)
            if m.empty:
                continue
            fin = _bool(m["finished"])
            for side in ("home_team", "away_team"):
                for r, f in zip(m.itertuples(), fin):
                    code = getattr(r, side)
                    if pd.notna(code) and int(code) in code2id:
                        rows.append(dict(match=str(r.match_id), comp=name, team=code2id[int(code)],
                                         kickoff=pd.to_datetime(r.kickoff_time, utc=True, errors="coerce"), finished=bool(f)))
            if fin.any():
                pm = _read(f"{folder}/playermatchstats.csv", cache)
                if not pm.empty:
                    pm = pm[["player_id", "match_id", "minutes_played"]].copy()
                    pm["match"] = pm["match_id"].astype(str)
                    mins.append(pm)
    cups = pd.DataFrame(rows, columns=["match", "comp", "team", "kickoff", "finished"]).drop_duplicates(["match", "team"])
    cm = pd.concat(mins, ignore_index=True) if mins else pd.DataFrame(columns=["player_id", "match", "minutes_played"])
    ko = cups.drop_duplicates("match").set_index("match")["kickoff"]
    cm = cm.assign(kickoff=cm["match"].map(ko), minutes=pd.to_numeric(cm["minutes_played"], errors="coerce").fillna(0),
                   player=pd.to_numeric(cm["player_id"], errors="coerce"), comp=cm["match"].map(
                       cups.drop_duplicates("match").set_index("match")["comp"]))
    return cups, cm[["player", "match", "comp", "kickoff", "minutes"]]


def build_history(pg, pm, players, fixtures):
    """One row per player per finished match, FPL scoring stats plus match detail."""
    if pg.empty:
        return pd.DataFrame()
    num = lambda df, cols: df.assign(**{c: pd.to_numeric(df[c], errors="coerce").fillna(0) for c in cols if c in df})
    pg = num(pg, ["minutes", "starts", "event_points", "goals_scored", "assists", "saves", "bonus", "yellow_cards",
                  "defensive_contribution", "expected_goals", "expected_assists", "clean_sheets"])
    team_of = dict(zip(players["id"], players["team"]))
    pg = pg.rename(columns={"id": "element"})
    pg["team"] = pg["element"].map(team_of)
    pg = pg.dropna(subset=["team"])
    pg["team"] = pg["team"].astype(int)

    done = fixtures[fixtures["finished"]]
    slot = {}
    for r in done.sort_values("kickoff").itertuples():
        slot.setdefault((r.event, r.team_h), (r.id, r.team_a, True, r.kickoff))
        slot.setdefault((r.event, r.team_a), (r.id, r.team_h, False, r.kickoff))
    rows = []
    for r in pg.itertuples():
        s = slot.get((int(r.gw), r.team))
        if s:
            rows.append((r.Index, *s))
    link = pd.DataFrame(rows, columns=["i", "fixture", "opp", "was_home", "kickoff"]).set_index("i")
    h = pg.join(link, how="inner")
    h["event"] = h["gw"].astype(int)

    if not pm.empty:
        extra = ["xg", "xa", "touches_opposition_box", "total_shots", "aerial_duels_won", "accurate_crosses",
                 "chances_created", "xgot", "goals_prevented", "corners", "penalties_scored", "touches", "accurate_passes",
                 "final_third_passes"]
        for c in extra:
            if c not in pm.columns:
                pm[c] = np.nan
        pm = num(pm, extra)
        pm = pm.rename(columns={"player_id": "element", "match_id": "fixture"})[["element", "fixture"] + extra]
        pm["fixture"] = pm["fixture"].astype(str)
        h = h.merge(pm.drop_duplicates(["element", "fixture"]), on=["element", "fixture"], how="left")
    for c in ["xg", "xa", "touches_opposition_box", "total_shots", "aerial_duels_won", "accurate_crosses",
              "chances_created", "xgot", "goals_prevented", "corners", "penalties_scored", "touches", "accurate_passes",
              "final_third_passes"]:
        if c not in h:
            h[c] = np.nan
    h["xg"] = h["xg"].where(h["minutes"] > 0, 0).fillna(h["expected_goals"])
    h["xa"] = h["xa"].where(h["minutes"] > 0, 0).fillna(h["expected_assists"])
    h = h.fillna({"touches_opposition_box": 0, "total_shots": 0, "aerial_duels_won": 0, "accurate_crosses": 0,
                  "chances_created": 0, "xgot": 0, "goals_prevented": 0, "corners": 0, "penalties_scored": 0})
    return h.sort_values(["element", "kickoff"]).reset_index(drop=True)


def build_shots(sh, fixtures):
    if sh.empty:
        return pd.DataFrame(columns=["fixture", "event", "team", "opp", "player", "xg", "xgot", "zone", "head", "sit", "pen", "raw"])
    f = fixtures.set_index("id")
    sh = sh[sh["match_id"].astype(str).isin(f.index)].copy()
    sh["fixture"] = sh["match_id"].astype(str)
    home = _bool(sh["is_home"])
    sh["team"] = np.where(home, sh["fixture"].map(f["team_h"]), sh["fixture"].map(f["team_a"]))
    sh["opp"] = np.where(home, sh["fixture"].map(f["team_a"]), sh["fixture"].map(f["team_h"]))
    sh["event"] = sh["fixture"].map(f["event"])
    sh["xg"] = pd.to_numeric(sh["xg"], errors="coerce").fillna(0)
    sh["zone"] = [zone_of(x, y) for x, y in zip(sh["start_x"].fillna(30), sh["start_y"].fillna(50))]
    sh["head"] = sh["body_part"].eq("head")
    sh["pen"] = sh["situation"].eq("penalty")
    sh["sit"] = np.where(sh["situation"].isin(SET_PIECES), "setpiece",
                         np.where(sh["situation"].eq("fast-break"), "fast", np.where(sh["pen"], "pen", "open")))
    sh["player"] = pd.to_numeric(sh["player_id"], errors="coerce")
    sh["raw"] = sh["situation"].astype(str)
    sh["xgot"] = pd.to_numeric(sh["xgot"], errors="coerce")
    return sh[["fixture", "event", "team", "opp", "player", "xg", "xgot", "zone", "head", "sit", "pen", "raw"]].reset_index(drop=True)


def attribute_shots(sh, hist):
    """About one shot in ten arrives without a player id. Give each to the teammate in that
    match whose expected goals are not yet accounted for by his tagged shots."""
    miss = sh["player"].isna()
    if not miss.any() or hist.empty:
        return sh
    got = sh[~miss].groupby(["fixture", "player"])["xg"].sum().to_dict()
    resid = {}
    for r in hist[hist["minutes"] > 0].itertuples():
        resid.setdefault((r.fixture, r.team), {})[r.element] = r.xg - got.get((r.fixture, r.element), 0.0)
    sh = sh.copy()
    for i in sh.index[miss].to_series().sort_values(key=lambda idx: -sh.loc[idx, "xg"]):
        cands = resid.get((sh.at[i, "fixture"], sh.at[i, "team"]))
        if not cands:
            continue
        pid, rem = max(cands.items(), key=lambda kv: kv[1])
        xg = sh.at[i, "xg"]
        if rem >= 0.5 * xg - 0.02:
            sh.at[i, "player"] = pid
            cands[pid] = rem - xg
    return sh


# Switched off after testing (November 2026):
# - ClubElo: an Elo rating built from every result since August 2025 made this season's predictions slightly worse
#   (goals log-likelihood -1.4863 against -1.4822 on GW3-9, -1.3733 against -1.3705 on GW7-9); ClubElo is the same
#   kind of results-based rating, and the site was returning HTTP 502 to GitHub's servers anyway.
# - Understat: now answers its data address from GitHub's servers with an ordinary web page instead of data, and it
#   only ever fed the "second opinion" figures on player pages, never the projections.
USE_CLUBELO = False
USE_UNDERSTAT = False
SOURCE_NOTES = {}
SOURCE_STATE = {}           # extra sources that only report when they were actually tried           # why a data source wasn't used this run, shown on the Model tab


def note_source(name, why):
    SOURCE_NOTES[name] = str(why)[:160]
    print(f"{name}: {why}")


_FPL_SESSION = None


def fpl_session():
    """One shared HTTP session for the official FPL API: keeps connections open, identifies itself,
    and retries politely (with backoff) when FPL is busy or updating between gameweeks."""
    global _FPL_SESSION
    if _FPL_SESSION is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0 (FPL Lab)"
        retry = Retry(total=4, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8))
        _FPL_SESSION = s
    return _FPL_SESSION


def overlay_api(ds):
    """Replace prices, ownership and injury news with live official values, if reachable."""
    try:
        import requests
        boot = fpl_session().get(f"{API}/bootstrap-static/", timeout=20,
                            headers={"User-Agent": "Mozilla/5.0 (FPL Lab)"}).json()
    except Exception as e:  # noqa: BLE001
        print("Official API not reachable, using dataset values:", e)
        return False
    live = pd.DataFrame(boot["elements"]).set_index("id")
    try:   # is the match data behind the official game? (FPL Core Insights updates twice a day)
        fpl_done = max((e["id"] for e in boot.get("events", []) if e.get("finished") and e.get("data_checked")), default=0)
        core_done = int(ds.fixtures.loc[ds.fixtures["finished"], "event"].max())
        ds.behind = max(0, fpl_done - core_done)
        if ds.behind:
            print(f"Match data is {ds.behind} gameweek(s) behind the official game")
    except Exception:  # noqa: BLE001
        ds.behind = 0
    p = ds.players.set_index("id")
    for col in ["status", "news", "chance_of_playing_next_round", "selected_by_percent", "penalties_order",
                "corners_and_indirect_freekicks_order", "ep_next", "transfers_in_event", "transfers_out_event",
                "price_change_percent", "price_change_hourly_rate", "price_change_projections",
                "price_change_locked_until"]:
        if col in live:
            val = live[col]
            if col in p and pd.api.types.is_numeric_dtype(p[col]):
                val = pd.to_numeric(val, errors="coerce")   # API sends some numbers as text, e.g. "25.3"
            elif col in p:
                p[col] = p[col].astype(object)
            p.loc[p.index.intersection(live.index), col] = val
    p.loc[p.index.intersection(live.index), "now_cost"] = live["now_cost"] / 10
    ds.players = p.reset_index()
    try:   # official fixture list: catches rearranged games (new doubles) and games awaiting a date (new blanks)
        import requests
        fx = fpl_session().get(f"{API}/fixtures/", timeout=20).json()
        pairs = {(r.team_h, r.team_a): r.Index for r in ds.fixtures.itertuples()}
        ds.unscheduled = []
        for f in fx:
            idx = pairs.get((f["team_h"], f["team_a"]))
            if idx is None:
                continue
            if f.get("event") is None:
                ds.unscheduled.append(dict(h=int(f["team_h"]), a=int(f["team_a"])))
                ds.fixtures.loc[idx, "event"] = 99
            else:
                ds.fixtures.loc[idx, "event"] = int(f["event"])
                if f.get("kickoff_time"):
                    ds.fixtures.loc[idx, "kickoff"] = pd.Timestamp(f["kickoff_time"])
        ds.fixtures = ds.fixtures[ds.fixtures["event"] != 99]
    except Exception as e:  # noqa: BLE001
        print("Official fixture list not reachable:", e)
    ds.source = "FPL Core Insights, with live prices and news from the official FPL API"
    ds.live_ok = True
    return True


def at_cutoff(ds, gw):
    """The dataset exactly as it looked after gameweek `gw` (for backtesting)."""
    fx = ds.fixtures.copy()
    fx["finished"] = fx["finished"] & (fx["event"] <= gw)
    snap = ds.snaps[ds.snaps["gw"] == gw]
    keep = ["id", "status", "news", "chance_of_playing_next_round", "now_cost", "selected_by_percent",
            "penalties_order", "corners_and_indirect_freekicks_order", "ep_next"]
    players = ds.players.drop(columns=[c for c in keep[1:] if c in ds.players]).merge(
        snap[[c for c in keep if c in snap]], on="id", how="inner")
    return SimpleNamespace(players=players, teams=ds.teams, fixtures=fx,
                           hist=ds.hist[ds.hist["event"] <= gw], shots=ds.shots[ds.shots["event"] <= gw],
                           snaps=ds.snaps, cups=ds.cups, cup_mins=ds.cup_mins, source=ds.source, season=ds.season,
                           code2id=ds.code2id, team_prior=ds.team_prior, player_prior=ds.player_prior,
                           refs=getattr(ds, "refs", {}), results_odds=getattr(ds, "results_odds", {}))


# ============================================================ team model
def team_model(ds):
    teams, fixtures = ds.teams, ds.fixtures
    done = fixtures[fixtures["finished"] & fixtures["xg_h"].notna()]
    last_gw = int(done["event"].max()) if len(done) else 0
    dcs = ds.hist.groupby(["fixture", "team"])["defensive_contribution"].sum().to_dict() if len(ds.hist) else {}

    m = pd.DataFrame([dict(
        id=r.id, h=r.team_h, a=r.team_a, ev=r.event, w=TEAM_DECAY ** (last_gw - r.event),
        ph=(1 - GOAL_BLEND) * r.xg_h + GOAL_BLEND * r.hs, pa=(1 - GOAL_BLEND) * r.xg_a + GOAL_BLEND * r.as_,
        xh=r.xg_h, xa=r.xg_a, gh=r.hs, ga=r.as_,
        dch=dcs.get((r.id, r.team_h), 0.0), dca=dcs.get((r.id, r.team_a), 0.0),
    ) for r in done.rename(columns={"as": "as_"}).itertuples()])

    ids = sorted(teams["id"].astype(int))
    att = {t: 1.0 for t in ids}
    dfn = {t: 1.0 for t in ids}
    if m.empty:
        prior = getattr(ds, "team_prior", {}) or {}
        att = {t: prior.get(t, (1.0, 1.0))[0] for t in ids}
        dfn = {t: prior.get(t, (1.0, 1.0))[1] for t in ids}
        return dict(mu_h=1.45, mu_a=1.2, att=att, dfn=dfn, gen={t: 1.0 for t in ids}, last_gw=0, m=m, ids=ids)

    mu_h, mu_a = float(np.average(m.ph, weights=m.w)), float(np.average(m.pa, weights=m.w))
    k = TEAM_PRIOR_MATCHES * (mu_h + mu_a) / 2
    prior = getattr(ds, "team_prior", {}) or {}
    # vectorised fit (same maths as a per-team loop, much faster)
    pos_of = {t: i for i, t in enumerate(ids)}
    hi, ai = m.h.map(pos_of).to_numpy(), m.a.map(pos_of).to_numpy()
    w, ph, pa = m.w.to_numpy(), m.ph.to_numpy(), m.pa.to_numpy()
    nT = len(ids)
    a0 = np.array([prior.get(t, (1.0, 1.0))[0] for t in ids])
    d0 = np.array([prior.get(t, (1.0, 1.0))[1] for t in ids])
    bc = lambda idx, v: np.bincount(idx, weights=v, minlength=nT)
    made = bc(hi, w * ph) + bc(ai, w * pa)
    allowed = bc(hi, w * pa) + bc(ai, w * ph)
    A, Dv = np.ones(nT), np.ones(nT)
    for _ in range(40):
        na = (made + k * a0) / (bc(hi, w * mu_h * Dv[ai]) + bc(ai, w * mu_a * Dv[hi]) + k)
        nd = (allowed + k * d0) / (bc(hi, w * mu_a * A[ai]) + bc(ai, w * mu_h * A[hi]) + k)
        A = na / math.exp(np.mean(np.log(na)))
        Dv = nd / math.exp(np.mean(np.log(nd)))
    att = {t: float(A[i]) for i, t in enumerate(ids)}
    dfn = {t: float(Dv[i]) for i, t in enumerate(ids)}

    vf = venue_factors(m, ids, att, dfn, mu_h, mu_a)
    form = recent_form(m, ids, att, dfn, mu_h, mu_a)

    mean_dc = float(np.average(np.r_[m.dch, m.dca], weights=np.r_[m.w, m.w])) or 1.0
    gen = {}
    for t in ids:
        H, A = m[m.h == t], m[m.a == t]
        forced = (H.w * H.dca).sum() + (A.w * A.dch).sum()
        gen[t] = ((forced + 3 * mean_dc) / (H.w.sum() + A.w.sum() + 3)) / mean_dc
    return dict(mu_h=mu_h, mu_a=mu_a, att=att, dfn=dfn, gen=gen, last_gw=last_gw, m=m, ids=ids, vf=vf, form=form,
                pen=penalty_rates(ds, m, ids))


def recent_form(m, ids, att, dfn, mu_h, mu_a, n=FORM_N):
    """Each team's last n matches against what its overall rating predicted for those games:
    above 1 means hotter than its rating, below 1 colder. Shrunk towards 1 for small samples."""
    out = {}
    mu = (mu_h + mu_a) / 2
    for t in ids:
        g = m[(m.h == t) | (m.a == t)].sort_values("ev").tail(n)
        if g.empty:
            out[t] = dict(att=1.0, dfn=1.0, n=0, xg=0.0, xga=0.0)
            continue
        home = g.h == t
        made = np.where(home, g.ph, g.pa).sum()
        allowed = np.where(home, g.pa, g.ph).sum()
        exp_made = np.where(home, mu_h * att[t] * g.a.map(dfn), mu_a * att[t] * g.h.map(dfn)).sum()
        exp_allowed = np.where(home, mu_a * dfn[t] * g.a.map(att), mu_h * dfn[t] * g.h.map(att)).sum()
        k = FORM_PRIOR_MATCHES * mu
        out[t] = dict(att=float((made + k) / (exp_made + k)), dfn=float((allowed + k) / (exp_allowed + k)), n=int(len(g)),
                      xg=float(np.where(home, g.xh, g.xa).sum()), xga=float(np.where(home, g.xa, g.xh).sum()))
    return out


def penalty_rates(ds, m, ids):
    """Penalty xG each team wins per match, shrunk towards the league average."""
    sh = ds.shots
    if sh is None or not len(sh) or m.empty:
        return {t: PEN_XG * 0.12 for t in ids}
    fx = set(m.id)
    pens = sh[sh["pen"] & sh["fixture"].isin(fx)].groupby("team")["xg"].sum().to_dict()
    games = {t: int(((m.h == t) | (m.a == t)).sum()) for t in ids}
    league = sum(pens.values()) / max(1, sum(games.values()))
    return {t: (pens.get(t, 0.0) + PEN_PRIOR_MATCHES * league) / (games[t] + PEN_PRIOR_MATCHES) for t in ids}


def venue_factors(m, ids, att, dfn, mu_h, mu_a):
    """How much better or worse each team attacks and defends at home and away than its overall rating
    implies, shrunk towards no difference until there are enough matches at that venue."""
    one = {"ah": 1.0, "aa": 1.0, "dh": 1.0, "da": 1.0}
    if VENUE_PRIOR_MATCHES is None or m.empty:
        return {t: dict(one) for t in ids}
    out = {}
    for t in ids:
        H, A = m[m.h == t], m[m.a == t]
        kh, ka = VENUE_PRIOR_MATCHES * mu_h, VENUE_PRIOR_MATCHES * mu_a
        f = lambda obs, exp, k: float((obs + k) / (exp + k))
        out[t] = dict(
            ah=f((H.w * H.ph).sum(), (H.w * mu_h * att[t] * H.a.map(dfn)).sum(), kh),   # attack at home
            aa=f((A.w * A.pa).sum(), (A.w * mu_a * att[t] * A.h.map(dfn)).sum(), ka),   # attack away
            dh=f((H.w * H.pa).sum(), (H.w * mu_a * dfn[t] * H.a.map(att)).sum(), ka),   # conceding at home
            da=f((A.w * A.ph).sum(), (A.w * mu_h * dfn[t] * A.h.map(att)).sum(), kh),   # conceding away
        )
    return out


def team_profiles(ds, tm):
    """xG created and conceded per match by shot category, versus league average."""
    sh = ds.shots[~ds.shots["pen"]]
    m = tm["m"]
    ids = tm["ids"]
    if sh.empty or m.empty:
        return {t: {"def": {}, "att": {}} for t in ids}, {}
    with_shots = set(sh["fixture"])
    m = m[m["id"].isin(with_shots)]
    w_of = dict(zip(m["id"], m["w"]))
    sh = sh[sh["fixture"].isin(w_of)].assign(w=lambda d: d["fixture"].map(w_of))
    tagged = []
    for c in CATS:
        mask = sh["zone"].eq(c) if c in ZONES else sh["head"] if c == "head" else sh["sit"].eq(c)
        tagged.append(sh[mask].assign(cat=c))
    tagged = pd.concat(tagged)
    tagged["wxg"] = tagged["w"] * tagged["xg"]
    made = tagged.groupby(["team", "cat"])["wxg"].sum()
    allowed = tagged.groupby(["opp", "cat"])["wxg"].sum()
    games = {t: m.loc[(m.h == t) | (m.a == t), "w"].sum() for t in ids}
    total_games = sum(games.values())
    league = {c: tagged.loc[tagged.cat == c, "wxg"].sum() / total_games for c in CATS}
    prof = {}
    for t in ids:
        prof[t] = {"def": {}, "att": {}}
        for c in CATS:
            for side, series in (("def", allowed), ("att", made)):
                v = float(series.get((t, c), 0.0))
                pm = (v + TEAM_PRIOR_MATCHES * league[c]) / (games[t] + TEAM_PRIOR_MATCHES)
                prof[t][side][c] = {"pm": pm, "idx": pm / league[c] if league[c] else 1.0}
    for side in ("def", "att"):
        for c in CATS:
            order = sorted(ids, key=lambda t: -prof[t][side][c]["pm"])
            for i, t in enumerate(order):
                prof[t][side][c]["most"] = i + 1      # 1 = highest in the league
    return prof, league


def team_table(ds, tm):
    m = tm["m"]
    out = {}
    for t in tm["ids"]:
        H, A = m[m.h == t], m[m.a == t]
        n = len(H) + len(A)
        out[t] = dict(
            n=n,
            xgf=(H.xh.sum() + A.xa.sum()) / n if n else None, xga=(H.xa.sum() + A.xh.sum()) / n if n else None,
            xgf_h=H.xh.mean() if len(H) else None, xgf_a=A.xa.mean() if len(A) else None,
            xga_h=H.xa.mean() if len(H) else None, xga_a=A.xh.mean() if len(A) else None,
            cs=(((H.ga == 0).sum() + (A.gh == 0).sum()) / n) if n else None,
        )
    for key, reverse in (("xgf", True), ("xga", False)):
        order = sorted([t for t in out if out[t][key] is not None], key=lambda t: -out[t][key] if reverse else out[t][key])
        for i, t in enumerate(order):
            out[t]["r_" + key] = i + 1   # xgf: 1 = creates most; xga: 1 = concedes least
    # home and away form
    for t in tm["ids"]:
        H, A = m[m.h == t].sort_values("ev"), m[m.a == t].sort_values("ev")
        out[t]["home"] = venue_block([(r.ev, r.gh, r.ga, r.xh, r.xa) for r in H.itertuples()])
        out[t]["away"] = venue_block([(r.ev, r.ga, r.gh, r.xa, r.xh) for r in A.itertuples()])
        out[t]["vf"] = {k: round(v, 3) for k, v in tm.get("vf", {}).get(t, {}).items()}
    for side in ("home", "away"):
        have = [t for t in out if out[t][side]]
        for key, rev in (("xgf", True), ("xga", False), ("ppg", True)):
            for i, t in enumerate(sorted(have, key=lambda t: -out[t][side][key] if rev else out[t][side][key])):
                out[t][side]["r_" + key] = i + 1   # 1 = best in the league at that venue
    return out


def venue_block(rows):
    """Record at one venue: results, goals, xG, last five."""
    if not rows:
        return None
    n = len(rows)
    w = sum(1 for r in rows if r[1] > r[2]); d = sum(1 for r in rows if r[1] == r[2]); l = n - w - d
    return dict(n=n, w=w, d=d, l=l, gf=int(sum(r[1] for r in rows)), ga=int(sum(r[2] for r in rows)),
                xgf=round(float(np.mean([r[3] for r in rows])), 2), xga=round(float(np.mean([r[4] for r in rows])), 2),
                ppg=round((3 * w + d) / n, 2),
                form="".join("W" if r[1] > r[2] else "D" if r[1] == r[2] else "L" for r in rows[-5:]))


def fixture_lambdas(tm, h, a, venue=None):
    venue = VENUE_IN_MODEL if venue is None else venue
    vf = tm.get("vf") if venue else None
    vh, va = (vf.get(h), vf.get(a)) if vf else (None, None)
    one = {"ah": 1.0, "aa": 1.0, "dh": 1.0, "da": 1.0}
    vh, va = vh or one, va or one
    fm = tm.get("form") or {}
    one_f = {"att": 1.0, "dfn": 1.0}
    fh, fa = fm.get(h, one_f), fm.get(a, one_f)
    return (tm["mu_h"] * tm["att"][h] * vh["ah"] * tm["dfn"][a] * va["da"] * fh["att"] ** FORM_W_ATT * fa["dfn"] ** FORM_W_DEF,
            tm["mu_a"] * tm["att"][a] * va["aa"] * tm["dfn"][h] * vh["dh"] * fa["att"] ** FORM_W_ATT * fh["dfn"] ** FORM_W_DEF)


# ========================================================== player model
def _shrink(num, den, prior, weight):
    return (num + prior * weight) / (den + weight)


def _text(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def player_model(ds, tm):
    hist, shots, players = ds.hist, ds.shots, ds.players
    last_gw = tm["last_gw"]
    fx = ds.fixtures.set_index("id")
    team_xg = {}
    for r in ds.fixtures[ds.fixtures["finished"]].itertuples():
        team_xg[(r.id, r.team_h)], team_xg[(r.id, r.team_a)] = r.xg_h, r.xg_a
    pos_of = dict(zip(players["id"], players["element_type"]))

    # league-average shot style by position, used as a prior for each player
    np_shots = shots[~shots["pen"]].copy()
    np_shots["pos"] = np_shots["player"].map(pos_of)
    pos_style = {}
    for pos in (1, 2, 3, 4):
        s = np_shots[np_shots["pos"] == pos]
        tot = s["xg"].sum() or 1.0
        pos_style[pos] = {c: (s.loc[cat_mask(s, c), "xg"].sum() / tot) for c in CATS}
    by_player = {pid: g for pid, g in np_shots.groupby("player")}
    on_t = shots[shots["xgot"].notna()]
    placement = (on_t["xgot"] - on_t["xg"]).groupby(on_t["player"]).sum().to_dict()
    pen_xg = shots[shots["pen"]].groupby("player")["xg"].sum().to_dict()
    pen_pf = shots[shots["pen"]].groupby(["player", "fixture"])["xg"].sum().to_dict()      # penalty xG by player and match
    pen_tf = shots[shots["pen"]].groupby(["fixture", "team"])["xg"].sum().to_dict()        # penalty xG by team and match

    rates = {}
    for pid, g in hist.groupby("element"):
        pos = pos_of.get(pid)
        if pos is None:
            continue
        w = DECAY ** (last_gw - g["event"])
        mins = g["minutes"]
        txg = np.array([team_xg.get((f, t), 0.0) - pen_tf.get((f, t), 0.0) for f, t in zip(g.fixture, g.team)], dtype=float)
        pxg = np.array([pen_pf.get((pid, f), 0.0) for f in g.fixture], dtype=float)
        np_xg_match = np.maximum(0.0, g.xg.to_numpy() - pxg)       # open-play and set-piece xG, penalties removed
        oxg = np.array([team_xg.get((f, o), 0.0) for f, o in zip(g.fixture, g.opp)], dtype=float)
        on = np.nan_to_num(txg) * mins / 90
        pp = (getattr(ds, "player_prior", {}) or {}).get(pid)
        prior_g = pp["share_g"] if pp else SHARE_PRIOR["g"][pos]
        prior_a = pp["share_a"] if pp else SHARE_PRIOR["a"][pos]
        share_g = _shrink((w * np_xg_match).sum(), (w * on).sum(), prior_g, SHARE_PRIOR_WEIGHT)
        share_a = _shrink((w * g.xa).sum(), (w * on).sum(), prior_a, SHARE_PRIOR_WEIGHT)

        full = g[mins >= 60]
        if pos in DEFCON_THRESHOLD and len(full):
            hits = int((full.defensive_contribution >= DEFCON_THRESHOLD[pos]).sum())
            dc_hit = _shrink(hits, len(full), pp["dc_hit"] if pp and pp.get("dc_hit") is not None else DEFCON_PRIOR[pos], 3)
            dc90 = full.defensive_contribution.sum() / full.minutes.sum() * 90
        else:
            hits, dc_hit, dc90 = 0, DEFCON_PRIOR.get(pos, 0.0), 0.0
        tot_m = float((w * mins).sum())
        b_prior = pp["bonus90"] if pp else BONUS_PRIOR[pos]
        bonus90 = _shrink((w * g.bonus).sum() * 90, tot_m, b_prior, 450) if tot_m else b_prior
        yc90 = _shrink((w * g.yellow_cards).sum() * 90, tot_m, 0.12, 450) if tot_m else 0.12
        save_ratio = _shrink((w * g.saves).sum(), (w * np.nan_to_num(oxg) * mins / 90).sum(), 2.6, 3) if pos == 1 else 0.0

        recent = g.tail(len(MINUTE_WEIGHTS)).iloc[::-1]
        mw = np.array(MINUTE_WEIGHTS[: len(recent)])
        rm = recent["minutes"].to_numpy()
        emins = float((mw * rm).sum() / mw.sum()) if len(rm) else 0.0
        p60 = float((mw * (rm >= 60)).sum() / mw.sum()) if len(rm) else 0.0
        p1 = float((mw * ((rm > 0) & (rm < 60))).sum() / mw.sum()) if len(rm) else 0.0

        # shot style profile (non-penalty xG by category), shrunk to the position average
        ps = by_player.get(pid)
        np_xg = float(ps["xg"].sum()) if ps is not None else 0.0
        profile = {}
        for c in CATS:
            own = float(ps.loc[cat_mask(ps, c), "xg"].sum()) if ps is not None else 0.0
            base_c = pp["profile"][c] if pp and pp.get("profile") else pos_style[pos][c]
            weight = PROFILE_PRIOR_XG * (3 if pp and pp.get("profile") else 1)
            profile[c] = (own + base_c * weight) / (np_xg + weight)

        m_total = float(mins.sum())
        per90 = lambda v: float(v) / m_total * 90 if m_total else 0.0
        last4 = g[mins > 0].tail(4)
        l4m = float(last4.minutes.sum())
        rates[pid] = dict(
            pos=pos, share_g=share_g, share_a=share_a, dc_hit=dc_hit, dc90=dc90, dc_hits=hits, dc_games=len(full),
            bonus90=bonus90, yc90=yc90, save_ratio=save_ratio, emins=emins, p60=p60, p1=p1, profile=profile,
            np_xg=np_xg, pen_xg=float(pen_xg.get(pid, 0.0)),
            season=dict(
                mins=int(m_total), apps=int((mins > 0).sum()), starts=int(g.starts.sum()), games=int(len(g)),
                goals=int(g.goals_scored.sum()), assists=int(g.assists.sum()), pts=int(g.event_points.sum()),
                xg90=per90(g.xg.sum()), xa90=per90(g.xa.sum()), npxg90=per90(np_xg),
                xg_total=float(g.xg.sum()), xa_total=float(g.xa.sum()), xgot_total=float(g.xgot.sum()),
                gp_total=float(g.goals_prevented.sum()), pens_scored=int(g.penalties_scored.sum()),
                place=float(placement.get(pid, 0.0)),
                shots90=per90(g.total_shots.sum()), box90=per90(g.touches_opposition_box.sum()),
                aerial90=per90(g.aerial_duels_won.sum()), crosses90=per90(g.accurate_crosses.sum()),
                chances90=per90(g.chances_created.sum()), dc90=dc90,
                xgi90_last4=float((last4.xg.sum() + last4.xa.sum()) / l4m * 90) if l4m >= 180 else None,
            ),
        )
    return rates


def cat_mask(s, c):
    if c in ZONES:
        return s["zone"].eq(c)
    if c == "head":
        return s["head"]
    return s["sit"].eq(c)


def percentiles(rates, keys):
    out = {pid: {} for pid in rates}
    for pos in (1, 2, 3, 4):
        pool = [pid for pid, r in rates.items() if r["pos"] == pos and r["season"]["mins"] >= MIN_MINS_FOR_RANKS]
        for k in keys:
            vals = sorted((rates[p]["season"].get(k) or 0.0) if k != "share_g" else rates[p]["share_g"] for p in pool)
            n = len(vals)
            for pid, r in rates.items():
                if r["pos"] != pos or n == 0:
                    continue
                v = r["share_g"] if k == "share_g" else (r["season"].get(k) or 0.0)
                below = int(np.searchsorted(vals, v, side="left"))
                rank = n - int(np.searchsorted(vals, v, side="right")) + 1
                out[pid][k] = dict(p=round(100 * below / max(1, n - 1)) if n > 1 else 50, rank=rank, of=n,
                                   ranked=r["season"]["mins"] >= MIN_MINS_FOR_RANKS)
    return out


# ============================================================ projection
def availability(p, gw_dates, gws):
    status = str(p.get("status", "a"))
    chance = p.get("chance_of_playing_next_round")
    chance = 1.0 if chance is None or pd.isna(chance) else float(chance) / 100
    if status in ("u", "n"):
        return [0.0] * len(gws)
    if status == "a":
        # an available player is fully available: FPL leaves the chance blank for them, but some saved snapshots
        # record that blank as 0, which made backtests think nobody would play
        return [1.0] * len(gws)
    if status == "s":
        return [0.0] + [1.0] * (len(gws) - 1)
    news = _text(p.get("news"))
    back = re.search(r"[Ee]xpected back (\d{1,2}) ([A-Z][a-z]{2})", news)
    if back and gw_dates:
        first = min(gw_dates.values())
        try:
            ret = dt.datetime.strptime(f"{back.group(1)} {back.group(2)} {first.year}", "%d %b %Y")
            if ret.date() < first.date() - dt.timedelta(days=60):
                ret = ret.replace(year=first.year + 1)
            return [0.0 if gw_dates.get(g) is None or gw_dates[g].date() < ret.date() else 0.85 for g in gws]
        except ValueError:
            pass
    ramp = [chance]
    for _ in range(len(gws) - 1):
        ramp.append(min(1.0, ramp[-1] + 0.25))
    return ramp


def e_floor_div(lam, d):
    if lam <= 0:
        return 0.0
    total, pk = 0.0, math.exp(-lam)
    for k in range(30):
        if k:
            pk *= lam / k
        total += (k // d) * pk
    return total


def style_factor(profile, opp_prof):
    """How well a player's shot style fits what the opponent concedes, relative to their overall level."""
    if not profile or not opp_prof:
        return 1.0, []
    zone = sum(profile[c] * opp_prof[c]["idx"] for c in ZONES) / max(1e-6, sum(profile[c] for c in ZONES))
    zone_avg = sum(opp_prof[c]["pm"] for c in ZONES)
    league_zone = sum(opp_prof[c]["pm"] / opp_prof[c]["idx"] for c in ZONES if opp_prof[c]["idx"])
    overall = zone_avg / league_zone if league_zone else 1.0
    head = 1 + profile["head"] * (opp_prof["head"]["idx"] / overall - 1)
    sp = 1 + profile["setpiece"] * (opp_prof["setpiece"]["idx"] / overall - 1)
    fast = 1 + profile["fast"] * (opp_prof["fast"]["idx"] / overall - 1)
    raw = (zone / overall) * head * sp * fast
    f = 1 + STYLE_WEIGHT * (raw - 1)
    f = min(STYLE_CAP[1], max(STYLE_CAP[0], f))
    drivers = []
    for c in CATS:
        share = profile[c]
        if share >= 0.12:
            drivers.append((share * (opp_prof[c]["idx"] / overall - 1), c))
    drivers.sort()
    return f, drivers


def fixture_points(r, lam_for, lam_against, gen, avail, style=1.0, extra_xg=0.0):
    pos = r["pos"]
    p60, p1 = r["p60"] * avail, r["p1"] * avail
    mf = r["emins"] * avail / 90
    pen_team = r.get("pen_rate", 0.0) * min(1.6, lam_for / 1.35)        # the team's expected penalty xG in this match
    lam_np = max(0.05, lam_for - pen_team)
    xg_pen = (pen_team + extra_xg) * mf if r.get("taker") else 0.0
    xg_np = r["share_g"] * lam_np * mf * style
    xg = xg_np + xg_pen
    xa = r["share_a"] * lam_for * mf
    cs_prob = math.exp(-lam_against)
    pts = {
        "app": 2 * p60 + p1, "goals": GOAL[pos] * xg_np, "pens": GOAL[pos] * xg_pen, "assists": ASSIST * xa,
        "cs": CLEAN_SHEET[pos] * p60 * cs_prob,
        "gc": -p60 * e_floor_div(lam_against, 2) if pos in (1, 2) else 0.0,
        "saves": p60 * e_floor_div(r["save_ratio"] * lam_against, 3) if pos == 1 else 0.0,
        "cards": -r["yc90"] * mf,
    }
    dc_p = 0.0
    if pos in DEFCON_THRESHOLD and r["dc_hit"] > 0:
        odds = r["dc_hit"] / max(1e-6, 1 - r["dc_hit"]) * gen ** 2
        dc_p = odds / (1 + odds)
    pts["defcon"] = DEFCON_POINTS * p60 * dc_p
    detail = dict(xg=xg, xa=xa, cs=cs_prob, dc_p=dc_p * avail, dc_cond=dc_p, dc_proj=r["dc90"] * gen * mf,
                  ret=1 - math.exp(-(xg + xa)), lf=lam_for, la=lam_against, style=style, pen_extra=extra_xg * mf,
                  xg_np=xg_np, xg_pen=xg_pen, lam_np=lam_np)
    return pts, detail


def default_rates(pos):
    return dict(pos=pos, share_g=SHARE_PRIOR["g"][pos], share_a=SHARE_PRIOR["a"][pos], dc_hit=DEFCON_PRIOR[pos],
                dc90=0.0, dc_hits=0, dc_games=0, bonus90=BONUS_PRIOR[pos], yc90=0.12, save_ratio=2.6, emins=0.0,
                p60=0.0, p1=0.0, profile=None, np_xg=0.0, pen_xg=0.0, season=None)


MODEL_VERSION = "2026.11-c"   # bump when the projection model changes (not for app-only changes)
CARD_BANS = True
# Premier League booking bans (2026/27): five league yellows within a club's first 19 league matches = one-match
# ban; ten up to and including match 32 = two-match ban. Served in the next league match(es); cup cards don't count.
CARD_RULES = [(5, 19, 1), (10, 32, 2)]
# Players close to a ban get booked about half as often as their booking rate suggests (more careful, and part of a
# high early rate is luck). Fitted on 2025/26: of 35 players on four yellows by GW15, 57% were banned by GW19
# against 77% predicted at the full rate; the best scale was 0.45-0.55 across three such groups.
CARD_RISK_SCALE = 0.5


def ban_chances(yc, yc90, emins_by_fixture, match_nos, risk_slots=None):
    """Chance he's available (not serving a booking ban) for each upcoming league fixture, and the chance a ban
    starts within these fixtures. yc: league yellows so far; yc90: booking rate per 90; emins_by_fixture: expected
    minutes per fixture; match_nos: the club's league match number for each fixture."""
    if not CARD_BANS or not any(yc < n for n, _, _ in CARD_RULES):
        return [1.0] * len(match_nos), 0.0, None
    nxt = next(((n, by, ban) for n, by, ban in CARD_RULES if yc < n), None)
    if nxt and match_nos and match_nos[0] > nxt[1]:
        nxt = next(((n, by, ban) for n, by, ban in CARD_RULES if yc < n and match_nos[0] <= by), None)
    state = {(int(yc), 0): 1.0}
    avail, started = [], 0.0
    for idx, (em, m) in enumerate(zip(emins_by_fixture, match_nos)):
        avail.append(sum(p for (y, b), p in state.items() if b == 0))
        counting = risk_slots is None or idx < risk_slots
        q = 1 - math.exp(-CARD_RISK_SCALE * max(0.0, yc90) * max(0.0, em) / 90)
        new = {}
        for (y, b), p in state.items():
            if b > 0:
                new[(y, b - 1)] = new.get((y, b - 1), 0.0) + p
                continue
            ban = next((bn for n, by, bn in CARD_RULES if y + 1 == n and m <= by), 0)
            if counting and nxt and y + 1 == nxt[0] and ban:   # the next threshold only, within the shown weeks
                started += p * q
            new[(min(y + 1, 20), ban)] = new.get((min(y + 1, 20), ban), 0.0) + p * q
            new[(y, 0)] = new.get((y, 0), 0.0) + p * (1 - q)
        state = new
    info = dict(y=int(yc), need=nxt[0], by=nxt[1], ban=nxt[2], left=max(0, nxt[1] - (match_nos[0] - 1)) if match_nos else 0,
                risk=round(started, 3)) if nxt else None
    return avail, started, info


def project(ds, tm, rates, prof, gws, keep_detail=True, rot=None, simulate_gws=False):
    rot = rot or {"fixture": {}, "player": {}}
    rng = np.random.default_rng(2026)
    fx = ds.fixtures
    upcoming = fx[(fx["event"].isin(gws)) & ~fx["finished"]]
    gw_dates = {g: upcoming.loc[upcoming.event == g, "kickoff"].min().to_pydatetime().replace(tzinfo=None)
                for g in gws if (upcoming.event == g).any() and upcoming.loc[upcoming.event == g, "kickoff"].notna().any()}
    by_team = {g: {} for g in gws}
    # each club's league match number for every upcoming fixture, for the booking-ban rules
    played = {}
    for r in fx[fx["finished"]].itertuples():
        played[r.team_h] = played.get(r.team_h, 0) + 1
        played[r.team_a] = played.get(r.team_a, 0) + 1
    order = {}
    for r in fx[~fx["finished"]].sort_values(["kickoff", "event"], na_position="last").itertuples():
        for t_ in (r.team_h, r.team_a):
            order[t_] = order.get(t_, 0) + 1
            order[(t_, r.id)] = played.get(t_, 0) + order[t_]
    yellows = ds.hist.groupby("element")["yellow_cards"].sum().to_dict() if "yellow_cards" in ds.hist else {}
    fx_id = {}
    for r in upcoming.itertuples():
        fx_id.setdefault((r.event, r.team_h), []).append(r.id)
        fx_id.setdefault((r.event, r.team_a), []).append(r.id)
    for r in upcoming.sort_values("kickoff").itertuples():
        lh, la = blended_lambdas(tm, r.team_h, r.team_a)
        by_team[r.event].setdefault(r.team_h, []).append((r.team_a, True, lh, la))
        by_team[r.event].setdefault(r.team_a, []).append((r.team_h, False, la, lh))
    mu = (tm["mu_h"] + tm["mu_a"]) / 2
    out = []
    for p in ds.players.to_dict("records"):
        pid, team, pos = int(p["id"]), int(p["team"]), int(p["element_type"])
        r = rates.get(pid) or default_rates(pos)
        if str(p.get("status")) == "u" and r["season"] is None:
            continue
        po = p.get("penalties_order")
        r = dict(r, taker=po is not None and not pd.isna(po) and int(po) == 1, pen_rate=tm.get("pen", {}).get(team, 0.0))
        av = availability(p, gw_dates, gws)
        neutral, _ = fixture_points(r, mu, mu, 1.0, 1.0)
        neutral_ret = sum(v for k, v in neutral.items() if k not in ("app", "cards", "gc")) or 1e-6
        ep, nxt, sims, rot_notes, epf = [], None, [], [], []
        comp6 = {}
        rf = dict(r, p60=1.0, p1=0.0, emins=88.0)      # the same player if he starts and plays 88 minutes
        # booking-ban risk, fixture by fixture
        slots = [(i, g, k) for i, g in enumerate(gws) for k in range(len(by_team[g].get(team, [])))]
        mnos = [order.get((team, (fx_id.get((g, team)) or [None])[min(k, len(fx_id.get((g, team)) or [None]) - 1)]), 0) or 99 for i, g, k in slots]
        b_av, _, cards = ban_chances(yellows.get(pid, 0), r["yc90"], [r["emins"] * av[i] for i, g, k in slots], mnos,
                                     risk_slots=sum(1 for i, g, k in slots if i < HORIZON))
        ban_av = {(i, k): v for (i, g, k), v in zip(slots, b_av)}
        for i, g in enumerate(gws):
            total, total_full = 0.0, 0.0
            sim_total = np.zeros(SIM_N) if simulate_gws else None
            for k_fx, (opp, home, lf, la) in enumerate(by_team[g].get(team, [])):
                cong = rot["fixture"].get((team, g, opp))
                f_rot, note = rotation_factor(rot["player"].get(pid), cong, r)
                rr = dict(r, p60=min(0.97, r["p60"] * f_rot), emins=min(87.0, r["emins"] * f_rot)) if f_rot > 1.001 else r if f_rot >= 0.999 else dict(r, p60=r["p60"] * f_rot,
                                                   p1=min(1.0, r["p1"] + r["p60"] * (1 - f_rot) * 0.5),
                                                   emins=r["emins"] * (f_rot + (1 - f_rot) * 0.3))
                if note and i == 0:
                    rot_notes.append(note)
                sf, drivers = style_factor(r["profile"], prof.get(opp, {}).get("def"))
                ref = tm.get("refs", {}).get((team, opp) if home else (opp, team))
                extra = 0.0
                pen_order = p.get("penalties_order")
                if ref and ref.get("league") and pen_order is not None and not pd.isna(pen_order) and int(pen_order) == 1:
                    extra = (ref["pens"] - ref["league"]) * lf / max(lf + la, 1e-6) * PEN_XG
                pts, det = fixture_points(rr, lf, la, tm["gen"][opp], av[i], sf, extra)
                ret = sum(v for k, v in pts.items() if k not in ("app", "cards", "gc"))
                quality = min(1.6, max(0.5, ret / (neutral_ret * max(av[i], 1e-6))))
                pts["bonus"] = rr["bonus90"] * rr["emins"] * av[i] / 90 * quality
                if USE_AVAIL_DECAY and i >= 1:   # injuries, bans, rotation and dropped players add up the further ahead you look
                    fa = max(AVAIL_FLOOR, 1 - AVAIL_DECAY.get(pos, 0.04) * min(i, 5))
                    pts = {k_: v_ * fa for k_, v_ in pts.items()}
                bf = ban_av.get((i, k_fx), 1.0)
                if bf < 0.999:   # chance he's serving a booking ban for this match
                    pts = {k_: v_ * bf for k_, v_ in pts.items()}
                total += sum(pts.values())
                if i < HORIZON:
                    for k_, v_ in pts.items():
                        comp6[k_] = comp6.get(k_, 0.0) + v_
                    pf, _ = fixture_points(rf, lf, la, tm["gen"][opp], 1.0, sf, extra)
                    qf = min(1.6, max(0.5, sum(v for k, v in pf.items() if k not in ("app", "cards", "gc")) / neutral_ret))
                    total_full += sum(pf.values()) + r["bonus90"] * 88 / 90 * qf
                if simulate_gws and ((i < HORIZON and (i == 0 or ep_hint(r))) or (i >= HORIZON and tc_candidate(r))):
                    sim_total += simulate_fixture(rr, lf, la, det["dc_cond"], pts["bonus"], av[i], sf, rng)
                if i == 0 and nxt is None and keep_detail:
                    nxt = dict(det, opp=opp, home=home, drivers=drivers, rot=cong, rot_factor=f_rot, ref=ref, rested=bool(note and note.get("rested")),
                               breakdown={k: round(v, 2) for k, v in pts.items()})
            ep.append(round(total, 2))
            if i < HORIZON:
                epf.append(round(total_full, 2))
            if simulate_gws:
                sims.append(sim_total)
        out.append(dict(p=p, r=r, ep=ep, av=av, next=nxt, sims=sims, rot_notes=rot_notes, epf=epf, cards=cards,
                        comp6={k_: round(v_, 2) for k_, v_ in comp6.items()}))
    return out


def tc_candidate(r):
    """Players worth simulating for Triple Captain planning beyond the six-week window."""
    return r["emins"] >= 60 and r["share_g"] + r["share_a"] >= 0.2


def blended_lambdas(tm, h, a):
    """Model expected goals, blended with bookmaker-implied figures where the odds exist."""
    mh, ma = fixture_lambdas(tm, h, a)
    book = tm.get("book", {}).get((h, a))
    if not book:
        return mh, ma
    return BOOK_WEIGHT * book["lh"] + (1 - BOOK_WEIGHT) * mh, BOOK_WEIGHT * book["la"] + (1 - BOOK_WEIGHT) * ma


def ep_hint(r):
    """Only simulate later gameweeks for players who might plausibly matter (keeps runs fast)."""
    return r["emins"] >= 30


def simulate_fixture(r, lf, la, dc_cond, bonus_exp, avail, style, rng, n=SIM_N):
    """One fixture played n times: minutes, goals, assists, clean sheet, saves, DEFCON, cards, bonus."""
    pos = r["pos"]
    p60, p1 = r["p60"] * avail, r["p1"] * avail
    u = rng.random(n)
    full, sub = u < p60, (u >= p60) & (u < p60 + p1)
    m_sub = min(59.0, max(10.0, (r["emins"] - 87 * r["p60"]) / r["p1"])) if r["p1"] > 0.02 else 25.0
    mins = np.where(full, 87.0, np.where(sub, m_sub, 0.0))
    pen_team = r.get("pen_rate", 0.0) * min(1.6, lf / 1.35)
    g = rng.poisson(r["share_g"] * max(0.05, lf - pen_team) * style * mins / 90) + (rng.poisson(pen_team * mins / 90) if r.get("taker") else 0)
    a = rng.poisson(r["share_a"] * lf * mins / 90)
    conc = rng.poisson(la, n)
    cs = full & (conc == 0)
    pts = np.where(full, 2, np.where(sub, 1, 0)) + GOAL[pos] * g + ASSIST * a + CLEAN_SHEET[pos] * cs
    if pos <= 2:
        pts = pts - np.where(full, conc // 2, 0)
    if pos == 1:
        pts = pts + np.where(full, rng.poisson(r["save_ratio"] * la, n) // 3, 0)
    if pos in DEFCON_THRESHOLD:
        pts = pts + DEFCON_POINTS * (full & (rng.random(n) < dc_cond))
    pts = pts - (rng.random(n) < r["yc90"] * mins / 90)
    w = (1 + 2 * g + a + (cs & (pos <= 2))) * (mins > 0)
    k = bonus_exp / w.mean() if w.mean() > 0 else 0.0
    return pts + np.minimum(3, rng.poisson(k * w))


def sim_summary(s):
    if s is None or not len(s):
        return None
    bins = [-99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 99]   # last bins widen
    hist = np.histogram(s, bins=[b + 0.5 for b in bins])[0] / len(s)
    return dict(mean=round(float(s.mean()), 2), haul=round(float((s >= 10).mean()), 3),
                blank=round(float((s <= 2).mean()), 3), ret=round(float((s >= 4).mean()), 3),
                q=[int(np.percentile(s, 10)), int(np.percentile(s, 50)), int(np.percentile(s, 90))],
                dist=[round(float(x), 3) for x in hist])


# ============================================================ rotation
def rotation_model(ds, gws):
    """Fixture congestion from cup and European games, and how each player's starts react to it."""
    cups = ds.cups
    other = {t: g.sort_values("kickoff") for t, g in cups.groupby("team")} if len(cups) else {}

    def around(team, ko):
        g = other.get(team)
        if g is None or pd.isna(ko):
            return None
        before = g[(g.kickoff < ko) & (g.kickoff >= ko - CONGESTED_BEFORE)]
        after = g[(g.kickoff > ko) & (g.kickoff <= ko + CONGESTED_AFTER)]
        if before.empty and after.empty:
            return None
        pick = before.iloc[-1] if len(before) else after.iloc[0]
        days = (ko - pick.kickoff).total_seconds() / 86400
        return dict(comp=pick.comp, ko=pick.kickoff.isoformat(), days=round(days, 1), before=bool(len(before)))

    fixture = {}
    for r in ds.fixtures[ds.fixtures["event"].isin(gws) & ~ds.fixtures["finished"]].itertuples():
        for team, opp in ((r.team_h, r.team_a), (r.team_a, r.team_h)):
            c = around(team, r.kickoff)
            if c:
                fixture[(team, r.event, opp)] = c

    player = {}
    if len(ds.hist):
        flags = {}
        for r in ds.fixtures[ds.fixtures["finished"]].itertuples():
            for team in (r.team_h, r.team_a):
                flags[(r.id, team)] = around(team, r.kickoff) is not None
        h = ds.hist.assign(cong=[flags.get((f, t), False) for f, t in zip(ds.hist.fixture, ds.hist.team)])
        h["started"] = (h["starts"] > 0) | (h["minutes"] >= 60)
        for pid, g in h.groupby("element"):
            c, n = g[g.cong], g[~g.cong]
            player[pid] = dict(n_c=int(len(c)), s_c=int(c.started.sum()), n_n=int(len(n)), s_n=int(n.started.sum()))
        cm = ds.cup_mins[ds.cup_mins["kickoff"].notna()].sort_values("kickoff")
        for pid, g in cm.groupby("player"):
            last = g.iloc[-1]
            if pid in player:
                player[pid]["cup_by_day"] = {k.date().isoformat(): int(m) for k, m in zip(g.kickoff, g.minutes)}
            if pid in player:
                player[pid].update(last_cup=dict(comp=last.comp, mins=int(last.minutes), ko=last.kickoff.isoformat()),
                                   cup_starts=int((g.minutes >= 60).sum()), cup_games=int((g.minutes > 0).sum()))
    return dict(fixture=fixture, player=player)


RESTED_BOOST = 0.0   # tested on this season: no boost 2.080 MAE, a quarter 2.088, a half 2.099; so rested players just lose the flag


def rotation_factor(ph, cong, r):
    """Scale a player's chance of starting when his team has a cup or European game either side.
    If that game was before the league match and has already been played, what he actually did decides it:
    rested (under 30 minutes) means fresh legs and a likelier start; a full game keeps the usual risk."""
    if not cong or not ph or r["p60"] < 0.3:
        return 1.0, None
    if cong.get("before"):
        ko = pd.Timestamp(cong["ko"])
        if ko < pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=3):
            mins = (ph.get("cup_by_day") or {}).get(ko.date().isoformat(), 0)
            if mins < 30:
                target = min(0.97, r["p60"] + (1 - r["p60"]) * RESTED_BOOST)
                f = target / max(r["p60"], 0.01)
                return f, dict(cong, rested=True, mins=mins, f=round(f, 2), n_c=ph["n_c"], s_c=ph["s_c"], n_n=ph["n_n"], s_n=ph["s_n"])
    rate_n = ph["s_n"] / ph["n_n"] if ph["n_n"] else None
    if ph["n_c"] >= 2 and rate_n:
        rate_c = ph["s_c"] / ph["n_c"]
        f = min(1.0, max(0.6, rate_c / max(rate_n, 0.1)))
    else:
        f = 0.97
    return f, dict(cong, n_c=ph["n_c"], s_c=ph["s_c"], n_n=ph["n_n"], s_n=ph["s_n"], f=round(f, 2),
                   last_cup=ph.get("last_cup"))


# ========================================================== explanations
def ordinal(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def most_phrase(rank, n=20):
    return "the most in the league" if rank == 1 else f"the {ordinal(rank)}-most" if rank <= n / 2 else \
        ("the fewest in the league" if rank == n else f"the {ordinal(n + 1 - rank)}-fewest")


def explain(item, teams, ttab, prof, pct, tm=None):
    p, r, nx = item["p"], item["r"], item["next"]
    if not nx or r["season"] is None:
        return []
    pid, pos, name = int(p["id"]), r["pos"], _text(p.get("web_name"))
    team, opp = int(p["team"]), nx["opp"]
    T, O = teams[team], teams[opp]
    tt, ot = ttab[team], ttab[opp]
    venue_for = "at home" if nx["home"] else "away"
    ctx = (tm or {}).get("ctx", {}) if tm is not None else {}
    why_opp = lambda side: (" Why: " + ctx[opp][side + "_short"]) if ctx.get(opp, {}).get(side + "_short") else ""
    why_own = lambda side: (" Why: " + ctx[team][side + "_short"]) if ctx.get(team, {}).get(side + "_short") else ""
    s = r["season"]
    pc = pct.get(pid, {})
    why = []

    def add(tone, head, text, stat=None, weight=1.0):
        why.append(dict(t=tone, h=head, s=text, st=stat, w=weight))

    def pstat(key, label):
        q = pc.get(key)
        return dict(p=q["p"], label=label, rank=q["rank"], of=q["of"]) if q and q["ranked"] else None

    attacker = pos >= 3 or r["share_g"] >= 0.08
    if attacker and s["mins"] >= 180:
        q = pc.get("npxg90")
        tone = "up" if q and q["p"] >= 75 else "down" if q and q["p"] <= 30 else "flat"
        rank_txt = f", {ordinal(q['rank'])} of {q['of']} {POS_NAME[pos]}" if q and q["ranked"] else ""
        add(tone, "Shot quality", f"{s['npxg90']:.2f} non-penalty xG per 90{rank_txt}. "
            f"He takes {r['share_g']:.0%} of {T['name']}’s non-penalty expected goals when he’s on the pitch.",
            pstat("npxg90", "npxG per 90"), (3 if pos >= 3 else 1.5) if tone == "up" else 1.2)
        if s["box90"] > 0:
            add("up" if (pc.get("box90") or {}).get("p", 0) >= 80 else "flat", "Box presence",
                f"{s['box90']:.1f} touches in the opposition box per 90 and {s['shots90']:.1f} shots per 90.",
                pstat("box90", "Box touches per 90"), 1.2)
        if s["xa90"] >= 0.12:
            add("up" if (pc.get("xa90") or {}).get("p", 0) >= 75 else "flat", "Creativity",
                f"{s['xa90']:.2f} xA per 90 from {s['chances90']:.1f} chances created per 90.",
                pstat("xa90", "xA per 90"), 1.2)
        if s.get("xgi90_last4") is not None:
            season_xgi = s["xg90"] + s["xa90"]
            if season_xgi > 0.1:
                ch = s["xgi90_last4"] / season_xgi - 1
                if abs(ch) >= 0.2:
                    add("up" if ch > 0 else "down", "Recent form",
                        f"{s['xgi90_last4']:.2f} xG plus xA per 90 over his last four appearances, "
                        f"{'up' if ch > 0 else 'down'} {abs(ch):.0%} on his season rate of {season_xgi:.2f}.",
                        None, 1.4)

    if pos <= 2:
        cs = nx["cs"]
        tone = "up" if cs >= 0.38 else "down" if cs <= 0.22 else "flat"
        add(tone, "Clean sheet chance",
            f"{cs:.0%} chance of a clean sheet. {O['name']} create {ot['xgf']:.2f} xG per match, "
            f"{most_phrase(ot['r_xgf'])}; the model expects {nx['la']:.2f} against {T['short']} {venue_for}." + why_opp("att"),
            None, 2.5)
        if tt.get("xga") is not None:
            add("up" if tt["r_xga"] <= 6 else "down" if tt["r_xga"] >= 15 else "flat", "Team defence",
                f"{T['name']} concede {tt['xga']:.2f} xG per match, {most_phrase(21 - tt['r_xga'])}." + why_own("def"),
                None, 1.2)
    if pos in DEFCON_THRESHOLD and r["dc_games"] >= 2 and (pos != 4):
        gen = O.get("gen", 1.0)
        tone = "up" if nx["dc_p"] >= 0.45 else "flat"
        add(tone, "DEFCON",
            f"Hit the threshold in {r['dc_hits']} of {r['dc_games']} full matches ({s['dc90']:.1f} per 90). "
            f"{O['name']} make opponents do {abs(gen - 1):.0%} {'more' if gen >= 1 else 'less'} defending than average, "
            f"so {nx['dc_p']:.0%} chance this week.", pstat("dc90", "DEFCON per 90"), 1.6 if tone == "up" else 0.8)
    if pos == 1:
        add("flat", "Saves", f"{O['name']} take {ot['xgf']:.2f} xG per match; he makes about "
            f"{r['save_ratio']:.1f} saves for every 1 xG faced.", None, 1.0)

    if attacker:
        add("up" if nx["lf"] >= 1.9 else "down" if nx["lf"] <= 1.1 else "flat", "Fixture",
            f"{T['name']} are expected to score {nx['lf']:.2f} {venue_for} against {O['name']}. "
            f"{O['name']} concede {ot['xga']:.2f} xG per match, {most_phrase(21 - ot['r_xga'])}; "
            f"{T['short']} create {tt['xgf']:.2f}, {most_phrase(tt['r_xgf'])}." + why_opp("def"),
            None, 2.2 if nx["lf"] >= 1.9 or nx["lf"] <= 1.1 else 1.0)

    # situational matchup: the strongest positive and negative style links
    od = prof.get(opp, {}).get("def", {})
    if r["profile"] and od and r["np_xg"] >= 0.6:
        picks = nx["drivers"][-1:] + [d for d in nx["drivers"][:1] if d not in nx["drivers"][-1:]]
        for contribution, c in picks:
            info = od[c]
            share = r["profile"][c]
            if contribution > 0.04 and info["idx"] >= 1.15:
                add("up", f"Matchup: {CAT_SHORT[c].lower()}",
                    f"{share:.0%} of his non-penalty xG comes from {CATS[c]}. {O['name']} concede "
                    f"{info['pm']:.2f} xG per match from {CATS[c]}, {info['idx']:.1f}× the league average "
                    f"({most_phrase(info['most'])}).", dict(p=round(100 * (20 - info['most']) / 19),
                                                           label=f"{O['short']} concede from {CAT_SHORT[c].lower()}",
                                                           rank=info["most"], of=20), 2.6)
            elif contribution < -0.04 and info["idx"] <= 0.85:
                add("down", f"Matchup: {CAT_SHORT[c].lower()}",
                    f"{share:.0%} of his non-penalty xG comes from {CATS[c]}, but {O['name']} only concede "
                    f"{info['pm']:.2f} per match there, {info['idx']:.1f}× the league average "
                    f"({most_phrase(info['most'])}).", None, 2.0)
        if abs(nx["style"] - 1) >= 0.02:
            why.append(dict(t="up" if nx["style"] > 1 else "down", h="Style matchup",
                            s=f"Crossing his shot profile with what {O['name']} concede moves his expected goals "
                              f"{abs(nx['style'] - 1):.0%} {'up' if nx['style'] > 1 else 'down'} for this fixture.",
                            st=None, w=0.9))

    for note in item.get("rot_notes", [])[:1]:
        when = pd.Timestamp(note["ko"]).strftime("%a %-d %b")
        side = f"{abs(note['days']):.0f} days {'before' if note['before'] else 'after'}"
        hist_txt = (f" After those games he has started {note['s_c']} of {note['n_c']} league matches, against "
                    f"{note['s_n']} of {note['n_n']} otherwise." if note["n_c"] >= 2 else "")
        last = note.get("last_cup")
        cup_txt = f" He played {last['mins']} minutes in their last {last['comp']} match." if last else ""
        add("down" if note["f"] < 0.9 else "flat", "Rotation risk",
            f"{T['name']} play in the {note['comp']} on {when}, {side} this match.{hist_txt}{cup_txt} " +
            (f"The projection cuts his chance of starting by {1 - note['f']:.0%}." if note["f"] < 0.995
             else "That history says he isn’t rotated, so the projection leaves him alone."),
            None, 2.4 if note["f"] < 0.9 else 1.0)
    if s.get("goals") is not None and s["mins"] >= 450:
        gap = s["goals"] - s.get("xg_total", 0)
        if abs(gap) >= 1.8:
            add("flat", "Finishing", f"{s['goals']} goals from {s['xg_total']:.1f} xG this season "
                f"({'+' if gap > 0 else '−'}{abs(gap):.1f}). The model works from chances, not goals, so it "
                f"{'expects him to score less often from here' if gap > 0 else 'thinks the goals should come'}.", None, 1.1)

    ref = nx.get("ref")
    if ref and nx.get("pen_extra") and abs(nx["pen_extra"]) >= 0.005:
        hot = ref["pens"] >= 0.28
        add("up" if nx["pen_extra"] > 0 else "down", "Referee",
            f"{ref['name']} gives {ref['pens']:.2f} penalties a game over {ref['n']} matches (league {ref['league']:.2f}). "
            f"As first-choice taker that moves his expected goals {'up' if nx['pen_extra'] > 0 else 'down'} "
            f"{abs(nx['pen_extra']):.2f}.", None, 1.8 if hot else 0.9)
    book = (tm or {}).get("book", {}).get((team, opp) if nx["home"] else (opp, team))
    if book:
        bf, ba = (book["lh"], book["la"]) if nx["home"] else (book["la"], book["lh"])
        mf, ma_ = fixture_lambdas(tm, team, opp) if nx["home"] else fixture_lambdas(tm, opp, team)[::-1]
        win = book["ph"] if nx["home"] else book["pa"]
        add("up" if bf > mf + 0.15 else "down" if bf < mf - 0.15 else "flat", "Bookmakers",
            f"The odds give {T['name']} a {win:.0%} chance of winning and {bf:.2f} expected goals "
            f"({ba:.2f} against). The model alone said {mf:.2f}; the projection blends the two "
            f"{BOOK_WEIGHT:.0%} to {1 - BOOK_WEIGHT:.0%} in favour of the market.", None, 1.6)

    pen = p.get("penalties_order")
    if pen is not None and not pd.isna(pen) and int(pen) >= 2 and r.get("pen_xg", 0) >= 0.7:
        add("down", "Penalties", f"He’s scored or taken penalties this season ({r['pen_xg']:.1f} xG from the spot), but FPL now lists him "
            f"as {ordinal(int(pen))} choice, so the projection gives him no penalty threat.", None, 2.2)
    if pen is not None and not pd.isna(pen) and int(pen) == 1:
        add("up", "Penalties", "First-choice penalty taker." +
            (f" Penalties have added {r['pen_xg']:.2f} xG this season." if r["pen_xg"] > 0 else ""), None, 1.3)
    corners = p.get("corners_and_indirect_freekicks_order")
    if corners is not None and not pd.isna(corners) and int(corners) == 1:
        add("up", "Set pieces", "On corners and indirect free kicks.", None, 0.8)

    av = item["av"][0]
    if av < 1:
        add("down", "Availability", f"{_text(p.get('news')) or 'Flagged'}. The projection assumes a {av:.0%} chance he plays.",
            None, 3)
    if s["games"]:
        if r["p60"] < 0.7 and av >= 1:
            add("down", "Minutes risk", f"Started {s['starts']} of {s['games']} matches; "
                f"{r['emins']:.0f} expected minutes this week.", None, 2.2)
        elif r["p60"] >= 0.9:
            add("up", "Nailed", f"Started {s['starts']} of {s['games']} matches; {r['emins']:.0f} expected minutes.",
                None, 0.7)

    # home and away form: always part of the reasoning
    tv, ov = tt.get("home" if nx["home"] else "away"), ot.get("away" if nx["home"] else "home")
    if tv and ov and tm is not None:
        here, there = ("at home", "away from home") if nx["home"] else ("away from home", "at home")
        side_t, side_o = ("home", "away") if nx["home"] else ("away", "home")
        adv = (tm["mu_h"] / tm["mu_a"] - 1)
        note = (f" The model counts normal home advantage (home sides create {adv:.0%} more), but not each team’s own "
                f"split: a few games at one venue are mostly noise, and using them made past predictions worse.")
        rec = lambda v: f"{v['w']}W {v['d']}D {v['l']}L"
        lf5 = lambda v: f"last five {v['form']}" if v["n"] >= 5 else f"form {v['form']}"
        lead = (f"{T['name']} {here}: {rec(tv)} from {tv['n']} ({lf5(tv)}), with {tv['xgf']:.2f} xG for and "
                f"{tv['xga']:.2f} against per game. {O['name']} {there}: {rec(ov)} ({lf5(ov)}), ")
        if pos <= 2:
            body = f"creating {ov['xgf']:.2f} xG per game, {most_phrase(ov['r_xgf'])} of any {side_o} side."
            good = tv["r_xga"] <= 6 or ov["r_xgf"] >= 15
            bad = tv["r_xga"] >= 15 or ov["r_xgf"] <= 6
        else:
            body = f"conceding {ov['xga']:.2f} xG per game, {most_phrase(21 - ov['r_xga'])} of any {side_o} side."
            good = tv["r_xgf"] <= 6 or ov["r_xga"] >= 15
            bad = tv["r_xgf"] >= 15 or ov["r_xga"] <= 6
        tone = "up" if good and not bad else "down" if bad and not good else "flat"
        add(tone, "Home and away form", lead + body + note, None, 2.05)

    why.sort(key=lambda x: -x["w"])
    for x in why:
        x.pop("w", None)
    keep = why[:8]
    venue = next((x for x in why if x["h"] == "Home and away form"), None)
    if venue and venue not in keep:
        keep = keep[:7] + [venue]
    return keep


# ============================================================ accuracy
TEAM_MINUTES_CAP = True     # a team only has 11 x 90 = 990 minutes to share; see cap_team_minutes()
TEAM_MINUTES = 990.0


def cap_team_minutes(ds, rates, cap=TEAM_MINUTES):
    """Each player's expected minutes are estimated on their own, so a team's total can come to more than the 990
    minutes that exist (three strikers each expected to play 70 minutes for one place). Where a team's available
    players add up to more than 990, scale their expected minutes, and their chances of playing, down to fit."""
    if not TEAM_MINUTES_CAP:
        return {}
    pl = ds.players
    status = dict(zip(pl["id"].astype(int), pl["status"].fillna("a"))) if "status" in pl else {}
    team = dict(zip(pl["id"].astype(int), pl["team"].astype(int)))
    total = {}
    for pid, r in rates.items():
        if status.get(int(pid), "a") in ("i", "s", "u", "n") or int(pid) not in team:
            continue
        total[team[int(pid)]] = total.get(team[int(pid)], 0.0) + r["emins"]
    scaled = {t: cap / v for t, v in total.items() if v > cap}
    for pid, r in rates.items():
        f = scaled.get(team.get(int(pid)))
        if f:
            r["emins"] *= f
            r["p60"] *= f
            r["p1"] = min(1.0, r["p1"] * f)
    return {int(t): round(f, 3) for t, f in scaled.items()}


def fit_and_project(ds, gws, detail=False, simulate=False, book=None, refs=None):
    tm = team_model(ds)
    tm["book"] = book or {}
    tm["refs"] = refs or {}
    prof, league = team_profiles(ds, tm)
    rates = player_model(ds, tm)
    tm["minutes_capped"] = cap_team_minutes(ds, rates)
    rot = rotation_model(ds, gws)
    return tm, prof, league, rates, rot, project(ds, tm, rates, prof, gws, keep_detail=detail, rot=rot,
                                                 simulate_gws=simulate)


def actual_points(ds, k):
    actual = ds.hist[ds.hist["event"] == k].groupby("element").agg(pts=("event_points", "sum"), mins=("minutes", "sum"))
    done = ds.fixtures[ds.fixtures["finished"] & (ds.fixtures["event"] == k)]
    return actual, set(done["team_h"]) | set(done["team_a"])


def score_week(df, k):
    """Compare one gameweek's projections with what happened. df: id, n, pos, ours, fpl, pts, mins."""
    pool = df[(df.ours >= 1) | (df.fpl.fillna(0) >= 1)]
    both = pool.dropna(subset=["fpl"])
    played = both[both.mins > 0]
    best = df.sort_values("pts", ascending=False).iloc[0]
    cap = df.sort_values("ours", ascending=False).iloc[0]
    fcap = both.sort_values("fpl", ascending=False).iloc[0] if len(both) else None
    by_pos = {}
    for pos in (1, 2, 3, 4):
        b = both[both.pos == pos]
        if len(b) >= 5:
            by_pos[pos] = [round(float((b.ours - b.pts).abs().mean()), 2), round(float((b.fpl - b.pts).abs().mean()), 2)]
    return dict(
        gw=k, n=len(both),
        mae=round(float((both.ours - both.pts).abs().mean()), 3),
        mae_fpl=round(float((both.fpl - both.pts).abs().mean()), 3) if len(both) else None,
        corr=round(float(played.ours.corr(played.pts, method="spearman")), 3) if len(played) > 5 else None,
        corr_fpl=round(float(played.fpl.corr(played.pts, method="spearman")), 3) if len(played) > 5 else None,
        top10=round(float(df.nlargest(10, "ours").pts.mean()), 2),
        top10_fpl=round(float(both.nlargest(10, "fpl").pts.mean()), 2) if len(both) else None,
        cap=dict(n=cap.n, proj=round(cap.ours, 1), pts=int(cap.pts)),
        cap_fpl=dict(n=fcap.n, proj=round(fcap.fpl, 1), pts=int(fcap.pts)) if fcap is not None else None,
        best=dict(n=best.n, pts=int(best.pts)), by_pos=by_pos,
        over=[dict(n=r.n, proj=round(r.ours, 1), pts=int(r.pts)) for r in pool.assign(d=pool.ours - pool.pts).nlargest(3, "d").itertuples()],
        under=[dict(n=r.n, proj=round(r.ours, 1), pts=int(r.pts)) for r in pool.assign(d=pool.pts - pool.ours).nlargest(3, "d").itertuples()],
    ), pool


def summarise(weeks):
    if not weeks:
        return {}
    W = pd.DataFrame(weeks)
    return dict(
        gws=[int(W.gw.min()), int(W.gw.max())], n=len(W),
        mae=round(float(W.mae.mean()), 3), mae_fpl=round(float(W.mae_fpl.mean()), 3),
        corr=round(float(W["corr"].mean()), 3), corr_fpl=round(float(W["corr_fpl"].mean()), 3),
        top10=round(float(W.top10.mean()), 2), top10_fpl=round(float(W.top10_fpl.mean()), 2),
        cap=round(float(np.mean([w["cap"]["pts"] for w in weeks])), 2),
        cap_fpl=round(float(np.mean([w["cap_fpl"]["pts"] for w in weeks if w["cap_fpl"]])), 2),
        best=round(float(np.mean([w["best"]["pts"] for w in weeks])), 2),
        wins=int(sum(1 for w in weeks if w["mae_fpl"] is not None and w["mae"] < w["mae_fpl"])),
    )


def backtest(ds, first=4):
    last = int(ds.fixtures.loc[ds.fixtures["finished"], "event"].max())
    weeks, calib, calib_odds, stack = [], [], [], []
    for k in range(first, last + 1):
        cut = at_cutoff(ds, k - 1)
        if cut.players.empty:
            continue
        tm_k, *_, proj = fit_and_project(cut, [k], detail=True)
        for r in ds.fixtures[ds.fixtures["finished"] & (ds.fixtures["event"] == k)].rename(columns={"as": "as_"}).itertuples():
            row = getattr(ds, "results_odds", {}).get(r.id)
            lam = closing_lambdas(row) if row else None
            if lam:
                mh, ma = fixture_lambdas(tm_k, r.team_h, r.team_a)
                calib_odds.append((mh, lam[0], int(r.hs)))
                calib_odds.append((ma, lam[1], int(r.as_)))
        actual, played_teams = actual_points(ds, k)
        rows = []
        for it in proj:
            pid = int(it["p"]["id"])
            if int(it["p"]["team"]) not in played_teams:
                continue
            a = actual.loc[pid] if pid in actual.index else None
            fpl = it["p"].get("ep_next")
            rows.append(dict(id=pid, n=_text(it["p"].get("web_name")), pos=it["r"]["pos"], ours=it["ep"][0],
                             fpl=float(fpl) if fpl is not None and not pd.isna(fpl) else None,
                             pts=float(a["pts"]) if a is not None else 0.0, mins=float(a["mins"]) if a is not None else 0.0))
            nx = it.get("next") or {}
            if it["ep"][0] >= 0.5 or (rows[-1]["fpl"] or 0) >= 0.5:
                stack.append([k, round(it["ep"][0], 2), rows[-1]["fpl"], rows[-1]["pts"],
                              round(nx.get("xg", 0) + nx.get("xa", 0), 3), it["r"]["pos"], pid])
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        week, _ = score_week(df, k)
        weeks.append(week)
        calib.append(df[["ours", "pts", "fpl"]])
    cal = []
    if calib:
        c = pd.concat(calib)
        for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 99)]:
            b = c[(c.ours >= lo) & (c.ours < hi)]
            if len(b) >= 5:
                cal.append(dict(lo=lo, hi=hi if hi < 99 else None, n=len(b), proj=round(float(b.ours.mean()), 2),
                                actual=round(float(b.pts.mean()), 2)))
    return dict(summary=summarise(weeks), weeks=weeks, calibration=cal, odds_rows=calib_odds, stack_rows=stack)


# ============================================================ FPL Lab score
KAPPAS = [0, 0.5, 1, 1.5, 2, 3, 4]


def disagreement_record(rows, gap=1.5):
    """When our projection and FPL's differ by `gap` or more, who was closer to what the player actually scored?
    Split by whether we were above or below FPL. rows: backtest stack rows [gw, ours, fpl, pts, xgi, pos, id]."""
    rs = [r for r in rows if r[2] is not None and abs(r[1] - r[2]) >= gap]
    def part(sel):
        sel = [r for r in sel if abs(r[3] - r[1]) != abs(r[3] - r[2])]
        if not sel:
            return None
        closer = sum(abs(r[3] - r[1]) < abs(r[3] - r[2]) for r in sel) / len(sel)
        mean = lambda k: round(sum(r[k] for r in sel) / len(sel), 2)
        return dict(n=len(sel), closer=round(closer, 3), actual=mean(3), ours=mean(1), fpl=mean(2))
    return dict(gap=gap, all=part(rs), higher=part([r for r in rs if r[1] > r[2]]), lower=part([r for r in rs if r[1] < r[2]]))


def lab_tuning(rows):
    """Our own points system, learned from the backtests.
    1. Lab points: a blend of the structural model and FPL's own expected points, with the weight
       that best predicted actual points.
    2. Lab captain score: Lab points plus kappa x expected goal involvements. Goals and assists are
       where hauls come from, so this rewards upside. Kappa is the value whose top pick each
       gameweek scored the most actual points."""
    df = pd.DataFrame(rows, columns=["gw", "ours", "fpl", "pts", "xgi", "pos", "id"])
    df = df[df.fpl.notna()]
    if df.gw.nunique() < 5:
        return None
    best_w, best_err, errs = 0.0, None, {}
    for w in [x / 20 for x in range(0, 11)]:
        pred = (1 - w) * df.ours + w * df.fpl
        e = float((pred - df.pts).abs().mean())      # average error, the measure the app reports
        errs[w] = e
        if best_err is None or e < best_err - 1e-9:
            best_w, best_err = w, e
    df["lab"] = (1 - best_w) * df.ours + best_w * df.fpl
    caps = {}
    for kappa in KAPPAS:
        picks = df.assign(score=df.lab + kappa * df.xgi).sort_values("score", ascending=False).groupby("gw").head(1)
        caps[kappa] = round(float(picks.pts.mean()), 3)
    # with ~35 captain picks, a gap of under half a point a week is one lucky pick, so keep the simplest setting
    # (no bonus) unless a bonus is clearly better
    top = max(caps.values())
    best_k = min(k for k in caps if caps[k] >= top - 0.5)
    fpl_picks = df.sort_values("fpl", ascending=False).groupby("gw").head(1)
    best_pick = df.sort_values("pts", ascending=False).groupby("gw").head(1)
    return dict(w_fpl=best_w, kappa=best_k, n=len(df), gws=int(df.gw.nunique()),
                mae_model=round(float((df.ours - df.pts).abs().mean()), 3),
                mae_lab=round(float((df.lab - df.pts).abs().mean()), 3),
                mae_fpl=round(float((df.fpl - df.pts).abs().mean()), 3),
                cap=caps, cap_fpl=round(float(fpl_picks.pts.mean()), 3), cap_best=round(float(best_pick.pts.mean()), 3))



# ======================================================== narratives
def _list_names(items):
    items = [x for x in items if x]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def narratives(ds, tm, ttab, rates):
    """Plain-English reasons behind the numbers: who a team has played, which games skew the averages,
    where the chances come from, red cards, missing players and recent trends. Same for players."""
    m = tm["m"]
    if m.empty:
        return {}, {}
    names = dict(zip(ds.teams["id"], ds.teams["name"]))
    short = dict(zip(ds.teams["id"], ds.teams["short_name"]))
    att, dfn, ids = tm["att"], tm["dfn"], tm["ids"]
    h = ds.hist.copy()
    h["red_cards"] = pd.to_numeric(h.get("red_cards", 0), errors="coerce").fillna(0)
    reds = h.groupby(["fixture", "team"])["red_cards"].sum().to_dict()
    pos_of = dict(zip(ds.players["id"], ds.players["element_type"]))
    pname = dict(zip(ds.players["id"], ds.players["web_name"]))
    shots = ds.shots
    pens_conceded = shots[shots["pen"]].groupby("opp").size().to_dict() if len(shots) else {}
    pens_won = shots[shots["pen"]].groupby("team").size().to_dict() if len(shots) else {}
    league_pens = np.mean([pens_conceded.get(t, 0) for t in ids]) if ids else 0

    # each team's games, from its own side
    games = {}
    for r in m.itertuples():
        games.setdefault(r.h, []).append(dict(fx=r.id, ev=r.ev, opp=r.a, home=True, xf=r.xh, xa=r.xa, gf=r.gh, ga=r.ga))
        games.setdefault(r.a, []).append(dict(fx=r.id, ev=r.ev, opp=r.h, home=False, xf=r.xa, xa=r.xh, gf=r.ga, ga=r.gh))
    tough_def = {t: np.mean([att[g["opp"]] for g in games.get(t, [])]) for t in ids if games.get(t)}   # attacks faced
    tough_att = {t: np.mean([dfn[g["opp"]] for g in games.get(t, [])]) for t in ids if games.get(t)}   # defences faced
    rank = lambda d, t, rev: 1 + sorted(d, key=lambda k: -d[k] if rev else d[k]).index(t)
    adj_def_rank = {t: 1 + sorted(ids, key=lambda k: dfn[k]).index(t) for t in ids}     # 1 = best defence
    adj_att_rank = {t: 1 + sorted(ids, key=lambda k: -att[k]).index(t) for t in ids}     # 1 = best attack

    team_ctx = {}
    for t in ids:
        g = sorted(games.get(t, []), key=lambda x: x["ev"])
        if len(g) < 3:
            continue
        tt = ttab[t]
        out = {"def": [], "att": [], "def_short": None, "att_short": None}
        for side in ("def", "att"):
            key, avg = ("xa", tt["xga"]) if side == "def" else ("xf", tt["xgf"])
            lines, scores = [], []
            # schedule
            if side == "def":
                faced, frank = tough_def[t], rank(tough_def, t, True)
                top = sorted(g, key=lambda x: -att[x["opp"]])[:3]
                raw_rank, adj = tt["r_xga"], adj_def_rank[t]
                kind = "attacks"
            else:
                faced, frank = tough_att[t], 21 - rank(tough_att, t, False)   # 1 = toughest defences faced
                top = sorted(g, key=lambda x: dfn[x["opp"]])[:3]
                raw_rank, adj = tt["r_xgf"], adj_att_rank[t]
                kind = "defences"
            who = _list_names([names[x["opp"]] for x in top])
            if frank <= 5:
                txt = (f"They’ve faced the {ordinal(frank) + '-' if frank > 1 else ''}toughest set of {kind} in the league so far, "
                       f"including {who}. ")
                txt += (f"Allowing for that, the model rates them {ordinal(adj)}, better than their raw "
                        f"{ordinal(raw_rank)} suggests." if adj < raw_rank - 1 else
                        f"Even allowing for that, the model still rates them {ordinal(adj)}.")
                lines.append(("schedule", txt)); scores.append(3 if abs(adj - raw_rank) >= 3 else 2)
            elif frank >= 16:
                easy = _list_names([names[x["opp"]] for x in sorted(g, key=lambda x: att[x["opp"]] if side == "def" else -dfn[x["opp"]])[:3]])
                txt = (f"Their fixtures have been kind: the {ordinal(21 - frank) + '-' if frank < 20 else ''}easiest set of {kind} "
                       f"faced, including {easy}. ")
                txt += (f"Allowing for that, the model rates them {ordinal(adj)}, not the {ordinal(raw_rank)} the raw numbers suggest."
                        if adj > raw_rank + 1 else f"Even so, the model still rates them {ordinal(adj)}.")
                lines.append(("schedule", txt)); scores.append(3 if abs(adj - raw_rank) >= 3 else 2)
            else:
                lines.append(("schedule", f"Their opponents so far have been about average in {kind[:-1] if kind == 'attacks' else 'defence'}, so the raw numbers are a fair guide: "
                                          f"the model rates them {ordinal(adj)}."))
                scores.append(0.5)
            # one game skewing the average
            big = max(g, key=lambda x: x[key])
            rest = [x[key] for x in g if x is not big]
            if big[key] >= 2.2 and big[key] / max(1e-6, sum(x[key] for x in g)) >= 0.22 and rest:
                lines.append(("skew", f"One game does a lot of the work: {big[key]:.1f} xG "
                                      f"{'against' if side == 'def' else 'for'} them "
                                      f"{'at home to' if big['home'] else 'away at'} {names[big['opp']]} in GW{big['ev']}. "
                                      f"Without it they average {np.mean(rest):.2f} a game rather than {avg:.2f}."))
                scores.append(2.5)
            # red cards
            own_reds = [x for x in g if reds.get((x["fx"], t), 0) > 0]
            opp_reds = [x for x in g if reds.get((x["fx"], x["opp"]), 0) > 0]
            if side == "def" and own_reds:
                lines.append(("reds", f"They’ve had a player sent off in {len(own_reds)} game{'s' if len(own_reds) > 1 else ''} "
                                      f"({_list_names([names[x['opp']] + ' GW' + str(x['ev']) for x in own_reds])}), and "
                                      f"playing with ten men pushes the xG against up: {np.mean([x['xa'] for x in own_reds]):.2f} a game in those, "
                                      f"{np.mean([x['xa'] for x in g if x not in own_reds] or [0]):.2f} otherwise."))
                scores.append(2)
            if side == "att" and opp_reds:
                lines.append(("reds", f"Opponents have been down to ten men in {len(opp_reds)} of their games "
                                      f"({_list_names([names[x['opp']] + ' GW' + str(x['ev']) for x in opp_reds])}), which flatters the attacking "
                                      f"numbers a little: {np.mean([x['xf'] for x in opp_reds]):.2f} xG a game in those."))
                scores.append(1.5)
            # where the chances come from
            prof = tm.get("prof", {}).get(t, {}).get(side, {})
            if prof:
                best = max(("setpiece", "fast", "head"), key=lambda c: prof[c]["idx"])
                worst = min(("setpiece", "fast", "head"), key=lambda c: prof[c]["idx"])
                if prof[best]["idx"] >= 1.3:
                    lines.append(("source", f"A lot of it comes from {CATS[best]}: {prof[best]['pm']:.2f} xG a game, "
                                            f"{prof[best]['idx']:.1f}× the league average. "
                                            + ({"setpiece": "Set-piece weakness usually comes from height or marking set-ups, so it tends to persist.",
                                                "head": "Losing aerial duels in the box is usually about personnel and height, so it tends to persist.",
                                                "fast": "That points to a team caught upfield when they lose the ball, which tends to persist unless the shape changes."}[best]
                                               if side == "def" else "That’s a repeatable strength, so expect it to continue.")))
                    scores.append(2.2)
                elif prof[worst]["idx"] <= 0.7:
                    lines.append(("source", f"They give little away from {CATS[worst]} ({prof[worst]['idx']:.1f}× the league average)."
                                  if side == "def" else f"They rarely threaten from {CATS[worst]} ({prof[worst]['idx']:.1f}× the league average)."))
                    scores.append(1)
            if side == "def" and pens_conceded.get(t, 0) >= max(3, league_pens * 2):
                lines.append(("pens", f"They’ve given away {pens_conceded[t]} penalties, worth about {0.76 * pens_conceded[t]:.1f} xG on their own."))
                scores.append(2)
            if side == "att" and pens_won.get(t, 0) >= 3:
                lines.append(("pens", f"They’ve won {pens_won[t]} penalties, about {0.76 * pens_won[t]:.1f} of their xG."))
                scores.append(1.5)
            # missing players
            team_h = h[h["team"] == t]
            fx_ids = [x["fx"] for x in g]
            if side == "def":
                cands = team_h[team_h["element"].map(pos_of).isin([1, 2])].groupby("element")["minutes"].sum().nlargest(3).index
            else:
                cands = team_h[team_h["element"].map(pos_of).isin([3, 4])].groupby("element")["xg"].sum().nlargest(2).index
            for pid in cands:
                ph = team_h[team_h["element"] == pid].set_index("fixture")["minutes"]
                played = [x for x in g if ph.get(x["fx"], 0) >= 60]
                missed = [x for x in g if ph.get(x["fx"], 0) < 30]
                if len(played) >= 2 and len(missed) >= 2:
                    a1, a2 = np.mean([x[key] for x in played]), np.mean([x[key] for x in missed])
                    if (side == "def" and a2 - a1 >= 0.4) or (side == "att" and a1 - a2 >= 0.4):
                        lines.append(("absence", f"Losing {pname.get(pid, 'a key player')} has hurt: "
                                                 f"{a2:.2f} xG {'against' if side == 'def' else 'for'} a game in the {len(missed)} without him, "
                                                 f"{a1:.2f} in the {len(played)} with him."))
                        scores.append(2.6)
                        break
            # trend
            if len(g) >= 7:
                last, before = np.mean([x[key] for x in g[-4:]]), np.mean([x[key] for x in g[:-4]])
                ch = last / max(1e-6, before) - 1
                if abs(ch) >= 0.25:
                    better = (ch < 0) if side == "def" else (ch > 0)
                    lines.append(("trend", f"The trend is {'improving' if better else 'getting worse'}: {last:.2f} xG "
                                           f"{'against' if side == 'def' else 'for'} a game over the last four, "
                                           f"against {before:.2f} before that."))
                    scores.append(1.8)
            # finishing and goalkeeping
            gsum, xsum = int(sum(x["ga" if side == "def" else "gf"] for x in g)), sum(x[key] for x in g)
            if abs(gsum - xsum) >= 3:
                if side == "def":
                    lines.append(("luck", f"They’ve conceded {gsum} from {xsum:.1f} xG. " + (
                        "Good goalkeeping or luck is keeping the score down, and that rarely lasts." if gsum < xsum
                        else "Bad luck or poor goalkeeping has cost them, so results should improve if the chances stay the same.")))
                else:
                    lines.append(("luck", f"They’ve scored {gsum} from {xsum:.1f} xG. " + (
                        "Finishing above the chances usually cools off." if gsum > xsum
                        else "The goals should come if they keep making these chances.")))
                scores.append(1.6)
            order = sorted(range(len(lines)), key=lambda i: -scores[i])
            out[side] = [lines[i][1] for i in order]
            notable = [i for i in order if scores[i] >= 2]
            out[side + "_short"] = lines[notable[0]][1] if notable else None
        team_ctx[t] = out

    # players
    sh = shots
    pos_xg_per_shot = {}
    if len(sh):
        tmp = sh[~sh["pen"]].assign(pos=sh["player"].map(pos_of))
        per = tmp.groupby("player")["xg"].agg(["sum", "count"])
        per = per[per["count"] >= 8].join(tmp.groupby("player")["pos"].first())
        pos_xg_per_shot = (per["sum"] / per["count"]).groupby(per["pos"]).median().to_dict()
    player_ctx = {}
    for pid, r in rates.items():
        s_ = r.get("season") or {}
        if s_.get("mins", 0) < 270:
            continue
        lines = []
        g = h[(h["element"] == pid) & (h["minutes"] > 0)]
        tot_xg = float(g["xg"].sum())
        nm = pname.get(pid, "He")
        if tot_xg >= 0.5 and len(sh):
            ps0 = sh[sh["player"] == pid]
            if len(ps0):
                inbox = ps0[ps0["zone"].isin(["six", "central", "wide"]) & ~ps0["pen"]]
                lines.append(f"Where it comes from: {len(ps0)} shots worth {ps0['xg'].sum():.1f} xG "
                             f"({len(ps0) / max(1, s_["mins"]) * 90:.1f} per 90), with {len(inbox) / max(1, len(ps0)):.0%} taken inside the box"
                             + (f" and {int(ps0['pen'].sum())} penalt{'y' if int(ps0['pen'].sum()) == 1 else 'ies'}" if ps0["pen"].any() else "") + ".")
        if tot_xg >= 0.8:
            ps = sh[sh["player"] == pid] if len(sh) else sh
            pen = float(ps[ps["pen"]]["xg"].sum()) if len(ps) else 0
            if pen / tot_xg >= 0.3:
                lines.append(f"{pen / tot_xg:.0%} of his {tot_xg:.1f} xG has come from penalties, so his open-play threat is lower than the headline number.")
            npx = ps[~ps["pen"]] if len(ps) else ps
            if len(npx) >= 8 and pos_xg_per_shot.get(r["pos"]):
                q, base = float(npx["xg"].mean()), pos_xg_per_shot[r["pos"]]
                if q >= base * 1.35:
                    lines.append(f"He gets into good positions: {q:.2f} xG per shot, against {base:.2f} for a typical {POS_NAME[r['pos']][:-1]}. "
                                 f"That points to movement in the box, which is the most reliable kind of threat.")
                elif q <= base * 0.7:
                    lines.append(f"Lots of his shots are speculative: {q:.2f} xG per shot, against {base:.2f} for a typical {POS_NAME[r['pos']][:-1]}, "
                                 f"so the volume flatters him.")
            prof = r.get("profile") or {}
            if prof.get("setpiece", 0) >= 0.45 and r["pos"] <= 2:
                lines.append(f"Most of his threat ({prof['setpiece']:.0%}) comes from set pieces, so it depends on his team winning corners and free kicks.")
            elif prof.get("head", 0) >= 0.45:
                lines.append(f"{prof['head']:.0%} of his chances are headers, so he relies on crosses and set pieces reaching him.")
            if len(g) >= 4:
                best = g.loc[g["xg"].idxmax()]
                if best["xg"] >= 1.0 and best["xg"] / tot_xg >= 0.3:
                    lines.append(f"One game inflates it: {best['xg']:.1f} xG {'at home to' if best['was_home'] else 'away at'} "
                                 f"{names.get(best['opp'], 'one opponent')} in GW{int(best['event'])}, "
                                 f"{best['xg'] / tot_xg:.0%} of his season. Take that out and he averages "
                                 f"{(tot_xg - best['xg']) / max(1, g['minutes'].sum() - best['minutes']) * 90:.2f} xG per 90.")
        if len(g) >= 3:
            w_ = g["minutes"].to_numpy()
            faced = float(np.average([dfn[o] for o in g["opp"]], weights=w_))
            if faced >= 1.12 and r["pos"] >= 3:
                lines.append(f"He’s played against weaker defences than average so far (they’ve conceded {faced - 1:.0%} more than a typical side), "
                             f"so his rates may dip as the fixtures get harder.")
            elif faced <= 0.88 and r["pos"] >= 3:
                lines.append(f"His numbers have come against tougher defences than average ({1 - faced:.0%} stingier than a typical side), "
                             f"so there may be more to come.")
            starts = int(s_.get("starts", 0))
            subs_off = int(((g["minutes"] >= 45) & (g["minutes"] < 75)).sum())
            if starts >= 4 and subs_off >= max(2, starts // 2):
                lines.append(f"He’s often taken off early: {subs_off} of his {starts} starts ended before 75 minutes, which caps his minutes and his bonus.")
        if r["pos"] in (2, 3) and r.get("dc_games", 0) >= 3 and r.get("dc_hits", 0) >= 2:
            lines.append(f"DEFCON has been a steady source of points: {r['dc_hits']} of {r['dc_games']} full games over the line, "
                         f"averaging {r['dc90']:.1f} actions per 90.")
        if lines:
            player_ctx[pid] = lines[:5]
    return team_ctx, player_ctx


def tier(rank, n=20):
    return ("one of the best in the league" if rank <= 3 else "strong" if rank <= 7 else "around the middle"
            if rank <= 13 else "below average" if rank <= 17 else "one of the weakest in the league")


def plain_summaries(ds, tm, ttab, rates, gws, proj_by_id):
    """'In plain terms' paragraphs: what a team's or player's numbers mean for FPL decisions."""
    ids = tm["ids"]
    names = dict(zip(ds.teams["id"], ds.teams["name"]))
    adj_att = {t: 1 + sorted(ids, key=lambda k: -tm["att"][k]).index(t) for t in ids}
    adj_def = {t: 1 + sorted(ids, key=lambda k: tm["dfn"][k]).index(t) for t in ids}
    up = ds.fixtures[~ds.fixtures["finished"] & ds.fixtures["event"].isin(gws)]
    run_f, run_a = {t: 0.0 for t in ids}, {t: 0.0 for t in ids}
    for r in up.itertuples():
        lh, la = fixture_lambdas(tm, r.team_h, r.team_a)
        run_f[r.team_h] += lh; run_a[r.team_h] += la; run_f[r.team_a] += la; run_a[r.team_a] += lh
    ease_att = {t: 1 + sorted(ids, key=lambda k: -run_f[k]).index(t) for t in ids}     # 1 = most goals expected
    ease_def = {t: 1 + sorted(ids, key=lambda k: run_a[k]).index(t) for t in ids}      # 1 = fewest conceded
    run_word = lambda r: "very kind" if r <= 4 else "good" if r <= 8 else "average" if r <= 13 else "tough" if r <= 17 else "very tough"
    team_sum = {}
    for t in ids:
        a, d = adj_att[t], adj_def[t]
        implic = []
        if d <= 6:
            implic.append("their goalkeeper and defenders are among the best clean sheet bets")
        if a <= 6:
            implic.append("their main attackers are worth owning")
        if d >= 16:
            implic.append("attackers playing against them are worth targeting")
        if a >= 16:
            implic.append("defenders playing against them are good picks")
        if not implic:
            implic.append("their players are only worth picking when the fixtures line up")
        team_sum[t] = (f"In plain terms: {names[t]} are {tier(a)} going forward ({ordinal(a)} on the model’s rating) and "
                       f"{tier(d)} at the back ({ordinal(d)}). For FPL, that means {_list_names(implic)}. "
                       f"Over the next {len(gws)} gameweeks their fixtures are {run_word(ease_att[t])} for attackers "
                       f"({ordinal(ease_att[t])} for expected goals) and {run_word(ease_def[t])} for defenders "
                       f"({ordinal(ease_def[t])} for goals expected against).")
    pos_rank = {}
    for pos in (1, 2, 3, 4):
        pool = sorted([pid for pid, it in proj_by_id.items() if it["r"]["pos"] == pos and it["ep"]], key=lambda k: -proj_by_id[k]["ep"][0])
        for i, pid in enumerate(pool):
            pos_rank[pid] = (i + 1, len(pool))
    player_sum = {}
    for pid, it in proj_by_id.items():
        r = it["r"]
        s_ = r.get("season") or {}
        if s_.get("mins", 0) < 270 or not it["ep"]:
            continue
        pos = r["pos"]
        team = names.get(int(it["p"]["team"]), "his team")
        bits = []
        if pos == 4 or r["share_g"] >= 0.25:
            role = f"{team}’s main goal threat, taking {r['share_g']:.0%} of their expected goals"
        elif pos == 3 and r["share_a"] >= 0.15:
            role = f"one of {team}’s chief creators, with {r['share_a']:.0%} of their expected assists"
        elif pos == 3:
            role = f"a midfielder who chips in with {r['share_g']:.0%} of {team}’s expected goals"
        elif pos == 2 and (r.get("profile") or {}).get("setpiece", 0) >= 0.4 and r["share_g"] >= 0.05:
            role = f"a defender who carries a real goal threat from set pieces"
        elif pos == 2 and r["dc_hit"] >= 0.45:
            role = "a defender whose points come from clean sheets and DEFCON rather than goals"
        elif pos == 2:
            role = "a defender whose points depend mainly on clean sheets"
        else:
            role = "a goalkeeper whose points come from clean sheets and saves"
        bits.append(f"In plain terms: {_text(it['p'].get('web_name'))} is {role}.")
        if r["p60"] >= 0.9:
            bits.append("He’s nailed on when fit.")
        elif r["p60"] >= 0.6:
            bits.append("He usually starts, but not always.")
        elif r["emins"] > 0:
            bits.append("His minutes are the big risk: he isn’t a regular starter right now.")
        avg = sum(it["ep"][:len(gws)]) / max(1, len(it["ep"][:len(gws)]))
        if avg > 0.5:
            ch = it["ep"][0] / avg - 1
            bits.append("This week looks better than his average over the next six." if ch >= 0.12 else
                        "This week is weaker than his average over the next six, so he may be one to bench or rotate." if ch <= -0.12 else
                        "This week is about his normal level.")
        rk, n = pos_rank.get(pid, (None, None))
        if rk:
            bits.append((f"He’s the best {POS_NAME[pos][:-1]} this week on Lab points, out of {n}." if rk == 1 else
                         f"He’s the {ordinal(rk)}-best {POS_NAME[pos][:-1]} this week out of {n}, on Lab points.") if rk <= 15 else "")
        player_sum[pid] = " ".join(b for b in bits if b)
    return team_sum, player_sum

# ======================================================== trend spotter
def trend_signals(ds, tm, n_calls=3):
    """Forecast track record this season (calls made a few weeks ago against what happened) and each team's
    'surprise': how far its last two games beat what the model expected before each one."""
    fx = ds.fixtures
    done = fx[fx["finished"]]
    last = tm["last_gw"]
    models = {}

    def model_at(k):
        if k not in models:
            models[k] = team_model(SimpleNamespace(teams=ds.teams, fixtures=fx.assign(finished=fx["finished"] & (fx["event"] <= k)),
                                                   hist=ds.hist[ds.hist["event"] <= k], shots=ds.shots[ds.shots["event"] <= k],
                                                   team_prior=getattr(ds, "team_prior", {}), player_prior={}))
        return models[k]
    xg_for = lambda r, t: r.xg_h if r.team_h == t else r.xg_a
    surprise = {}
    for t in tm["ids"]:
        g = done[(done.team_h == t) | (done.team_a == t)].sort_values("event").tail(2)
        if len(g) < 2:
            continue
        diffs = []
        for r in g.itertuples():
            if r.event - 1 < 1:
                continue
            lh, la = fixture_lambdas(model_at(r.event - 1), r.team_h, r.team_a)
            d_ = xg_for(r, t) - (lh if r.team_h == t else la)
            if d_ == d_:          # skip matches whose xG isn't published yet (NaN)
                diffs.append(d_)
        if diffs:
            surprise[int(t)] = round(float(np.mean(diffs)), 3)
    calls = []
    for k in range(max(2, last - 5), last - 1):
        m = model_at(k)
        rows = []
        for t in m["ids"]:
            past = done[((done.team_h == t) | (done.team_a == t)) & (done.event <= k)].sort_values("event").tail(4)
            fut = done[((done.team_h == t) | (done.team_a == t)) & (done.event > k) & (done.event <= k + 4)]
            if len(past) < 3 or len(fut) < 2:
                continue
            l4 = float(np.mean([xg_for(r, t) for r in past.itertuples()]))
            pred = float(np.mean([fixture_lambdas(m, r.team_h, r.team_a)[0 if r.team_h == t else 1] for r in fut.itertuples()]))
            act = float(np.mean([xg_for(r, t) for r in fut.itertuples()]))
            rows.append(dict(t=int(t), made=k, last4=round(l4, 2), pred=round(pred, 2), actual=round(act, 2), games=int(len(fut))))
        rows.sort(key=lambda r: -(r["pred"] - r["last4"]))
        calls += [r for r in rows[:n_calls] if r["pred"] - r["last4"] >= 0.2]
    return surprise, calls


# ======================================================== live record
def write_snapshot(folder, data):
    """Save this run's projections for the coming gameweek. Runs after the deadline move on to the
    next gameweek, so the last file written before each deadline is the true forecast."""
    if not folder:
        return
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    g = data["meta"]["gws"][0]
    snap = dict(gw=g, saved=data["meta"]["generated"],
                proj={p["id"]: [p["ep"][0], p["fpl_ep"], (p.get("sim") or {}).get("haul")] for p in data["players"]})
    (folder / f"gw{g}.json").write_text(json.dumps(snap, separators=(",", ":")))


def live_record(folder, ds):
    if not folder or not Path(folder).exists():
        return dict(summary={}, weeks=[])
    pos_of = dict(zip(ds.players["id"], ds.players["element_type"]))
    name_of = dict(zip(ds.players["id"], ds.players["web_name"]))
    last = int(ds.fixtures.loc[ds.fixtures["finished"], "event"].max())
    weeks = []
    for f in sorted(Path(folder).glob("gw*.json"), key=lambda p: int(p.stem[2:])):
        snap = json.loads(f.read_text())
        k = snap["gw"]
        if k > last:
            continue
        actual, played_teams = actual_points(ds, k)
        team_of = dict(zip(ds.players["id"], ds.players["team"]))
        rows = []
        for pid, (ours, fpl, haul) in snap["proj"].items():
            pid = int(pid)
            if team_of.get(pid) not in played_teams:
                continue
            a = actual.loc[pid] if pid in actual.index else None
            rows.append(dict(id=pid, n=_text(name_of.get(pid)), pos=int(pos_of.get(pid, 3)), ours=ours, fpl=fpl,
                             pts=float(a["pts"]) if a is not None else 0.0, mins=float(a["mins"]) if a is not None else 0.0))
        if rows:
            week, _ = score_week(pd.DataFrame(rows), k)
            week["saved"] = snap["saved"]
            weeks.append(week)
    return dict(summary=summarise(weeks), weeks=weeks)


# ============================================================ insights
def player_form(ds, n=8):
    out = {}
    names = dict(zip(ds.teams["id"], ds.teams["short_name"]))
    for pid, g in ds.hist.groupby("element"):
        g = g.tail(n)
        # [gw, opponent, home, minutes, xG, xA, FPL points, started, goals, assists, touches, passes, box touches]
        num_ = lambda v: None if v is None or pd.isna(v) else int(v)
        out[int(pid)] = [[int(r.event), names.get(r.opp, "?"), int(r.was_home), int(r.minutes), round(float(r.xg), 2),
                          round(float(r.xa), 2), int(r.event_points), int(r.starts > 0 or r.minutes >= 60),
                          int(r.goals_scored), int(r.assists), num_(getattr(r, "touches", None)),
                          num_(getattr(r, "accurate_passes", None)), num_(getattr(r, "touches_opposition_box", None))] for r in g.itertuples()]
    return out


def team_extras(ds, tm):
    """Per team: recent form, finishing and goalkeeping luck, and who takes the set pieces."""
    out = {}
    m = tm["m"]
    names = dict(zip(ds.teams["id"], ds.teams["short_name"]))
    sh = ds.shots
    recent_fx = {}
    for t in tm["ids"]:
        H, A = m[m.h == t], m[m.a == t]
        rows = sorted([(r.ev, names[r.a], 1, r.xh, r.xa, r.gh, r.ga) for r in H.itertuples()] +
                      [(r.ev, names[r.h], 0, r.xa, r.xh, r.ga, r.gh) for r in A.itertuples()])
        recent_fx[t] = set(H.id) | set(A.id)
        gf = sum(x[5] for x in rows); ga = sum(x[6] for x in rows)
        xgf = sum(x[3] for x in rows); xga = sum(x[4] for x in rows)
        h = ds.hist[ds.hist.team == t]
        tsh = sh[sh.team == t]
        pens = tsh[tsh.sit == "pen"].groupby("player").size()
        fk = tsh[tsh.raw == "free-kick"].groupby("player").size()
        pl = ds.players[ds.players.team == t]
        order = lambda col: [int(x) for x in pl.sort_values(col)[pl.sort_values(col)[col].notna()]["id"].head(2)] \
            if col in pl else []
        out[t] = dict(
            form=[[int(x[0]), x[1], x[2], round(float(x[3]), 2), round(float(x[4]), 2), int(x[5]), int(x[6])] for x in rows[-8:]],
            results=[[int(x[0]), x[1], x[2], round(float(x[3]), 2), round(float(x[4]), 2), int(x[5]), int(x[6])] for x in rows],
            luck=dict(gf=int(gf), xgf=round(float(xgf), 1), ga=int(ga), xga=round(float(xga), 1)),
            sp=dict(
                pens=[[int(pid), int(v)] for pid, v in pens.sort_values(ascending=False).head(3).items() if pd.notna(pid)],
                fk=[[int(pid), int(v)] for pid, v in fk.sort_values(ascending=False).head(3).items() if pd.notna(pid)],
                order=dict(pen=order("penalties_order"), corner=order("corners_and_indirect_freekicks_order"),
                           fk=order("direct_freekicks_order")),
            ),
        )
    return out


# ============================================================ your team
def import_team(team_id, next_gw):
    """Your squad, bank, free transfers and chips from the official FPL API (public endpoints)."""
    try:
        import requests
        s = fpl_session()
        get = lambda path: s.get(f"{API}/{path}", timeout=20).json()
        entry = get(f"entry/{team_id}/")
        history = get(f"entry/{team_id}/history/")
    except Exception as e:  # noqa: BLE001
        print("Could not import team:", e)
        return None
    my = team_from_api(team_id, next_gw, entry, history, lambda gw: get(f"entry/{team_id}/event/{gw}/picks/"))
    if my:
        my["first_name"] = entry.get("player_first_name")
        my["overall_rank"] = entry.get("summary_overall_rank")
    try:
        transfers = get(f"entry/{team_id}/transfers/")
        picks = {}
        for row in history.get("current", []):
            try:
                picks[row["event"]] = get(f"entry/{team_id}/event/{row['event']}/picks/")
            except Exception:  # noqa: BLE001
                pass
        my["raw"] = dict(history=history, transfers=transfers, picks=picks)
    except Exception as e:  # noqa: BLE001
        print("Could not load transfer history:", e)
    return my


def selling_prices(raw, ds, ids):
    """FPL selling price for each player you own: what you paid plus half of any rise, rounded down to £0.1m.
    Purchase price comes from your transfer history, or the price when he first appears in your team."""
    now = dict(zip(ds.players["id"], ds.players["now_cost"]))
    snaps = ds.snaps.dropna(subset=["now_cost"]) if "now_cost" in ds.snaps else pd.DataFrame()
    price_at = {(int(r.id), int(r.gw)): float(r.now_cost) for r in snaps.itertuples()} if len(snaps) else {}
    bought = {}
    for t in sorted(raw.get("transfers", []), key=lambda t: (t["event"], t.get("time", ""))):
        bought[t["element_in"]] = t.get("element_in_cost", 0) / 10
    first_seen = {}
    for gw, pk in sorted(raw.get("picks", {}).items(), key=lambda kv: int(kv[0])):
        for p in pk.get("picks", []):
            first_seen.setdefault(p["element"], int(gw))
    out = {}
    for pid in ids:
        cur = now.get(pid)
        if cur is None:
            continue
        paid = bought.get(pid)
        if paid is None and pid in first_seen:
            g = first_seen[pid]
            paid = price_at.get((pid, g - 1)) or price_at.get((pid, g))
        if paid is None:
            continue
        sell = cur if cur <= paid else paid + math.floor(round((cur - paid) * 10) / 2) / 10
        out[int(pid)] = round(sell, 1)
    return out


def decisions_vs_luck(raw, ds, stack_rows):
    """Split your season into decisions and luck. For each gameweek: what your starting XI was expected to score,
    using the model's projection made before that deadline (rebuilt from the data available then), against what
    it scored; the same for your captain against the best-projected option in your XI, and for each transfer."""
    if not raw or not stack_rows:
        return None
    ours = {(int(r[0]), int(r[6])): float(r[1]) for r in stack_rows}
    proj_gws = {int(r[0]) for r in stack_rows}
    pts = ds.hist.groupby(["element", "event"])["event_points"].sum().to_dict()
    name = dict(zip(ds.players["id"], ds.players["web_name"]))
    last = int(ds.fixtures.loc[ds.fixtures["finished"], "event"].max()) if ds.fixtures["finished"].any() else 0
    chips = {c["event"]: c["name"] for c in raw.get("history", {}).get("chips", [])}
    weeks = []
    for gw, pk in sorted(raw.get("picks", {}).items(), key=lambda kv: int(kv[0])):
        gw = int(gw)
        if gw > last or gw not in proj_gws:
            continue
        xi = [p for p in pk.get("picks", []) if p.get("multiplier", 0) > 0]
        if not xi:
            continue
        exp = sum(ours.get((gw, p["element"]), 0.0) * p["multiplier"] for p in xi)
        act = sum(pts.get((p["element"], gw), 0) * p["multiplier"] for p in xi)
        cap = next((p for p in xi if p.get("is_captain")), None)
        row = dict(gw=gw, exp=round(exp, 1), act=int(act), chip=chips.get(gw))
        if cap:
            extra = (cap.get("multiplier", 2) or 2) - 1
            best = max(xi, key=lambda p: ours.get((gw, p["element"]), 0.0))
            ce, be = ours.get((gw, cap["element"]), 0.0), ours.get((gw, best["element"]), 0.0)
            row.update(cap=name.get(cap["element"], ""), cap_exp=round(ce * extra, 1), cap_act=int(pts.get((cap["element"], gw), 0) * extra),
                       cap_best=name.get(best["element"], ""), cap_cost=round((be - ce) * extra, 1))
        weeks.append(row)
    trs = []
    tr = sorted(raw.get("transfers", []), key=lambda t: (t["event"], t.get("time", "")))
    for k, t in enumerate(tr):
        gw, pin, pout = t["event"], t["element_in"], t["element_out"]
        if chips.get(gw) in ("wildcard", "freehit") or gw not in proj_gws:
            continue
        end = next((u["event"] - 1 for u in tr[k + 1:] if u["element_out"] == pin), last)
        span = [g for g in range(gw, end + 1) if g <= last and g in proj_gws]
        if not span:
            continue
        e = sum(ours.get((g, pin), 0.0) - ours.get((g, pout), 0.0) for g in span)
        a = sum(pts.get((pin, g), 0) - pts.get((pout, g), 0) for g in span)
        trs.append(dict(gw=gw, inn=name.get(pin, ""), out=name.get(pout, ""), weeks=len(span), exp=round(e, 1), act=int(a)))
    if not weeks:
        return None
    sm = lambda rows, k: round(sum(r.get(k, 0) or 0 for r in rows), 1)
    return dict(weeks=weeks, transfers=trs, first_gw=weeks[0]["gw"], summary=dict(
        exp=sm(weeks, "exp"), act=sm(weeks, "act"), cap_exp=sm(weeks, "cap_exp"), cap_act=sm(weeks, "cap_act"),
        cap_cost=sm(weeks, "cap_cost"), tr_exp=sm(trs, "exp"), tr_act=sm(trs, "act"), n=len(weeks), tr_n=len(trs)))


def season_audit(raw, ds):
    """How your decisions have worked out: every transfer against keeping the player, every captain
    against the best option in your XI, points left on the bench, and rank week by week."""
    if not raw:
        return None
    pts = ds.hist.groupby(["element", "event"])["event_points"].sum().to_dict()
    name = dict(zip(ds.players["id"], ds.players["web_name"]))
    last = int(ds.fixtures.loc[ds.fixtures["finished"], "event"].max()) if ds.fixtures["finished"].any() else 0
    hist = sorted(raw.get("history", {}).get("current", []), key=lambda r: r["event"])
    chips = {c["event"]: c["name"] for c in raw.get("history", {}).get("chips", [])}
    hits = {r["event"]: int(r.get("event_transfers_cost", 0)) for r in hist}
    n_by_gw = {}
    tr = sorted(raw.get("transfers", []), key=lambda t: (t["event"], t.get("time", "")))
    for t in tr:
        n_by_gw[t["event"]] = n_by_gw.get(t["event"], 0) + 1
    rows = []
    for k, t in enumerate(tr):
        gw, pin, pout = t["event"], t["element_in"], t["element_out"]
        if chips.get(gw) in ("wildcard", "freehit"):
            continue
        end = next((u["event"] - 1 for u in tr[k + 1:] if u["element_out"] == pin), last)
        weeks = [g for g in range(gw, end + 1) if g <= last]
        a = sum(pts.get((pin, g), 0) for g in weeks)
        b = sum(pts.get((pout, g), 0) for g in weeks)
        cost = hits.get(gw, 0) / max(1, n_by_gw.get(gw, 1))
        rows.append(dict(gw=gw, inn=name.get(pin, str(pin)), out=name.get(pout, str(pout)), pin=pin, pout=pout,
                         weeks=len(weeks), pts_in=int(a), pts_out=int(b), hit=round(cost, 1), net=round(a - b - cost, 1),
                         price_in=t.get("element_in_cost", 0) / 10, price_out=t.get("element_out_cost", 0) / 10))
    caps = []
    for gw, pk in sorted(raw.get("picks", {}).items(), key=lambda kv: int(kv[0])):
        gw = int(gw)
        if gw > last:
            continue
        picks = pk.get("picks", [])
        xi = [p for p in picks if p.get("multiplier", 0) > 0]
        cap = next((p for p in picks if p.get("is_captain")), None)
        if not cap or not xi:
            continue
        mult = cap.get("multiplier", 2) or 2
        cpts = pts.get((cap["element"], gw), 0)
        best = max(xi, key=lambda p: pts.get((p["element"], gw), 0))
        bpts = pts.get((best["element"], gw), 0)
        caps.append(dict(gw=gw, cap=name.get(cap["element"], ""), cap_pts=int(cpts), mult=int(mult),
                         best=name.get(best["element"], ""), best_pts=int(bpts), lost=int((bpts - cpts) * (mult - 1))))
    weeks = [dict(gw=r["event"], pts=r.get("points"), total=r.get("total_points"), rank=r.get("overall_rank"),
                  gw_rank=r.get("rank"), bench=r.get("points_on_bench"), hits=r.get("event_transfers_cost"),
                  chip=chips.get(r["event"])) for r in hist]
    done_tr = [r for r in rows if r["weeks"] >= 1]
    return dict(
        transfers=rows, captains=caps, weeks=weeks,
        summary=dict(
            transfers=len(done_tr), net=round(sum(r["net"] for r in done_tr), 1),
            good=sum(1 for r in done_tr if r["net"] > 0), hits=int(sum(hits.values())),
            cap_best=sum(1 for c in caps if c["lost"] == 0), cap_n=len(caps), cap_lost=int(sum(c["lost"] for c in caps)),
            bench=int(sum((w["bench"] or 0) for w in weeks)),
            best_rank=min((w["rank"] for w in weeks if w["rank"]), default=None),
        ),
    )


def team_from_api(team_id, next_gw, entry, history, picks_for):
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    cur = entry.get("current_event") or next_gw - 1
    pick_gw = cur - 1 if chips.get(cur) == "freehit" else cur      # a Free Hit squad reverts afterwards
    picks = picks_for(pick_gw)
    latest = picks_for(cur) if pick_gw != cur else picks
    ft = 1
    for row in sorted(history.get("current", []), key=lambda x: x["event"]):
        ev = row["event"]
        if ev == 1:
            ft = 1
            continue
        if chips.get(ev) not in ("wildcard", "freehit"):
            ft = max(0, ft - int(row.get("event_transfers", 0)))
        ft = min(5, ft + 1)
    half = (1, 19) if next_gw <= 19 else (20, 38)
    used = {name for ev, name in chips.items() if half[0] <= ev <= half[1]}
    names = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}
    eh = latest.get("entry_history", {})
    return dict(
        id=int(team_id), team=entry.get("name"), manager=f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
        ids=[int(p["element"]) for p in sorted(picks.get("picks", []), key=lambda p: p["position"])],
        bank=round(eh.get("bank", entry.get("last_deadline_bank", 0)) / 10, 1), ft=ft, gw=cur,
        rank=entry.get("summary_overall_rank"), points=entry.get("summary_overall_points"),
        chips={k: k not in used for k in names}, chip_names=names, half_ends=half[1],
    )


# ====================================================== last season priors
PREV_SEASON = "2025-2026"
PROMOTED_PRIOR = (0.85, 1.18)
ELO_K = 0.0012


def build_priors(ds, prev, elo=None):
    """Team and player starting points from last season (and ClubElo), so early weeks and new
    arrivals aren't judged on a handful of matches."""
    tm_prev = team_model(prev)
    rates_prev = player_model(prev, tm_prev)
    prev_code = {int(i): int(c) for c, i in prev.code2id.items()}
    cur_id = {int(c): int(i) for c, i in ds.code2id.items()}
    team_prior = {}
    for pid_t, code in prev_code.items():
        if code in cur_id:
            a, d = tm_prev["att"][pid_t], tm_prev["dfn"][pid_t]
            team_prior[cur_id[code]] = (a ** 0.7, d ** 0.7)        # regress a third of the way to average
    for t in ds.teams["id"].astype(int):
        team_prior.setdefault(int(t), PROMOTED_PRIOR)
    if elo:
        mean = np.mean(list(elo.values()))
        for t, e in elo.items():
            ea, ed = math.exp(ELO_K * (e - mean)), math.exp(-ELO_K * (e - mean))
            a0, d0 = team_prior.get(t, (1.0, 1.0))
            team_prior[t] = (math.sqrt(a0 * ea), math.sqrt(d0 * ed))
    code_of_prev = dict(zip(prev.players["id"], prev.players["player_code"]))
    team_code_prev = dict(zip(prev.players["id"], prev.players["team_code"]))
    by_code = {}
    for pid, r in rates_prev.items():
        if r["season"] and r["season"]["mins"] >= 900:
            by_code[int(code_of_prev[pid])] = (r, team_code_prev.get(pid))
    player_prior = {}
    for p in ds.players.itertuples():
        hit = by_code.get(int(p.player_code))
        if not hit:
            continue
        r, old_team = hit
        same_team = old_team == p.team_code
        mix = (lambda v, d: v) if same_team else (lambda v, d: 0.5 * v + 0.5 * d)
        pos = int(p.element_type)
        if r["pos"] != pos:
            continue
        player_prior[int(p.id)] = dict(
            share_g=mix(r["share_g"], SHARE_PRIOR["g"][pos]), share_a=mix(r["share_a"], SHARE_PRIOR["a"][pos]),
            dc_hit=r["dc_hit"] if pos in DEFCON_THRESHOLD else None, bonus90=r["bonus90"], profile=r["profile"],
            mins=r["season"]["mins"], same_team=same_team)
    ds.team_prior, ds.player_prior = team_prior, player_prior
    return tm_prev


def fetch_elo(teams, day=None):
    """Club Elo ratings from api.clubelo.com (free). Elo updates after every result, so it is current-season information."""
    import io
    last = None
    days = [day] if day else [dt.date.today().isoformat(), (dt.date.today() - dt.timedelta(days=1)).isoformat()]
    for d in days:
        for scheme in ("https", "http"):
            try:
                r = fpl_session().get(f"{scheme}://api.clubelo.com/{d}", timeout=25)
                if not r.ok:
                    last = f"HTTP {r.status_code} from api.clubelo.com"
                    continue
                df = pd.read_csv(io.StringIO(r.text))
                look = team_lookup(teams)
                out = {}
                for row in df.itertuples():
                    t = look.get(_team_key(row.Club)) or look.get(_team_key(TEAM_ALIASES.get(_team_key(row.Club), "")))
                    if t and row.Country == "ENG":
                        out[t] = float(row.Elo)
                if len(out) >= 15:
                    try:
                        cache = Path(__file__).resolve().parents[1] / "site" / "history" / "elo_last.json"
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps({"date": d, "elo": out}))
                    except Exception:  # noqa: BLE001
                        pass
                    return out
                last = f"only matched {len(out)} of 20 teams (club names may have changed)"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
    cache = Path(__file__).resolve().parents[1] / "site" / "history" / "elo_last.json"
    if cache.exists():
        try:
            saved = json.loads(cache.read_text())
            age = (dt.date.today() - dt.date.fromisoformat(saved["date"])).days
            if age <= 14 and len(saved["elo"]) >= 15:
                note_source("ClubElo ratings", f"{last}; using the ratings saved on {saved['date']} instead")
                fetch_elo.from_cache = saved["date"]
                return {int(k): v for k, v in saved["elo"].items()}
        except Exception:  # noqa: BLE001
            pass
    note_source("ClubElo ratings", last or "no data returned")
    return {}


# ============================================================ referees
FD_SEASON = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
REF_PRIOR_MATCHES = 8
PEN_XG = 0.76


def season_code(season):      # "2026-2027" -> "2627"
    a, b = season.split("-")
    return a[2:] + b[2:]


def load_results_file(season, cache):
    try:
        return _read(FD_SEASON.format(code=season_code(season)), cache, 6 * 3600 if season == SEASON else None)
    except Exception as e:  # noqa: BLE001
        print("football-data season file not available:", e)
        return pd.DataFrame()


def match_results_file(df, ds):
    """Join a football-data.co.uk season file to our fixtures by home and away team."""
    if df.empty or "HomeTeam" not in df:
        return {}
    look = team_lookup(ds.teams)
    pair_to_fix = {(r.team_h, r.team_a): r.id for r in ds.fixtures.itertuples()}
    out = {}
    for r in df.to_dict("records"):
        h, a = look.get(_team_key(r.get("HomeTeam"))), look.get(_team_key(r.get("AwayTeam")))
        fid = pair_to_fix.get((h, a))
        if fid:
            out[fid] = r
    return out


def referee_table(seasons):
    """Penalties, cards and fouls per game for each referee, from shot data and football-data.co.uk files.
    seasons: list of (ds, joined_rows) pairs."""
    rows = []
    for ds, joined in seasons:
        pens = ds.shots[ds.shots["pen"]].groupby("fixture").size().to_dict() if len(ds.shots) else {}
        for fid, r in joined.items():
            ref = _text(r.get("Referee")).strip()
            if not ref or pd.isna(r.get("FTHG")):
                continue
            num = lambda c: float(pd.to_numeric(r.get(c), errors="coerce") or 0)
            rows.append(dict(ref=ref, pens=pens.get(fid, 0), yel=num("HY") + num("AY"), red=num("HR") + num("AR"),
                             fouls=num("HF") + num("AF")))
    if not rows:
        return {}, None
    df = pd.DataFrame(rows)
    league = dict(pens=df.pens.mean(), yel=df.yel.mean(), red=df.red.mean(), fouls=df.fouls.mean())
    out = {}
    for ref, g in df.groupby("ref"):
        n = len(g)
        out[ref] = dict(n=n, **{k: round((g[k].sum() + REF_PRIOR_MATCHES * league[k]) / (n + REF_PRIOR_MATCHES), 3)
                                for k in ("pens", "yel", "red", "fouls")},
                        raw_pens=int(g.pens.sum()))
    return out, {k: round(v, 3) for k, v in league.items()}


def upcoming_referees(teams, fixtures_df):
    """Referees for the next round from football-data.co.uk's fixtures file (published with the odds)."""
    if fixtures_df is None or fixtures_df.empty or "Referee" not in fixtures_df:
        return {}
    look = team_lookup(teams)
    out = {}
    for r in fixtures_df[fixtures_df["Div"].astype(str) == "E0"].to_dict("records"):
        h, a = look.get(_team_key(r.get("HomeTeam"))), look.get(_team_key(r.get("AwayTeam")))
        ref = _text(r.get("Referee")).strip()
        if h and a and ref:
            out[(h, a)] = ref
    return out


# ============================================================ odds calibration
def fit_book_weight(rows):
    """Choose the model/market blend that best predicted actual goals (Poisson log loss)."""
    if len(rows) < 40:
        return None
    def loss(w):
        tot = 0.0
        for m, k, g in rows:
            lam = max(0.05, w * k + (1 - w) * m)
            tot += lam - g * math.log(lam) + math.lgamma(g + 1)
        return tot / len(rows)
    grid = [round(x * 0.05, 2) for x in range(21)]
    losses = {w: loss(w) for w in grid}
    best = min(losses, key=losses.get)
    return dict(w=best, n=len(rows), model=round(losses[0.0], 4), market=round(losses[1.0], 4), blend=round(losses[best], 4))


def closing_lambdas(r):
    for H, Dw, A, O, U in (("AvgCH", "AvgCD", "AvgCA", "AvgC>2.5", "AvgC<2.5"), ("B365CH", "B365CD", "B365CA", "B365C>2.5", "B365C<2.5"),
                           ("AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5")):
        vals = [pd.to_numeric(r.get(c), errors="coerce") for c in (H, Dw, A, O, U)]
        if all(v is not None and not pd.isna(v) and v > 1 for v in vals):
            ph, _, pa = _devig(*vals[:3])
            return implied_goals(ph, pa, _devig(vals[3], vals[4])[0])
    return None


# ============================================================ elite managers
ELITE_N = 500


def import_elite(gw, folder, n=ELITE_N, delay=0.12):
    """Ownership and captaincy among the top n managers in the overall league, once per gameweek (cached)."""
    if gw < 1:
        return None
    path = Path(folder) / f"elite_gw{gw}.json" if folder else None
    if path and path.exists():
        return json.loads(path.read_text())
    try:
        import requests
        s = fpl_session()
        entries = []
        for page in range(1, n // 50 + 1):
            res = s.get(f"{API}/leagues-classic/314/standings/", params={"page_standings": page}, timeout=20).json()
            entries += [r["entry"] for r in res["standings"]["results"]]
            time.sleep(delay)
        own, cap, chips, done = {}, {}, {}, 0
        for e in entries[:n]:
            pk = s.get(f"{API}/entry/{e}/event/{gw}/picks/", timeout=20)
            time.sleep(delay)
            if not pk.ok:
                continue
            pk = pk.json()
            done += 1
            for p in pk.get("picks", []):
                own[p["element"]] = own.get(p["element"], 0) + 1
                if p.get("is_captain"):
                    cap[p["element"]] = cap.get(p["element"], 0) + (2 if pk.get("active_chip") == "3xc" else 1)
            if pk.get("active_chip"):
                chips[pk["active_chip"]] = chips.get(pk["active_chip"], 0) + 1
    except Exception as e:  # noqa: BLE001
        print("Elite sample failed:", e)
        return None
    if not done:
        return None
    out = dict(gw=gw, n=done, own={k: round(v / done, 3) for k, v in own.items()},
               cap={k: round(v / done, 3) for k, v in cap.items()}, chips=chips)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, separators=(",", ":")))
    return out


# ============================================================ set-piece notes
def fetch_set_piece_notes():
    try:
        import requests
        js = fpl_session().get(f"{API}/team/set-piece-notes/", timeout=20).json()
    except Exception as e:  # noqa: BLE001
        print("Set-piece notes not reachable:", e)
        return {}
    return parse_set_piece_notes(js)


def parse_set_piece_notes(js):
    out = {}
    for t in (js.get("teams", []) if isinstance(js, dict) else []):
        notes = [_text(n.get("info_message")).strip() for n in t.get("notes", []) if isinstance(n, dict)]
        notes = [re.sub(r"<[^>]+>", "", n) for n in notes if n]
        if notes and t.get("id") is not None:
            out[int(t["id"])] = notes
    return out


# ============================================================ understat
def fetch_understat(season_start_year):
    """Season xG, xA, xGChain and xGBuildup per player from understat.com (a second opinion).
    Understat now loads its tables through an AJAX endpoint; we ask for it the same way its own pages do."""
    import requests
    url = f"https://understat.com/getLeagueData/EPL/{season_start_year}"
    page = f"https://understat.com/league/EPL/{season_start_year}"
    variants = [
        {"X-Requested-With": "XMLHttpRequest"},
        {"X-Requested-With": "XMLHttpRequest", "Referer": page},
        {"X-Requested-With": "XMLHttpRequest", "Referer": page, "Accept": "application/json, text/javascript, */*; q=0.01",
         "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15"},
    ]
    seen = []
    try:
        with requests.Session() as sess:
            for h in variants:
                if "Referer" in h:
                    sess.get(page, headers={k: v for k, v in h.items() if k == "User-Agent"}, timeout=20)   # pick up any cookies first
                r = sess.get(url, headers=h, timeout=20)
                try:
                    data = r.json()
                except ValueError:
                    data = None
                if isinstance(data, dict) and data.get("players"):
                    return data["players"]
                seen.append(f"HTTP {r.status_code}, {r.headers.get('content-type', '?').split(';')[0]}, starts {r.text.strip()[:30]!r}")
            resp = sess.get(page, timeout=20)
            m = re.search(r"playersData\s*=\s*JSON\.parse\('(.*?)'\)", resp.text)
            if m:
                import codecs
                return json.loads(codecs.decode(m.group(1), "unicode_escape"))
        note_source("Understat (second opinion)", "data endpoint gave " + " | ".join(seen[-1:]) + "; league page had no player table")
    except Exception as e:  # noqa: BLE001
        note_source("Understat (second opinion)", f"{type(e).__name__}: {e}")
    return []


def _ascii(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()


def match_understat(rows, players, teams):
    look = team_lookup(teams)
    by_team = {}
    for p in players.itertuples():
        by_team.setdefault(int(p.team), []).append(p)
    out = {}
    for r in rows:
        team_names = str(r.get("team_title", "")).split(",")
        tid = next((look.get(_team_key(n)) for n in reversed(team_names) if look.get(_team_key(n))), None)
        if not tid:
            continue
        name = _ascii(r.get("player_name", ""))
        last = name.split()[-1] if name else ""
        best = None
        for p in by_team.get(tid, []):
            full = _ascii(f"{_text(p.first_name)} {_text(p.second_name)}")
            web = _ascii(_text(p.web_name))
            if name == full or name == web or (last and (last in full.split() or web.endswith(last))):
                best = p
                if name == full:
                    break
        mins = float(r.get("time") or 0)
        if best is not None and mins > 0:
            per = lambda k: round(float(r.get(k) or 0) / mins * 90, 3)
            out[int(best.id)] = dict(mins=int(mins), xg90=per("xG"), npxg90=per("npxG"), xa90=per("xA"),
                                     chain90=per("xGChain"), build90=per("xGBuildup"), goals=int(float(r.get("goals") or 0)))
    return out


# ================================================================ odds
def _team_key(name):
    k = re.sub(r"[^a-z0-9 ]", "", str(name).lower().replace("&", "and")).strip()
    return re.sub(r"\s+", " ", k)


def team_lookup(teams):
    look = {}
    for r in teams.itertuples():
        for n in (r.name, r.short_name):
            look[_team_key(n)] = int(r.id)
    for alias, target in TEAM_ALIASES.items():
        hit = teams.loc[teams["name"] == target, "id"]
        if len(hit):
            look[_team_key(alias)] = int(hit.iloc[0])
    return look


def _pois(lam, k):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def implied_goals(p_home, p_away, p_over):
    """Expected goals for each side that reproduce the bookmakers' match and over/under 2.5 prices."""
    lo, hi = 0.2, 7.0
    for _ in range(60):                       # total goals from the over/under price
        mid = (lo + hi) / 2
        over = 1 - sum(_pois(mid, k) for k in range(3))
        lo, hi = (mid, hi) if over < p_over else (lo, mid)
    mu = (lo + hi) / 2
    target = p_home - p_away
    lo, hi = 0.03, 0.97
    for _ in range(60):                       # split between the sides from the match odds
        s = (lo + hi) / 2
        lh, la = mu * s, mu * (1 - s)
        ph = [_pois(lh, k) for k in range(12)]
        pa = [_pois(la, k) for k in range(12)]
        diff = sum(ph[i] * pa[j] for i in range(12) for j in range(12) if i > j) - \
            sum(ph[i] * pa[j] for i in range(12) for j in range(12) if i < j)
        lo, hi = (s, hi) if diff < target else (lo, s)
    s = (lo + hi) / 2
    return mu * s, mu * (1 - s)


def _devig(*odds):
    inv = [1 / o for o in odds]
    return [x / sum(inv) for x in inv]


def odds_from_football_data(csv_text_or_df, look):
    df = csv_text_or_df if isinstance(csv_text_or_df, pd.DataFrame) else pd.read_csv(csv_text_or_df)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    df = df[df.get("Div", pd.Series(dtype=str)).astype(str) == "E0"]
    out = {}
    for r in df.to_dict("records"):
        h, a = look.get(_team_key(r.get("HomeTeam"))), look.get(_team_key(r.get("AwayTeam")))
        if not h or not a:
            continue
        for src, H, Dw, A, O, U in (("Betfair Exchange", "BFEH", "BFED", "BFEA", "BFE>2.5", "BFE<2.5"),
                                    ("bookmaker average", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5")):
            vals = [pd.to_numeric(r.get(c), errors="coerce") for c in (H, Dw, A, O, U)]
            if all(v is not None and not pd.isna(v) and v > 1 for v in vals):
                ph, pd_, pa = _devig(*vals[:3])
                po = _devig(vals[3], vals[4])[0]
                lh, la = implied_goals(ph, pa, po)
                out[(h, a)] = dict(lh=round(lh, 3), la=round(la, 3), ph=round(ph, 3), pd=round(pd_, 3), pa=round(pa, 3),
                                   po=round(po, 3), src=f"football-data.co.uk ({src})", date=str(r.get("Date")))
                break
    return out


def odds_from_odds_api(events, look):
    out = {}
    for ev in events:
        h, a = look.get(_team_key(ev.get("home_team"))), look.get(_team_key(ev.get("away_team")))
        if not h or not a:
            continue
        h2h, tot = [], []
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                prices = {o["name"]: o.get("price") for o in m.get("outcomes", []) if o.get("price")}
                if m["key"] == "h2h" and {ev["home_team"], ev["away_team"], "Draw"} <= set(prices):
                    h2h.append(_devig(prices[ev["home_team"]], prices["Draw"], prices[ev["away_team"]]))
                if m["key"] == "totals":
                    o = [x for x in m["outcomes"] if x.get("point") == 2.5]
                    pr = {x["name"]: x["price"] for x in o}
                    if {"Over", "Under"} <= set(pr):
                        tot.append(_devig(pr["Over"], pr["Under"])[0])
        if h2h and tot:
            ph, pd_, pa = (float(np.mean([x[i] for x in h2h])) for i in range(3))
            po = float(np.mean(tot))
            lh, la = implied_goals(ph, pa, po)
            out[(h, a)] = dict(lh=round(lh, 3), la=round(la, 3), ph=round(ph, 3), pd=round(pd_, 3), pa=round(pa, 3),
                               po=round(po, 3), src=f"The Odds API ({len(h2h)} bookmakers)", date=ev.get("commence_time"))
    return out


SCORER_MARKET = "player_goal_scorer_anytime"      # The Odds API: soccer player props, US bookmakers, one event per call
SCORER_DAYS = 4                                   # only fetched this close to the gameweek, to stay inside the free quota


def match_scorer_names(prices, cands):
    """Match bookmaker player names to FPL players in the two teams. prices: {name: [implied chances]}.
    cands: [(pid, full_name, web_name)]. Returns {pid: dict(p=mean implied chance, n=bookmakers)}."""
    full = {_ascii(fn): pid for pid, fn, wn in cands if fn}
    web = {}
    last = {}
    for pid, fn, wn in cands:
        web.setdefault(_ascii(wn), []).append(pid)
        if fn:
            last.setdefault(_ascii(fn).split(" ")[-1], []).append(pid)
    out = {}
    for name, ps in prices.items():
        key = _ascii(name).strip()
        pid = full.get(key)
        if pid is None and len(web.get(key, [])) == 1:
            pid = web[key][0]
        if pid is None:
            tail = key.split(" ")[-1] if key else ""
            if len(last.get(tail, [])) == 1:
                pid = last[tail][0]
            elif len(web.get(tail, [])) == 1:
                pid = web[tail][0]
        if pid is not None and ps:
            out[int(pid)] = dict(p=round(float(min(0.95, max(0.01, np.mean(ps)))), 3), n=len(ps))
    return out


def scorer_prices_from_event(ev):
    """{player name: [implied chance per bookmaker]} from one event-odds reply."""
    prices = {}
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != SCORER_MARKET:
                continue
            for o in m.get("outcomes", []):
                if str(o.get("name", "")).lower() == "no" or not o.get("price") or o["price"] <= 1:
                    continue
                name = o.get("description") or o.get("name")
                if name and str(name).lower() != "yes":
                    prices.setdefault(str(name), []).append(1.0 / float(o["price"]))
    return prices


def fetch_scorer_odds(ds, gw, history):
    """Bookmakers' anytime-goalscorer prices for the coming gameweek, for display beside the model's own chance.
    They are NOT used in the projections. Fetched at most once a day, and only close to the gameweek."""
    key = os.environ.get("ODDS_API_KEY")
    label = "Goalscorer prices (comparison only)"
    if not key:
        return {}
    cache = Path(history) / "scorer_odds.json" if history else None
    saved = None
    if cache and cache.exists():
        try:
            saved = json.loads(cache.read_text())
        except Exception:  # noqa: BLE001
            saved = None
    today = dt.date.today().isoformat()
    same_gw = bool(saved and saved.get("gw") == gw)
    from_saved = lambda: {int(k): v for k, v in saved["odds"].items()} if same_gw else {}
    if same_gw and saved.get("day") == today:
        return from_saved()
    fx = ds.fixtures[(ds.fixtures["event"] == gw) & ~ds.fixtures["finished"]]
    if not len(fx):
        return from_saved()
    first = fx["kickoff"].min()
    days = (first - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400 if pd.notna(first) else 99
    if days > SCORER_DAYS:
        return from_saved()
    look = team_lookup(ds.teams)
    pairs = set(zip(fx.team_h.astype(int), fx.team_a.astype(int)))
    pl = ds.players
    names = {int(r.id): (f"{_text(getattr(r, 'first_name', ''))} {_text(getattr(r, 'second_name', ''))}".strip(), _text(r.web_name), int(r.team)) for r in pl.itertuples()}
    out, seen, with_prices = {}, 0, 0
    try:
        import requests
        base = "https://api.the-odds-api.com/v4/sports/soccer_epl/events"
        r = requests.get(base, params=dict(apiKey=key), timeout=20)
        if not r.ok:
            note_source(label, f"The Odds API refused the events list (HTTP {r.status_code}: {r.text[:80]})")
            SOURCE_STATE[label] = bool(from_saved())
            return from_saved()
        for ev in r.json():
            h, a = look.get(_team_key(ev.get("home_team"))), look.get(_team_key(ev.get("away_team")))
            if not h or not a or (int(h), int(a)) not in pairs:
                continue
            seen += 1
            e = requests.get(f"{base}/{ev['id']}/odds", params=dict(apiKey=key, regions="us", markets=SCORER_MARKET, oddsFormat="decimal"), timeout=20)
            if not e.ok:
                continue
            prices = scorer_prices_from_event(e.json())
            if prices:
                with_prices += 1
                cands = [(pid, fn, wn) for pid, (fn, wn, t) in names.items() if t in (int(h), int(a))]
                out.update(match_scorer_names(prices, cands))
    except Exception as e:  # noqa: BLE001
        note_source(label, f"{type(e).__name__}: {e}")
        SOURCE_STATE[label] = bool(from_saved())
        return from_saved()
    if not out:
        note_source(label, f"no goalscorer prices came back ({seen} matches asked, {with_prices} with prices); The Odds API only has these from US bookmakers, and not for every match")
        SOURCE_STATE[label] = bool(from_saved())
        return from_saved()
    SOURCE_STATE[label] = True
    if cache:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(dict(day=today, gw=int(gw), odds=out)))
        except Exception:  # noqa: BLE001
            pass
    print(f"Goalscorer prices: {len(out)} players from {with_prices} of {seen} matches")
    return out


def fetch_odds(teams, upcoming_pairs):
    """Bookmaker-implied expected goals for upcoming fixtures. The Odds API if a key is set, else football-data.co.uk."""
    import os
    look = team_lookup(teams)
    odds = {}
    key = os.environ.get("ODDS_API_KEY")
    why = []
    if key:
        try:
            import requests
            r = requests.get(ODDS_API, params=dict(apiKey=key, regions="uk", markets="h2h,totals", oddsFormat="decimal"), timeout=20)
            if r.ok:
                got = odds_from_odds_api(r.json(), look)
                odds.update(got)
                print("Odds API requests left this month:", r.headers.get("x-requests-remaining"))
                if not got:
                    why.append(f"The Odds API answered but returned no matches the engine could read ({len(r.json())} events)")
            else:
                why.append(f"The Odds API refused the request (HTTP {r.status_code}: {r.text[:80]})")
        except Exception as e:  # noqa: BLE001
            why.append(f"The Odds API failed ({type(e).__name__})")
    else:
        why.append("no ODDS_API_KEY secret is set")
    fd = None
    try:
        fd = pd.read_csv(FD_FIXTURES)
        fd.columns = [c.lstrip("\ufeff") for c in fd.columns]
        if not odds:
            got = odds_from_football_data(fd, look)
            odds.update(got)
            if not got:
                why.append("football-data.co.uk hasn't published odds for this round yet (usually Thursday or Friday)")
    except Exception as e:  # noqa: BLE001
        why.append(f"football-data.co.uk fixtures file not available ({type(e).__name__})")
    fetch_odds.fixtures_file = fd
    out = {k: v for k, v in odds.items() if k in upcoming_pairs}
    if not out:
        note_source("Bookmaker odds for the next round", "; ".join(why) or "no odds matched the coming fixtures")
    return out


# ================================================================ prices
def price_info(p):
    """FPL's own price-change predictor where the official API provides it, otherwise net transfer pressure."""
    pctv = p.get("price_change_percent")
    if pctv is not None and not pd.isna(pctv):
        proj = p.get("price_change_projections")
        if isinstance(proj, str):
            try:
                import ast
                proj = ast.literal_eval(proj)
            except (ValueError, SyntaxError):
                proj = None
        steps = [float(x.get("projected_percent", 0)) for x in proj][:3] if isinstance(proj, list) else []
        like = int(proj[0].get("likelihood", 0)) if isinstance(proj, list) and proj else 0
        return dict(src="fpl", pct=round(float(pctv), 1), rate=float(p.get("price_change_hourly_rate") or 0),
                    steps=[round(x, 1) for x in steps], like=like,
                    locked=_text(p.get("price_change_locked_until")) or None)
    tin, tout = p.get("transfers_in_event"), p.get("transfers_out_event")
    if tin is None or pd.isna(tin):
        return None
    return dict(src="transfers", net=int(float(tin) - float(tout or 0)), tin=int(float(tin)), tout=int(float(tout or 0)))


# ================================================================ rivals
def import_league(league_id, my_id, cur_gw, limit=20):
    """Rivals in a classic mini-league: their squads, captains and chips (public FPL endpoints)."""
    try:
        import requests
        s = fpl_session()
        get = lambda path: s.get(f"{API}/{path}", timeout=20).json()
        standings = get(f"leagues-classic/{league_id}/standings/")
    except Exception as e:  # noqa: BLE001
        print("Could not import league:", e)
        return None
    return league_from_api(standings, my_id, cur_gw, limit,
                           lambda e: get(f"entry/{e}/event/{cur_gw}/picks/"), lambda e: get(f"entry/{e}/history/"))


def league_from_api(standings, my_id, cur_gw, limit, picks_for, history_for):
    rows = standings.get("standings", {}).get("results", [])
    me = next((i for i, r in enumerate(rows) if r["entry"] == my_id), None)
    keep = set(range(min(10, len(rows))))
    if me is not None:
        keep |= set(range(max(0, me - 5), min(len(rows), me + 6)))
    entries = []
    for i in sorted(keep)[:limit + 1]:
        r = rows[i]
        try:
            picks = picks_for(r["entry"])
            hist = history_for(r["entry"])
        except Exception:  # noqa: BLE001
            continue
        pk = picks.get("picks", [])
        entries.append(dict(
            id=r["entry"], team=r.get("entry_name"), manager=r.get("player_name"), rank=r.get("rank"), last_rank=r.get("last_rank"),
            total=r.get("total"), gw_pts=r.get("event_total"), me=r["entry"] == my_id,
            ids=[p["element"] for p in sorted(pk, key=lambda p: p["position"])],
            cap=next((p["element"] for p in pk if p.get("is_captain")), None),
            vice=next((p["element"] for p in pk if p.get("is_vice_captain")), None),
            chip=picks.get("active_chip"), bank=round(picks.get("entry_history", {}).get("bank", 0) / 10, 1),
            chips_used=[dict(name=c["name"], gw=c["event"]) for c in hist.get("chips", [])],
        ))
    return dict(id=standings.get("league", {}).get("id"), name=standings.get("league", {}).get("name"), gw=cur_gw,
                entries=entries)


LEAGUE_PAGE = 50      # FPL returns league standings 50 managers to a page


def import_all_leagues(my_id, cur_gw, first_id=None, max_leagues=None, around=10):
    """Every private or public league the manager is in, classic and head-to-head (FPL lists both on his entry;
    system leagues such as Overall or his country are left out), with the managers closest to him in each: their
    squads, captains and chips. In leagues bigger than one page of standings, the page he's on is fetched, so
    "closest to you" really is. Leagues he has only just joined have no standings until the next gameweek and
    come back marked as new. History requests are shared across leagues."""
    try:
        s = fpl_session()
        get = lambda path, **params: s.get(f"{API}/{path}", params=params or None, timeout=20).json()
        entry = get(f"entry/{my_id}/")
    except Exception as e:  # noqa: BLE001
        print("Could not list leagues:", e)
        return []
    lg_all = entry.get("leagues", {}) or {}
    leagues = [dict(l, kind="classic") for l in lg_all.get("classic", []) if l.get("league_type") != "s"]
    leagues += [dict(l, kind="h2h") for l in lg_all.get("h2h", []) if l.get("league_type") != "s"]
    if first_id:
        leagues.sort(key=lambda l: (str(l.get("id")) != str(first_id), l.get("entry_rank") or 1e9))
    picks_cache, hist_cache, out = {}, {}, []
    def picks_for(e):
        if e not in picks_cache:
            picks_cache[e] = get(f"entry/{e}/event/{cur_gw}/picks/")
        return picks_cache[e]
    def hist_for(e):
        if e not in hist_cache:
            hist_cache[e] = get(f"entry/{e}/history/")
        return hist_cache[e]
    for l in leagues[:max_leagues] if max_leagues else leagues:
        path = f"leagues-{'h2h' if l['kind'] == 'h2h' else 'classic'}/{l['id']}/standings/"
        try:
            first = get(path)
        except Exception:  # noqa: BLE001
            continue
        st = first.get("standings", {}) or {}
        rows = list(st.get("results", []) or [])
        rank = l.get("entry_rank") or 0
        # big league: fetch the page he's on, and keep the top three from page one
        if rank and rank > len(rows) and st.get("has_next"):
            page = (int(rank) - 1) // LEAGUE_PAGE + 1
            try:
                more = (get(path, page_standings=page).get("standings", {}) or {}).get("results", []) or []
                rows = rows[:3] + [r for r in more if r not in rows[:3]]
            except Exception:  # noqa: BLE001
                pass
        if not rows:
            new = (first.get("new_entries", {}) or {}).get("results", []) or []
            out.append(dict(id=l["id"], name=_text(l.get("name")), kind=l["kind"], new=True, size=len(new), my_rank=None,
                            entries=[], gw=cur_gw))
            continue
        me = next((i for i, r in enumerate(rows) if r["entry"] == my_id), None)
        if len(rows) <= 25:
            keep = set(range(len(rows)))
        else:
            keep = set(range(min(3, len(rows))))
            if me is not None:
                keep |= set(range(max(0, me - around), min(len(rows), me + around + 1)))
        sub = dict(first, standings=dict(results=[rows[i] for i in sorted(keep)]))
        lg = league_from_api(sub, my_id, cur_gw, 60, picks_for, hist_for)
        lg.update(size=("50+" if st.get("has_next") else len(rows)), my_rank=rank or None, kind=l["kind"])
        out.append(lg)
    print(f"Leagues: {len(out)} ({sum(1 for x in out if x.get('kind') == 'h2h')} head-to-head, {sum(1 for x in out if x.get('new'))} not started yet)")
    return out


# ================================================================== main
def next_gameweeks(fixtures, now):
    """The first gameweek whose deadline (90 minutes before its first kick-off) is still ahead."""
    starts = fixtures[fixtures["kickoff"].notna()].groupby("event")["kickoff"].min() - pd.Timedelta(minutes=90)
    future = starts[starts > now]
    first = int(future.index.min()) if len(future) else int(fixtures["event"].max())
    return list(range(first, min(39, first + HORIZON)))


def build(ds, with_backtest=True, history=None, team_id=None, league_id=None, odds=True, prev=None, cache=None,
          online=True):
    global BOOK_WEIGHT
    sources = {}
    done = ds.fixtures[ds.fixtures["finished"]]
    now = max(pd.Timestamp.now(tz="UTC"), done["kickoff"].max() + pd.Timedelta(minutes=1))
    gws = next_gameweeks(ds.fixtures, now)
    long_gws = list(range(gws[0], 39))
    pending = ds.fixtures[~ds.fixtures["finished"] & (ds.fixtures["event"] < gws[0])]
    up = ds.fixtures[~ds.fixtures["finished"] & ds.fixtures["event"].isin(long_gws)]

    # 1. ClubElo and last season: starting points for teams and players
    elo = fetch_elo(ds.teams) if (online and USE_CLUBELO) else {}
    if USE_CLUBELO:
        sources["ClubElo ratings"] = bool(elo)
    if prev is not None:
        build_priors(ds, prev, elo)
    elif elo:
        mean = np.mean(list(elo.values()))
        ds.team_prior = {t: (math.exp(ELO_K * (e - mean)), math.exp(-ELO_K * (e - mean))) for t, e in elo.items()}
    sources["Last season's data (team and player starting points)"] = prev is not None and bool(ds.player_prior)

    # 2. Results, referees and closing odds from football-data.co.uk season files
    ds.results_odds = match_results_file(load_results_file(SEASON, cache), ds) if online else {}
    seasons = [(ds, ds.results_odds)]
    if prev is not None:
        prev.results_odds = match_results_file(load_results_file(PREV_SEASON, cache), prev) if online else {}
        seasons.append((prev, prev.results_odds))
    ref_table, ref_league = referee_table(seasons)
    sources["Referee records (football-data.co.uk)"] = bool(ref_table)

    # 3. Accuracy: this season every run; last season once, then cached
    acc = backtest(ds) if with_backtest else None
    prev_acc = None
    if prev is not None and with_backtest:
        cache_file = Path(history) / f"backtest_{PREV_SEASON}.json" if history else None
        if cache_file and cache_file.exists():
            prev_acc = json.loads(cache_file.read_text())
        else:
            prev_acc = backtest(prev)
            prev_acc["odds_rows"] = prev_acc.get("odds_rows", [])
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(prev_acc, separators=(",", ":"), default=str))
    lab = lab_tuning([tuple(r) for r in (acc or {}).get("stack_rows", []) + (prev_acc or {}).get("stack_rows", [])])
    disagree = dict(this=disagreement_record((acc or {}).get("stack_rows", [])), last=disagreement_record((prev_acc or {}).get("stack_rows", [])))
    sources["FPL Lab score tuning"] = bool(lab)
    rows = (acc or {}).get("odds_rows", []) + (prev_acc or {}).get("odds_rows", [])
    fit = fit_book_weight([tuple(r) for r in rows])
    if fit:
        BOOK_WEIGHT = min(0.9, max(0.3, fit["w"]))
    sources["Closing odds calibration"] = bool(fit)

    # 4. Odds and referees for the coming fixtures
    book = fetch_odds(ds.teams, set(zip(up.team_h, up.team_a))) if odds and online else {}
    sources["Bookmaker odds for the next round"] = bool(book)
    gs = fetch_scorer_odds(ds, gws[0], history) if odds and online else {}
    sources.update(SOURCE_STATE)
    up_refs = upcoming_referees(ds.teams, getattr(fetch_odds, "fixtures_file", None)) if odds and online else {}
    refs = {}
    for pair, name in up_refs.items():
        rt = ref_table.get(name)
        if rt and ref_league:
            refs[pair] = dict(name=name, league=ref_league["pens"], **rt)
        elif ref_league:
            refs[pair] = dict(name=name, league=ref_league["pens"], pens=ref_league["pens"], yel=ref_league["yel"],
                              red=ref_league["red"], fouls=ref_league["fouls"], n=0, raw_pens=0)
    sources["Referee appointments for the next round"] = bool(refs)
    if not refs:
        note_source("Referee appointments for the next round", "not published yet; they appear in football-data.co.uk's fixtures file later in the week")

    # 5. Elite managers, set-piece notes, Understat
    live_ok = getattr(ds, "live_ok", False)
    elite = import_elite(gws[0] - 1, history) if online and live_ok else None
    sources["Top 500 managers' teams"] = bool(elite)
    sp_notes = fetch_set_piece_notes() if online and live_ok else {}
    sources["FPL set-piece notes"] = bool(sp_notes)
    us = match_understat(fetch_understat(int(SEASON[:4])), ds.players, ds.teams) if (online and USE_UNDERSTAT) else {}
    if USE_UNDERSTAT:
        sources["Understat (second opinion)"] = bool(us)
    sources["Match data up to date with the official game"] = not getattr(ds, "behind", 0)

    tm, prof, league, rates, rot, proj = fit_and_project(ds, long_gws, detail=True, simulate=True, book=book, refs=refs)
    # the same projections with each team's home and away splits switched on, for the app's venue view
    global VENUE_IN_MODEL
    VENUE_IN_MODEL = True
    try:
        proj_v = project(ds, tm, rates, prof, gws, keep_detail=False, rot=rot, simulate_gws=False)
        ep_venue = {int(it["p"]["id"]): it["ep"][:HORIZON] for it in proj_v}
        venue_lams = {(r.team_h, r.team_a): blended_lambdas(tm, r.team_h, r.team_a)
                      for r in ds.fixtures[~ds.fixtures["finished"]].itertuples()}
    finally:
        VENUE_IN_MODEL = False
    surprise, calls = trend_signals(ds, tm)
    ttab = team_table(ds, tm)
    pct = percentiles(rates, ["npxg90", "xa90", "box90", "shots90", "dc90", "aerial90", "share_g"])
    form = player_form(ds)
    extras = team_extras(ds, tm)
    tm["prof"] = prof
    team_ctx, player_ctx = narratives(ds, tm, ttab, rates)
    tm["ctx"] = team_ctx
    team_sum, player_sum = plain_summaries(ds, tm, ttab, rates, gws, {int(it["p"]["id"]): it for it in proj})

    att = np.array([tm["att"][t] for t in tm["ids"]])
    dfn = np.array([tm["dfn"][t] for t in tm["ids"]])
    teams = {}
    for i, t in enumerate(tm["ids"]):
        row = ds.teams.loc[ds.teams["id"] == t].iloc[0]
        teams[t] = dict(
            id=int(t), name=row["name"], short=row["short_name"],
            att=round(tm["att"][t], 3), dfn=round(tm["dfn"][t], 3), gen=round(tm["gen"][t], 3),
            att_score=round(100 * (att[i] - att.min()) / ((att.max() - att.min()) or 1)),
            def_score=round(100 * (dfn.max() - dfn[i]) / ((dfn.max() - dfn.min()) or 1)),
            **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in ttab[t].items()},
            prof={side: {c: [round(v["pm"], 3), round(v["idx"], 2), v["most"]] for c, v in prof.get(t, {}).get(side, {}).items()}
                  for side in ("def", "att")},
            **extras.get(t, {}),
            elo=round(elo[t]) if t in elo else None,
            notes=sp_notes.get(int(t), []),
            ctx=team_ctx.get(t), summary=team_sum.get(t), surprise=surprise.get(int(t)),
            pen=round(tm.get("pen", {}).get(t, 0.0), 3),
            rform={k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in tm.get("form", {}).get(t, {}).items()},
        )
    cong = {}
    for (team, g, opp), c in rot["fixture"].items():
        cong.setdefault(int(team), {})[int(g)] = dict(comp=c["comp"], ko=c["ko"], days=c["days"], before=c["before"])
    for t, c in cong.items():
        teams[t]["cong"] = c

    players = []
    for it in proj:
        p, r = it["p"], it["r"]
        nx = it["next"]
        own = float(p.get("selected_by_percent") or 0)
        why = explain(it, teams, ttab, prof, pct, tm) if nx and (it["ep"][0] >= 1.5 or own >= 2) else []
        s = r["season"] or {}
        pid = int(p["id"])
        sim0 = sim_summary(it["sims"][0]) if it["sims"] and nx and r["emins"] >= 20 else None
        haul_all = [round(float((x >= 10).mean()), 3) if x is not None and x.any() else 0.0 for x in it["sims"]]
        haul_by_gw = haul_all[:HORIZON]
        ep_long = it["ep"]
        it["ep"] = it["ep"][:HORIZON]
        rp = rot["player"].get(pid, {})
        players.append(dict(
            id=pid, n=_text(p.get("web_name")), fn=f"{_text(p.get('first_name'))} {_text(p.get('second_name'))}".strip(),
            t=int(p["team"]), pos=r["pos"], price=round(float(p.get("now_cost") or 0), 1), own=round(own, 1),
            st=_text(p.get("status")) or "a", news=_text(p.get("news")), gs=gs.get(pid),
            cards=it.get("cards") if it.get("cards") and it["cards"]["y"] >= it["cards"]["need"] - 2 else None,
            pen=int(p["penalties_order"]) if p.get("penalties_order") is not None and not pd.isna(p.get("penalties_order")) else None,
            ep=it["ep"], ep6=round(sum(it["ep"]), 1), av=[round(a, 2) for a in it["av"]],
            fpl_ep=float(p["ep_next"]) if p.get("ep_next") is not None and not pd.isna(p.get("ep_next")) else None,
            mins=round(r["emins"]), p60=round(r["p60"], 2), share_g=round(r["share_g"], 3), share_a=round(r["share_a"], 3),
            dc_hit=round(r["dc_hit"], 2), bonus90=round(r["bonus90"], 2),
            season={k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()},
            prof={c: round(v, 3) for c, v in r["profile"].items()} if r["profile"] and r["np_xg"] >= 0.3 else None,
            pct={k: v["p"] for k, v in pct.get(pid, {}).items() if v["ranked"]},
            next=dict(xg=round(nx["xg"], 3), xa=round(nx["xa"], 3), cs=round(nx["cs"], 3), dc_p=round(nx["dc_p"], 3),
                      dc_proj=round(nx["dc_proj"], 1), ret=round(nx["ret"], 3), style=round(nx["style"], 3),
                      opp=int(nx["opp"]), home=bool(nx["home"]), lf=round(nx["lf"], 2), la=round(nx["la"], 2),
                      rot=round(nx["rot_factor"], 2), rested=bool(nx.get("rested")), breakdown=nx["breakdown"]) if nx else None,
            sim=sim0, haul=haul_by_gw,
            ep_long=ep_long if r["emins"] >= 20 else None,
            haul_long=haul_all if tc_candidate(r) else None,
            pc=price_info(p),
            el=dict(own=elite["own"].get(str(int(p["id"])), elite["own"].get(int(p["id"]), 0)),
                    cap=elite["cap"].get(str(int(p["id"])), elite["cap"].get(int(p["id"]), 0))) if elite else None,
            us=us.get(int(p["id"])),
            prior=bool(ds.player_prior.get(int(p["id"]))),
            form=form.get(pid, []) if s.get("mins", 0) >= 90 else [],
            rot=dict(n_c=rp.get("n_c", 0), s_c=rp.get("s_c", 0), n_n=rp.get("n_n", 0), s_n=rp.get("s_n", 0),
                     last_cup=rp.get("last_cup")) if rp.get("n_c") or rp.get("last_cup") else None,
            epf=it.get("epf") if r["emins"] >= 1 or (r.get("season") or {}).get("mins") else None,
            bd6=it.get("comp6"), taker=bool(r.get("taker")),
            epv=[round(x, 2) for x in ep_venue.get(pid, [])] or None,
            ctx=player_ctx.get(pid), summary=player_sum.get(pid),
            why=why,
        ))
    w_fpl, kappa = (lab["w_fpl"], lab["kappa"]) if lab else (0.0, 1.0)
    recent = ds.hist[ds.hist["event"] > tm["last_gw"] - 3].groupby(["element", "event"])["event_points"].sum()
    hauled = {int(e) for (e, _), v in recent.items() if v >= 10}
    for x in players:   # FPL Lab points and captain score for the coming gameweek
        model0 = x["ep"][0]
        x["model_ep"] = model0
        # FPL's figure leans on recent points, so straight after a haul it overrates players (tested: it projected 5.8
        # for players with a 10+ score in their last three, who went on to average 3.6; our model said 3.8)
        wf = 0.0 if x["id"] in hauled else w_fpl
        x["hot_fpl_off"] = x["id"] in hauled and x["fpl_ep"] is not None and w_fpl > 0
        x["lab"] = round((1 - wf) * model0 + wf * x["fpl_ep"], 2) if x["fpl_ep"] is not None else model0
        x["ep"][0] = x["lab"]
        x["ep6"] = round(sum(x["ep"]), 1)
        if x["ep_long"]:
            x["ep_long"][0] = x["lab"]
        xgi = (x["next"]["xg"] + x["next"]["xa"]) if x["next"] else 0.0
        x["cap"] = round(x["lab"] + kappa * xgi, 2)
    for x in players:   # keep the file light: players who never feature carry only the basics
        if x["mins"] < 1 and not (x["season"] or {}).get("mins"):
            x.update(season={}, next=None, rot=None, haul=[], pct={}, why=[], form=[], sim=None)
        elif x["rot"] and not x["rot"]["n_c"]:
            x["rot"] = None
    players.sort(key=lambda x: -x["ep6"])

    pend_ids = set(pending["id"])
    fixtures = []
    for r in ds.fixtures[ds.fixtures["event"].isin(long_gws) & ~ds.fixtures["finished"]].sort_values("kickoff").itertuples():
        if r.id in pend_ids:
            continue
        lh, la = blended_lambdas(tm, r.team_h, r.team_a)
        mh, ma = fixture_lambdas(tm, r.team_h, r.team_a)
        b = book.get((r.team_h, r.team_a))
        vl = venue_lams.get((r.team_h, r.team_a), (lh, la))
        fixtures.append(dict(gw=int(r.event), h=int(r.team_h), a=int(r.team_a), lh=round(lh, 2), la=round(la, 2),
                             lhv=round(vl[0], 2), lav=round(vl[1], 2),
                             mh=round(mh, 2), ma=round(ma, 2), book=b, ref=refs.get((r.team_h, r.team_a)),
                             ko=r.kickoff.isoformat() if pd.notna(r.kickoff) else None))
    special = []
    counts = {}
    for f in fixtures:
        for t in (f["h"], f["a"]):
            counts[(t, f["gw"])] = counts.get((t, f["gw"]), 0) + 1
    for g in long_gws:
        doubles = [int(t) for t in tm["ids"] if counts.get((t, g), 0) >= 2]
        blanks = [int(t) for t in tm["ids"] if counts.get((t, g), 0) == 0]
        if doubles or blanks:
            special.append(dict(gw=g, doubles=doubles, blanks=blanks))
    first_ko = min((f["ko"] for f in fixtures if f["gw"] == gws[0] and f["ko"]), default=None)

    data = dict(
        meta=dict(generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), source=ds.source,
                  model_version=MODEL_VERSION, commit=(os.environ.get("GITHUB_SHA") or "")[:7] or None,
                  last_gw=tm["last_gw"], gws=gws, first_kickoff=first_ko, matches=int(len(tm["m"])),
                  pending=[dict(h=int(r.team_h), a=int(r.team_a), gw=int(r.event),
                                ko=r.kickoff.isoformat() if pd.notna(r.kickoff) else None) for r in pending.itertuples()],
                  mu_h=round(tm["mu_h"], 3), mu_a=round(tm["mu_a"], 3), sims=SIM_N,
                  cats={c: CAT_SHORT[c] for c in CATS}, league={c: round(v, 3) for c, v in league.items()},
                  rules=dict(ft_max=5, hit=4, chip_half_end=19, squad_max_per_club=3),
                  horizon=HORIZON, long_gws=long_gws, special=special,
                  unscheduled=getattr(ds, "unscheduled", []),
                  source_notes=SOURCE_NOTES, disagree=disagree, minutes_capped=tm.get("minutes_capped", {}), avail_decay=AVAIL_DECAY if USE_AVAIL_DECAY else None, haul_test=HAUL_TEST, venue=VENUE_TEST, form_test=FORM_TEST, future_test=FUTURE_TEST, breakout_test=BREAKOUT_TEST, trend_calls=calls, home_adv=round(tm["mu_h"] / tm["mu_a"] - 1, 3),
                  odds=dict(n=len(book), src=sorted({v["src"] for v in book.values()}), weight=BOOK_WEIGHT, fit=fit),
                  refs=dict(league=ref_league, n=len(ref_table)), sources=sources, lab=lab,
                  elite=dict(gw=elite["gw"], n=elite["n"], chips=elite["chips"]) if elite else None),
        teams=list(teams.values()), fixtures=fixtures, players=players,
        accuracy={k: v for k, v in acc.items() if k not in ("odds_rows", "stack_rows")} if acc else None,
        accuracy_prev={k: v for k, v in prev_acc.items() if k in ("summary", "calibration")} if prev_acc else None,
        referees=[dict(name=k, **v) for k, v in sorted(ref_table.items(), key=lambda kv: -kv[1]["pens"])],
        live=live_record(history, ds),
        my=import_team(team_id, gws[0]) if team_id else None,
    )
    if data["my"] and data["my"].get("raw"):
        data["my"]["sell"] = selling_prices(data["my"]["raw"], ds, data["my"]["ids"])
        raw_ = data["my"].pop("raw")
        data["my"]["audit"] = season_audit(raw_, ds)
        try:
            data["my"]["luck"] = decisions_vs_luck(raw_, ds, (acc or {}).get("stack_rows", []))
        except Exception as e:  # noqa: BLE001
            print("Decisions vs luck skipped:", e)
    cur = data["my"]["gw"] if data["my"] else gws[0] - 1
    if team_id and online:
        data["leagues"] = import_all_leagues(int(team_id), cur, first_id=league_id)
        if data["leagues"]:
            data["rivals"] = data["leagues"][0]
    if league_id and not data.get("rivals"):
        data["rivals"] = import_league(league_id, int(team_id) if team_id else None, cur)
    return data


def run_changes(folder, data):
    """What moved since the previous run: projections, injury news, prices and fixtures. Saves this run for next time."""
    if not folder:
        return None
    path = Path(folder) / "last_run.json"
    snap = dict(generated=data["meta"]["generated"], gw=data["meta"]["gws"][0],
                special=[x["gw"] for x in data["meta"].get("special", [])],
                players={p["id"]: [p["ep"][0] if p["ep"] else 0, p["ep6"], p["st"], p["news"], p["price"]] for p in data["players"]})
    old = json.loads(path.read_text()) if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, separators=(",", ":")))
    if not old:
        return None
    rows = []
    for p in data["players"]:
        o = old["players"].get(str(p["id"]))
        if not o:
            continue
        e0, e6, st, news, price = o
        if p["news"] != news and (p["news"] or news):
            rows.append(dict(id=p["id"], kind="news", text=p["news"] or "Injury flag removed", w=3 + p["ep6"] / 10))
        if round(p["price"] - price, 1):
            rows.append(dict(id=p["id"], kind="price", text=f"Price {'up' if p['price'] > price else 'down'} to £{p['price']:.1f}m",
                             w=1 + p["own"] / 20))
        if abs(p["ep6"] - e6) >= 3 and max(p["ep6"], e6) >= 15:
            rows.append(dict(id=p["id"], kind="proj", text=f"Next six gameweeks {e6:.1f} → {p['ep6']:.1f}", w=abs(p["ep6"] - e6) / 2,
                             up=p["ep6"] > e6))
    new_special = [g for g in snap["special"] if g not in old.get("special", [])]
    rows.sort(key=lambda r: -r["w"])
    return dict(since=old["generated"], new_gw=snap["gw"] != old["gw"], special=new_special, rows=rows[:25])


def deadline_reminder(data, folder, topic, url=None, now=None):
    """A phone notification about a day before the deadline (via ntfy.sh): captain, squad warnings and price falls.
    Sent once per gameweek; the gameweeks already notified are kept in site/history/notified.json."""
    if not topic or not data.get("my"):
        return None
    ko = data["meta"].get("first_kickoff")
    if not ko:
        return None
    gw = data["meta"]["gws"][0]
    dl = pd.Timestamp(ko).tz_convert("UTC") - pd.Timedelta(minutes=90)
    now = now or pd.Timestamp.now(tz="UTC")
    hours = (dl - now).total_seconds() / 3600
    path = Path(folder) / "notified.json"
    sent = json.loads(path.read_text()) if path.exists() else []
    if gw in sent or not (10 <= hours <= 34):
        return None
    P = {p["id"]: p for p in data["players"]}
    squad = [P[i] for i in data["my"]["ids"] if i in P]
    fit = [p for p in squad if p["st"] == "a"]
    caps = sorted(fit, key=lambda p: -(p.get("cap") if p.get("cap") is not None else p["ep"][0]))
    lines = []
    if caps:
        lines.append(f"Captain: {caps[0]['n']}" + (f" (vice {caps[1]['n']})" if len(caps) > 1 else ""))
    warn = [f"{p['n']} {('injured' if p['st'] == 'i' else 'suspended' if p['st'] == 's' else 'doubtful')}" for p in squad if p["st"] in ("i", "s", "d")]
    warn += [f"{p['n']} rotation risk" for p in squad if p.get("next") and p["next"].get("rot", 1) < 0.9]
    falls = [p["n"] for p in squad if (p.get("pc") or {}).get("src") == "fpl" and ((p["pc"].get("steps") or [0])[0] <= -100)]
    if warn:
        lines.append("Watch: " + ", ".join(warn[:4]))
    if falls:
        lines.append("Price falls tonight: " + ", ".join(falls[:4]))
    if len(lines) == 1:
        lines.append("No injuries or price falls in your squad.")
    body = "\n".join(lines)
    title = f"FPL Lab: GW{gw} deadline {dl.tz_convert('Europe/London').strftime('%a %H:%M')}"
    try:
        import requests
        headers = {"Title": title, "Tags": "soccer"}
        if url:
            headers["Click"] = url
        requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), headers=headers, timeout=15)
    except Exception as e:  # noqa: BLE001
        print("Reminder not sent:", e)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(set(sent + [gw]))))
    print("Deadline reminder sent:", title)
    return body


def json_safe(o):
    """Replace NaN and infinity with null: browsers reject them in JSON, which stops the whole app loading."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.generic):
        return json_safe(o.item())
    return o


def sanity_checks(data):
    """Checks a run must pass before it replaces the live data. Returns (passed, failed) lists of plain-English lines."""
    ok, bad = [], []
    def check(cond, good, fail):
        (ok if cond else bad).append(good if cond else fail)
    ps = data.get("players", [])
    check(len(ps) >= 400, f"{len(ps)} players loaded", f"only {len(ps)} players loaded (expected 400+)")
    gw0 = data["meta"]["gws"][0] if data["meta"].get("gws") else None
    fx = [f for f in data.get("fixtures", []) if f.get("gw") == gw0]
    check(len(fx) >= 6, f"{len(fx)} fixtures in GW{gw0}", f"only {len(fx)} fixtures found for GW{gw0}")
    regular = [p for p in ps if p.get("st") == "a" and (p.get("season") or {}).get("mins", 0) >= 270]
    live = [p for p in regular if p.get("ep") and (p["ep"][0] or 0) >= 1.0]
    share = len(live) / max(1, len(regular))
    check(share >= 0.6, f"{share:.0%} of regular players have a sensible projection",
          f"only {share:.0%} of regular players have a projection of 1+ point (expected most)")
    bad_num = sum(1 for p in ps for x in (p.get("ep") or []) if x is None or x != x or x < 0 or x > 30)
    check(bad_num == 0, "no impossible projections", f"{bad_num} impossible projections (blank, negative or over 30)")
    mc = data["meta"].get("minutes_capped") or {}
    ok.append("every team’s expected minutes fit within 990" if not mc else
              f"{len(mc)} team{'s' if len(mc) != 1 else ''}’ expected minutes scaled down to fit 990")
    acc = (data.get("accuracy") or {}).get("summary") or {}
    if acc.get("mae") is not None:
        check(1.0 <= acc["mae"] <= 3.5, f"backtest error {acc['mae']} is in the normal range",
              f"backtest error {acc['mae']} is outside the normal range (1.0 to 3.5)")
    return ok, bad


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=SEASON)
    ap.add_argument("--cache", default=None, help="folder to cache downloads in")
    ap.add_argument("--no-live", action="store_true", help="skip the official FPL API overlay")
    ap.add_argument("--team-id", default=os.environ.get("FPL_TEAM_ID") or None, help="your FPL team ID")
    ap.add_argument("--league-id", default=os.environ.get("FPL_LEAGUE_ID") or None, help="mini-league ID for rivals")
    ap.add_argument("--no-odds", action="store_true", help="skip bookmaker odds")
    ap.add_argument("--no-prev", action="store_true", help="skip last season's data")
    ap.add_argument("--offline", action="store_true", help="skip every online extra (Elo, odds, referees, Understat)")
    ap.add_argument("--history", default=str(Path(__file__).resolve().parents[1] / "site" / "history"),
                    help="folder of saved pre-deadline projections")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "site" / "data.json"))
    args = ap.parse_args()

    cache = Path(args.cache) if args.cache else None
    ds = load_core(args.season, cache)
    prev = None if args.no_prev else load_core(PREV_SEASON, cache, cups=False)
    if not args.no_live:
        overlay_api(ds)
    data = build(ds, history=args.history, team_id=args.team_id, league_id=args.league_id, odds=not args.no_odds,
                 prev=prev, cache=cache, online=not args.offline)
    write_snapshot(args.history, data)
    data["changes"] = run_changes(args.history, data)
    deadline_reminder(data, args.history, os.environ.get("NTFY_TOPIC"), os.environ.get("PAGES_URL"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    passed, failed = sanity_checks(data)
    data["meta"]["checks"] = dict(passed=passed, failed=failed, at=data["meta"]["generated"])
    if failed:
        # keep yesterday's numbers live; record why, and fail the run so GitHub emails you
        report = dict(at=data["meta"]["generated"], failed=failed, passed=passed)
        if args.history:
            Path(args.history).mkdir(parents=True, exist_ok=True)
            (Path(args.history) / "last_failed.json").write_text(json.dumps(report, indent=1))
        print("Safety net: this run's results look wrong, so the live data was NOT replaced:")
        for f in failed:
            print("  -", f)
        sys.exit(1)
    data = json_safe(data)
    text = json.dumps(data, separators=(",", ":"), default=str, allow_nan=False)
    json.loads(text)                                  # the file must be valid, strict JSON
    out.write_text(text)
    acc = data["accuracy"]["summary"] if data["accuracy"] else {}
    print(f"Wrote {out}: {len(data['players'])} players, GW{data['meta']['gws'][0]}-{data['meta']['gws'][-1]}, "
          f"data to GW{data['meta']['last_gw']}. Backtest MAE {acc.get('mae')} vs FPL {acc.get('mae_fpl')}. "
          f"Team import: {'yes' if data['my'] else 'no'}")


if __name__ == "__main__":
    main()
