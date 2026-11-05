# FPL Lab: handover notes for Claude Code

Charles's own Fantasy Premier League app. A Python engine pulls free data,
projects every player, explains every number in plain English, checks its own
accuracy, and writes `site/data.json`. A single-page app (`site/index.html`)
displays it. GitHub Actions runs the engine on a schedule and publishes the
site on GitHub Pages.

This project was built in a claude.ai chat, where nothing could reach the
internet except GitHub. **Several data sources have never had a real run**
(see "Not yet tested live" below). Verifying them is the first job here.

## Layout

- `engine/fpl_engine.py`: the whole engine (about 2,100 lines). Main entry:
  `build()`; command-line options in `main()`.
- `engine/build_preview.py`: inlines `data.json` into `index.html` to make a
  single shareable file.
- `site/index.html`: the app (vanilla JS, no build step). Loads `data.json`,
  or `window.FPL_DATA` when inlined.
- `site/history/`: saved pre-deadline projections (`gwN.json`), the elite
  manager sample (`elite_gwN.json`) and last season's backtest cache. The
  live accuracy log depends on these: never delete them.
- `.github/workflows/update.yml`: scheduled runs; commits `site/history`.

## Running

    pip install -r requirements.txt
    python engine/fpl_engine.py --cache .cache            # full run
    python engine/fpl_engine.py --cache .cache --offline  # skip online extras
    cd site && python -m http.server 8000

Options: `--team-id`, `--league-id` (or the `FPL_TEAM_ID` and `FPL_LEAGUE_ID`
environment variables), `ODDS_API_KEY` environment variable (optional),
`--no-live`, `--no-odds`, `--no-prev`. A full run takes a few minutes; the
last-season backtest is cached in `site/history`.

## Data sources

1. FPL Core Insights (github.com/olbauday/FPL-Core-Insights): the backbone.
   Match stats, every shot (location, body part, situation), cup and European
   minutes. This season and last. Tested.
2. Official FPL API: live prices, ownership, news, fixtures (rearranged and
   unscheduled games), FPL's price predictor, the user's team, mini-league
   rivals, top-500 manager sample, set-piece notes.
3. football-data.co.uk: next-round odds and referees (`fixtures.csv`), past
   seasons' results with referees, cards and closing odds (`mmz4281/YYYY/E0.csv`).
   The odds parser was tested on real rows of `fixtures.csv`.
4. The Odds API (optional key): odds on every run.
5. ClubElo (api.clubelo.com): team ratings for starting points.
6. Understat: second-opinion xG, shown only; it doesn't change projections.

## Not yet tested live

These were written against documented or observed formats and unit-tested on
sample data only. On the first real run, check each one works and fix any
parsing problems:

- Official FPL API: `overlay_api` (prices, news, fixtures overlay),
  `import_team`, `import_league`, `import_elite`, `fetch_set_piece_notes`,
  `price_info` with FPL's `price_change_*` fields.
- football-data.co.uk season files and the referee table, plus
  `fit_book_weight` (closing-odds calibration).
- ClubElo `fetch_elo`.
- Understat `fetch_understat` (the site's structure may have changed; it
  fails safely).
- The Odds API reader.

The Model tab lists which sources each run actually used (`meta.sources`).

## Model decisions (all evidence-based; keep it that way)

- Team ratings: recency-weighted xG (25% real goals), opponent-adjusted,
  league-wide home advantage, last-season and ClubElo starting points.
- Player projections: share of team xG and xA, expected minutes, DEFCON,
  bonus, saves, cards, Poisson points; 3,000 simulations per player for haul
  and blank chances.
- Style matchup (shot zones, headers, set pieces, fast breaks): capped at
  ±15%. This is the least proven part. The Tavernier captaincy case rests on it
  partly, so watch it in the live log.
- **Home and away splits are context only.** Tested on 369 matches; team-
  specific splits made goal predictions worse at every strength, so
  `VENUE_IN_MODEL = False`. They appear in every explanation but don't move
  numbers.
- **Lab score.** Lab points are 60% model and 40% FPL's projection; the
  captain score adds 3 × expected goals and assists. Both were picked by
  backtesting 35 gameweeks of 2025/26 (Lab points error 1.86, model alone 1.90,
  FPL 2.06; the captain pick averaged 6.3 points against FPL's top pick at
  4.8).
- Bookmaker odds are blended at 60% for the coming round; the weight is
  meant to be re-fitted from closing odds once those files load.

## Writing style (Charles cares about this)

- Plain, natural English explanations with the actual numbers and ranks in
  them, for example "2nd of 97 midfielders" or "the most in the league".
- Sentence case. Typographic apostrophes and quotes in user-facing text (’ “ ”).
- Honest: say when something is untested, a small sample, or made
  predictions worse.

## Changes since the first upload (v5.2)

- Engine: `narratives()` writes plain-English reasons behind team and player
  numbers (schedule faced, one game skewing an average, where chances come
  from, red cards, missing players, trends, finishing luck). Short versions
  are appended to the Fixture and Clean sheet reasons as "Why: ...".
- App: season-total xG, xA and xG plus xA sorts; filters in a bottom sheet
  opened from a floating button; sticky Close and action buttons at the
  bottom of every sheet; icon tab bar; bigger tap targets; tapping the
  current tab scrolls to the top.
- The live repo already had a fix in `overlay_api` (API numbers sent as
  text); this version keeps it.

## Changes in v6

- Exact solver: `buildLP`/`readPlan` in `site/index.html` build a mixed-integer
  programme (squad, XI, captain, transfers, banked free transfers, hits, bank)
  and solve it in the browser with HiGHS (WebAssembly, `site/vendor/`, MIT
  licence). Captaincy in week one uses the Lab captain score. A stress test
  re-solves six times with ±20% noise. If WebAssembly can't load, the app falls
  back to the quick beam-search planner and says so.
- Pitch view with team colours (plain coloured shirts, no crests), gameweek
  stepper, captain/vice and bench swaps that keep a legal formation.
- Your minutes: per-player overrides stored on the device. Uses `epf` (each
  player's projection at 88 minutes) from the engine to rescale points.
- Season review (needs `FPL_TEAM_ID`): `season_audit()` scores every transfer
  against keeping the player, every captain against the best in the XI,
  bench points and rank by week.
- "In plain terms" summaries for every team and regular player
  (`plain_summaries()`), plus a tap-for-meaning glossary.
- "What changed" card: since the user's last visit (stored on the device),
  or since the previous run (`site/history/last_run.json`, `run_changes()`).
- Percentile radar on player sheets and in head-to-heads.

## Changes in v6.1

- Teams tab: form league table (all, home or away; last 5, last 8 or season;
  real or expected points), built in the app from each team's `results`.
- Team sheets open with a plain-English outlook: model ratings against real
  league position, whether results match performances, and what the next
  fixture means for their attackers and defenders.

## Changes in v6.2 (all in site/index.html)

- Player sheets open with a "this week" paragraph (opponent, share of team
  xG, rank among his position, clean sheet odds, likely scores).
- Week tab: In form (players' xG, xA, xG plus xA or points over the last 4, 5
  or 8 games, against their season rate).
- Teams tab: head to head between any two clubs, with the likely result if
  they meet in the next six gameweeks.
- Fixtures tab: likely scores for every match, and fixture swings (next
  three against the last three, and the three after that).

## Changes in v6.3 (site/index.html)

- Planner rules, saved on the device: keep (locked in every plan), sell by
  GWx, buy by GWx, never buy. Set from any player sheet. Passed to the solver
  as `forced`, `sellBy`, `buyBy` and `banned`; `maxHits` and the horizon
  (3, 4 or 6 gameweeks) are options too. Infeasible rules show a message.
- Drafts: save a solved plan with its rules, compare drafts by gain, reload a
  draft's rules.
- The solver now starts a fresh HiGHS instance per solve and retries with a
  smaller player pool if the WebAssembly build crashes (it occasionally does
  on larger models, and with mip_rel_gap below 0.005).

## Changes in v6.4 (planner)

- Drafts are workspaces (Draft A, B, ...), each with its own locks, bans,
  sell-by and buy-by rules, chip choices and settings, saved on the device.
  The planner panel in the Squad tab has Lock and Sell-by on every squad
  player, and a search to Buy-by or Ban anyone else. Player sheets and the
  pitch have the same controls.
- Chips in the solver: Triple Captain, Bench Boost, Free Hit (separate
  one-week squad) and Wildcard, each "off", "solver decides" or a fixed week.
- Settings: hits (allowed, at most one, none), horizon (3, 4, 6), how much
  later weeks count (decay 1, 0.9, 0.8), value of a banked free transfer.
- Alternatives: the next two plans with a different first week (no-good
  cuts on week-one buys). Per-move "why" bars show the points each move adds
  week by week. Draft comparison table.
- Selling prices: `selling_prices()` in the engine (from FPL transfer
  history, needs FPL_TEAM_ID) or "What you paid" on a player sheet.
- HiGHS upgraded to 1.15.4 (1.8.0 crashed on chip models). The inlined
  preview loads it with `instantiateWasm`; the hosted site uses
  `vendor/highs.wasm`. Each solve uses a fresh instance and retries with a
  smaller player pool on failure.

## Changes in v6.5

- Adjusted xP: `formMult()` = team xG per game over its last 4 matches
  (`results`) / league average over the same games; `adjXP()` = Lab points ×
  that. Shown on player sheets, as a sort and on the Teams list. Display only:
  the main projection already weights recent team form, so it isn't fed back in.
- Split difficulty: `fdrAtt()` from the opponent's season xG conceded per game,
  `fdrDef()` from their xG created, each banded 1-5 by rank. "Split" is now
  the default Fixtures view; a two-part badge appears on player rows and sheets.
- Variance spotter: `variance()` over the last 4 games (form rows carry
  started, goals, assists): flags "Due a return" when xG+xA >= 1.0 and the
  expected goal/assist points exceed actual by >= 4. Sort, Week tab list, row
  highlight.
- Minutes security: `minutesSecurity()` traffic light from starts and minutes
  over the last 5, plus injury and suspension flags, on rows, sheets, the pitch
  and the planner.
- Engine: `fpl_session()` shared FPL API session with retries and backoff.

## Changes in v6.6

- Penalties separated: player shares use non-penalty xG; each team's penalty
  xG per match (`penalty_rates()`, shrunk) goes only to the current
  first-choice taker (`taker`). Points breakdowns have a separate "pens".
- Recent form tested (`FORM_TEST`): extra last-4 form weight (FORM_W_ATT /
  FORM_W_DEF) and heavier recency both predicted goals worse on 370 matches,
  so they stay at 0; team ratings now use TEAM_DECAY = 0.95 (slightly better).
  `recent_form()` ratios are exported as `rform` for the app's what-if.
- App: slice × pie on player sheets, sorts and head-to-heads; team trend
  words (strong/average/weak attack, hot to cold); Adjusted xP now a what-if
  applied to attacking points only, using recent form against the team's own
  rating; plain-number fixtures (expected goals and clean sheet %) replacing
  the 1-5 bands; "why A beats B" from six-week component breakdowns (`bd6`).

## Changes in v6.7 and v6.8

- Glanceable visuals: trend pills (hot to cold) with a four-game form strip;
  trend arrows on player rows; points split into a coloured bar (rows, sheets,
  six-week view); "If form continues" two-bar card; form map (attack quality
  against recent trend) in the Teams tab; colour key from the header.
- Fixture grades as phone-signal bars (Att = goals expected, Def = clean
  sheet chance; five bars best) on rows, sheets, the pitch and the ticker.
- What-if switches in the header: Form (recent form continues, attacking
  points only) and Home/away. The engine runs a second projection pass with
  VENUE_IN_MODEL = True (`epv` per player, `lhv`/`lav` per fixture) for the
  Home/away view; the default projections keep it off, as tested.
- xGC: home and away xG and xGC on team rows and sheets, sorts by home and
  away xGC, and venue sentences on player sheets.
- Trends tab (Trend spotter): next-four forecast vs last four (dumbbell),
  heating up / cooling down with reasons, players to get ahead of, surprise
  runs (`trend_signals()` surprise = last two games against forecast) with
  BREAKOUT_TEST evidence, and a track record of this season's calls.
  FUTURE_TEST: rises of 0.3+ xG a game called right 89% of the time.

## Changes in v6.9

- Header switches removed. The Week tab opens with a "This week" strip:
  deadline countdown (first kick-off minus 90 minutes) and four tappable
  cards: captain, best move, heating up, watch out (`thisWeekStrip()`).
- What-ifs inline: player rows show a Form or Home/Away record chip only when
  that view would move him by 0.5 points or more (`whatIfChips()`); player
  sheets show all three numbers in one card (`scenarioCard()`).
- The whole-app Form and Home/away switches now live in the Model tab; when
  one is on, a quiet line under each page title says so.
- Colour key opens from an "i" button beside every page title.
- Quieter rows: reason chips and the thin points bar are lighter so the name,
  minutes light, signal bars and the number lead.

## Changes in v6.10

- Player sheets: "Expected this week" card (`plainCard()`): one solid bar for
  the projection, a lighter band for the form and home/away what-ifs, and a
  plain sentence for each, plus a trust hint.
- Verdicts (`buildVerdicts()`): Buy / Watch / Don't buy for players you don't
  own, Hold / Sell for yours, with Strong or Lean and a short reason. Based on
  rank among same-position players within ±£0.5m over six weeks, whether a
  cheaper nailed player projects more, affordable upgrades (owned), minutes,
  rotation, fixture trend, price falls and the active draft's plan. Respects
  planner bans. Shown on sheets, rows, the pitch (Sell badge) and as a filter.
- Teams page: compact tiles (attack/defence bars, trend pill, next three
  fixtures for A and D, one FPL call); all other detail moved to the team
  sheet. New Quality map (attack against defence). Sort is now a dropdown.

## Changes in v6.11

- Install as an app: `site/manifest.json`, icons, `site/sw.js` (network first,
  falls back to the cached copy offline; never caches live.json).
- Squad tab: fixture grid for the whole squad (next six; Att colours for
  MID/FWD, clean sheet colours for GK/DEF) and a chip calendar to the end of
  the first half (best TC, BB and FH weeks from `chipPlan()`).
- This week strip: captain confidence meter (gap between the top two Lab
  captain scores), rival threat card (mini-league owners, or top-500 owners
  when no league is set), live card when `site/live.json` is recent, and a
  "How did we do?" review: a forecast snapshot is saved on the device before
  each deadline and scored from the players' match rows once the gameweek is in.
- Player sheets: better options at the same position and price under Sell,
  Don't buy and Watch verdicts. What-changed card includes verdict changes.
- `engine/live.py` + `.github/workflows/live.yml`: every 15 minutes in match
  windows, only while a gameweek is in progress; live points, provisional
  bonus (FPL tie rules) and auto-subs.
- `deadline_reminder()`: one ntfy.sh notification 10 to 34 hours before each
  deadline, if the NTFY_TOPIC secret is set; sent gameweeks kept in
  site/history/notified.json.

## Changes in v6.12

- Pick score (`buildPick()`, PICK_W): z-scores within position of value over a
  typical nailed player within ±£0.5m (ep6 minus their median), timing (next
  three against the three after), minutes (starts and minutes in the last 5)
  and ceiling (share of team xG + xA), weighted 0.745 / 0.147 / 0.065 / 0.037,
  then a percentile against regular starters (3+ starts in the last 5).
  Weights fitted on 2025/26 (nine cutoffs, six-week targets, held-out weeks):
  85+ beat others at their price by 4.6 points over six weeks; underlying
  form and price momentum added nothing.
- Verdicts read off the score: Buy 85+ with green minutes and no rotation flag,
  Watch 65 to 84 (or 85+ with a minutes or rotation issue, with a trigger),
  Don't buy under 65. Sell when an affordable replacement adds 3+ points more
  than a free transfer (1.5), or out, or rarely starting, or in the plan.
- Sheet shows a score ring (segments for each part, coloured good/okay/weak).

## Changes in v6.13

- "Where his points come from" on player sheets is now blocks (`pointBlocks()`):
  half a point per block this week, one point per block over six weeks, rows
  sorted biggest first. Dashed outline blocks show what he's missing against
  a typical nailed player at his position; a thin line marks where a typical
  one stops when he's above (`typicalBreakdown()`).
- Player type tag (`playerType()`): Goal threat, Creator, Clean-sheet
  defender (with a goal threat), Attacking defender, DEFCON earner, Bonus
  magnet, Save-heavy or Clean-sheet keeper, All-rounder. Shown on the card and
  on player rows (replacing the thin points bar).
- One plain sentence on the main source of points and his floor if he
  doesn't return.

## Changes in v6.14

- Players tab: "Pick score" sort (ties by six-week points); every sort can be
  flipped by tapping it again in Sort and filter, or with the arrow button
  beside the summary row (`state.pl.dir`); when sorting by pick score the list
  is grouped under Buy, Watch, Hold, Sell and Don't buy headers.

## Changes in v6.15

- `note_source()` / SOURCE_NOTES: every data source that isn't used records
  why (HTTP status, missing secret, not published yet) in `meta.source_notes`;
  the Model tab shows the reason and what each source is for.
- ClubElo fetch now tries https then http, today then yesterday, through the
  shared session, and reports when team names don't match.
- Model tab: "Where team ratings come from right now" bar (this-season share
  = effective decayed matches / (that + TEAM_PRIOR_MATCHES)). Prior weight
  re-tested on this season's matches (GW3-9 and GW7-9): 4 matches is about
  optimal for goals; no prior was worse.

## Changes in v6.16 (draft on the pitch)

- Pitch shows the active draft's squad for the selected gameweek
  (`draftIdsAt()`), with In and Sell badges on the shirt's corner and one
  signal bar per player (Def for GK/DEF, Att for MID/FWD).
- Tap a player > "Transfer out": replacements at that position within
  selling price + draft bank, club limit and bans respected, sorted by pick
  score with points gained to the end of the horizon (`openTransferOut()`).
- Draft moves are stored per draft (`RULES.moves`); `draftState()` tracks free
  transfers (FPL roll-over rules), hits and bank by gameweek; the draft bar
  shows them with the six-week points change. Undo, reset, "Let the solver
  finish this" (moves become buy-by/sell-by rules via `draftRules()`), and
  "Make GW moves my squad".

## Changes in v6.17

- Understat: requests getLeagueData the way the maintained understatapi fork
  does (session, X-Requested-With, then with Referer and a browser
  User-Agent), parses with r.json(), and records the status, content type
  and first characters of the reply if nothing parses.
- ClubElo: the last good ratings are saved to site/history/elo_last.json and
  used for up to 14 days when the site is down (noted on the Model tab).

## Changes in v6.18

- USE_CLUBELO = False: an Elo built from every result since August 2025 made
  this season's predictions slightly worse (goals log-likelihood -1.4863 v
  -1.4822 on GW3-9, -1.3733 v -1.3705 on GW7-9), and ClubElo returned 502.
- USE_UNDERSTAT = False: its data address returns an HTML page to GitHub's
  servers; it only fed the second-opinion figures. The Understat sort is
  hidden when there's no Understat data. Both switches can be turned back on.

## Changes in v6.19

- AVAIL_DECAY (GK 3%, DEF 4%, MID 4.5%, FWD 3.5% per week ahead, floor 78%):
  applied to weeks two onwards only, from the fall in the share of expected
  starters still playing 60+ minutes each week ahead (2025/26). Next week's
  projection is unchanged.
- Form rows now also carry touches, accurate passes and box touches; the app's
  role tracker (`roleTracker()`, midfielders only) labels each game deep, mixed
  or advanced from passes per 90 against the midfield median. Display only.
- UI: sheets size to the visible screen (dvh) and always open at the top;
  equal-height footer buttons; map labels placed to avoid overlaps with a halo;
  compact form table; fixture cells show opponent plus two coloured numbers.

## Changes in v6.20 (backtest fault fixed, blend re-tuned)

- Fault: 2025/26 snapshots up to GW10 store FPL's blank "chance of playing"
  as 0, so `availability()` treated available players as unavailable in
  backtests (next-week projections near zero after GW6 and GW10). Fixed:
  status "a" means fully available. Live runs were not affected.
- site/history/backtest_2025-2026.json regenerated: model now beats FPL in
  35 of 35 gameweeks (MAE 2.05 v 2.29; was 27 of 35).
- Re-tuned from the clean backtest: FPL weight in Lab points 5% (was 40%),
  captain kappa 0 (plain Lab points picked best: 7.3 a week v FPL's 5.1).
- HAUL_TEST: for players with a 10+ score in their last three, FPL projected
  5.77, our model 3.81, actual 3.55; best FPL weight 0. FPL's figure is left
  out for those players (`hot_fpl_off`).

## Changes in v6.21

- `lab_tuning()` now picks the FPL weight by average error (the measure the
  app reports): 0% on the clean backtest (0% beat 5% in every one of 2,000
  resamples of the 35 gameweeks; 5% was marginally better only on squared error).
- Captain bonus (kappa): the smallest setting within 0.5 points a week of the
  best is chosen, because ~35 picks can't separate smaller gaps; currently 0.
- Minutes lights also turn yellow when the model's rotation factor for the
  coming gameweek is under 0.85.

## Changes in v6.22

- Captain section: swipeable cards (FPL Lab pick, The easy choice, Ceiling,
  Gain rank, Top 500's pick; `CAP_MODES`, `captainSwipe()`), each with an
  outcome bar and a plain sentence; tapping one drives the full captain card
  and list below. "Captain compass" (`openCompass()`) opens in a sheet.
  Top 500's pick uses top-500 captains once FPL reveals them (elite gw equals
  the coming gameweek), otherwise top-500 ownership, otherwise overall.
- Outcome tags (`outcomeTags()`): Ceiling (top-15% haul chance for his
  position, 12%+ haul, blank under 30%), Boom or bust (top-30% haul chance,
  blank 35%+), Safe floor (blank 22% or less).
- Safety net (`sanity_checks()`): player count, fixtures, share of regular
  players with a projection of 1+, impossible values, backtest error range.
  On failure the engine writes site/history/last_failed.json, doesn't touch
  data.json and exits 1, so the deploy is skipped and GitHub emails the owner.
  Results show on the Model tab.
- Settings export and import (all fpllab.* local storage) on the Model tab.

## Changes in v6.23

- Home (was Week): greeting with first name (`D.my.first_name`), opens on
  Home unless the page is refreshed; swipe row of cards (`homeCards()`) in
  priority order: deadline check, price falls, watch out, your captain (from
  your squad, naming the best overall pick if you don't own him), chip tip in
  its week, best move, league edge, rising prices, best fixtures,
  differential, chip tip, heating up.
- Captains: "From my squad / Anyone" switch (`state.capSrc`); a note when the
  overall pick isn't in the squad.
- Live card becomes a "GW review" once live.json has done=true, until the
  next deadline (live data is accepted at any age; the card decides).
- Rotation: a midweek game already played before the league match uses what
  the player actually did. Rested (under 30 minutes) removes the rotation
  penalty and flag. Boosting rested players was tested on this season's
  backtest and made it worse (no boost 2.080 MAE, a quarter 2.088, a half
  2.099), so RESTED_BOOST = 0.
- Leagues: `import_all_leagues()` reads every non-system classic league from
  the manager's entry (up to 8), keeping everyone in leagues of 25 or fewer,
  otherwise the top 3 plus 10 either side of the manager; requests are cached
  across leagues. `D.leagues`, with `D.rivals` = the chosen one (dropdown in
  Rivals, remembered in local storage). Rivals shows rank, gap to the
  manager above, the leader's gap and weak spot, and "To get ahead" picks.
- Home no longer has Differentials, Luck and In form; the Players tab has
  quick views (Differentials, In form, Due a return) and player sheets show
  status chips.
- Bookies view: match cards (win/draw/loss bar, odds and model goals bars, a
  plain sentence on what the disagreement means). Luck table: diverging bars
  with Lucky/Unlucky tags. Form table uses Over-/Under-performing.

## Changes in v6.24

- Player rows: no left bar for "Due a return" (the tag says it); position
  labels (`posLab()`, GK amber, DEF green, MID blue, FWD red) in rows and
  sheet headers; small pills no longer wrap.
- Club accents (`clubCol()`): a team-colour strip on captain cards, the
  captain card, player sheet headers, and a two-team strip on bookies cards.
- Form table tags: "▲ Over" / "▼ Under" / Strong / Steady / Poor with a short
  line underneath (expect a drop, expect a rise, and deserved...).

## Next tasks

1. Run the engine with full internet access and fix any source that fails.
2. Help Charles create the GitHub repository, add the workflow, set
   `FPL_TEAM_ID`, `FPL_LEAGUE_ID` (and optionally the `ODDS_API_KEY` secret),
   and turn on GitHub Pages (Source: GitHub Actions).
3. After three or four live gameweeks, read the live accuracy log and improve
   whatever it shows is weakest (minutes, goal involvements, or the style
   matchup).
4. Ideas parked for later: importing his FPL Review export on-device (keep it
   out of the public repo), and a free weekly message with the captain pick,
   transfer plan and price alerts.
