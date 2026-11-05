# FPL Lab: setup guide

Your own FPL projections app, with the reasoning behind every number. A script
pulls fresh data several times a day, runs the model, simulates every player's
gameweek, checks its own accuracy, and publishes the app to a free website on
GitHub Pages.

## Where the data comes from

- **FPL Core Insights** (github.com/olbauday/FPL-Core-Insights): the official
  FPL data plus match stats, every shot with its location, body part and
  situation, and cup and European minutes. Refreshed at 07:30 and 17:30 UTC.
  Please credit it if you share the app.
- **The official FPL API**: on every run, prices, ownership and injury news are
  overwritten with live values, and your own team is imported if you've set
  your team ID.

## What's in here

- `engine/fpl_engine.py` is the model. It rates every team, builds shot-zone
  profiles, models rotation around cup and European games, projects every
  player for 6 gameweeks, simulates each player's next gameweek 3,000 times,
  writes the explanations, backtests itself against FPL's own projections,
  and writes `site/data.json`.
- `site/index.html` is the app.
- `site/vendor/` holds the HiGHS optimiser (MIT licence) that the exact transfer solver runs on, in your browser.
- `site/history/` holds the projections saved before each deadline. The live
  accuracy log scores these once each gameweek is played. Keep this folder.
- `.github/workflows/update.yml` tells GitHub to rerun everything on a schedule.

## Extra free sources (no setup needed)

Every GitHub run also tries these. If one is unreachable, the run carries on
without it and the Model tab lists what was used.

- **Last season's FPL Core Insights data**: starting points for every returning
  player and team, and a full-season backtest used to tune the Lab score.
- **football-data.co.uk season files**: referee records (penalties, cards) and
  closing odds, used to set how much weight the bookmakers get.
- **Top 500 managers' squads** from the official FPL site, sampled once per
  gameweek and saved in `site/history`, for elite ownership and captaincy.
- **ClubElo** ratings, to rate promoted sides and set early-season starting
  points.
- **FPL's official set-piece notes** for each club.
- **Understat** season xG and xA, shown as a second opinion next to ours.

## Put it online (about 10 minutes, easiest on a computer)

1. Create a free account at github.com.
2. Click **New repository**. Name it `fpl-lab`, set it to **Public**, tick
   **Add a README file**, and create it.
3. Click **Add file > Upload files**. Drag in the `engine` folder, the `site`
   folder (including `site/history`) and `requirements.txt`. Commit.
4. The workflow file is in a hidden folder, so add it by hand: click
   **Add file > Create new file**, type `.github/workflows/update.yml` as the
   name, paste in the contents of that file from this zip, and commit.
5. Go to **Settings > Pages**. Under **Source**, choose **GitHub Actions**.
6. Go to the **Actions** tab, click **Update FPL data**, then **Run workflow**.
7. Your app is at `https://YOUR-USERNAME.github.io/fpl-lab/`. Add it to your
   phone's home screen from Safari's share menu.

## Load your own team

1. Find your team ID: open your Points page on the FPL website. The number
   after `/entry/` in the address is your ID.
2. In your repository, go to **Settings > Secrets and variables > Actions**,
   open the **Variables** tab, and click **New repository variable**.
   Name: `FPL_TEAM_ID`. Value: your ID.
3. Run the workflow again. The Squad tab will offer to load your team, with
   your bank, free transfers and remaining chips filled in. You can still
   edit any of them by hand.

FPL doesn't publish selling prices, so the planner uses current prices. Your
selling price on a player who has risen will be a little lower.

## Add your mini-league

Open your league on the FPL website; the number after `/leagues/` in the
address is the league ID. Add it as another repository variable called
`FPL_LEAGUE_ID`. The Squad tab's Rivals view then shows the top ten and the
five managers either side of you: their squads, captains, chips left, the
players they own that you don't, and projected points against yours.

## Bookmaker odds (optional, free)

Without any setup, the app reads the free weekly odds file from
football-data.co.uk (published on Friday afternoons for weekend games and
Tuesdays for midweek ones). For odds on every run, get a free key from
the-odds-api.com (500 requests a month; the app uses about 70) and add it under
**Settings > Secrets and variables > Actions > Secrets** as `ODDS_API_KEY`.
Odds are turned into expected goals and blended 60/40 with the model for that
gameweek's fixtures.

## Schedule

The workflow runs at 08:00 and 18:00 UTC every day, plus three extra runs on
Fridays for late team news and the weekend odds. Press **Run workflow** any time, for example right
after a press conference. GitHub pauses scheduled runs in repositories with no
activity for 60 days; the weekly history saves count as activity.

## Run it on your own computer

    pip install -r requirements.txt
    python engine/fpl_engine.py --cache .cache --team-id 1234567
    cd site && python -m http.server 8000

Then open http://localhost:8000.

## Rules this season (2026/27)

Two sets of chips (Wildcard, Free Hit, Bench Boost, Triple Captain); the first
set must be played before the GW19 deadline. Up to five free transfers can be
banked; extra transfers cost 4 points. £100m budget, three players per club.
Scoring values are at the top of `fpl_engine.py`; check them each summer.

## Deadline reminder on your phone (optional)

1. Install the free **ntfy** app (iPhone or Android) and subscribe to a topic name only you know, e.g. `fpl-lab-` followed by random letters.
2. In GitHub: Settings > Secrets and variables > Actions > Secrets > New repository secret. Name `NTFY_TOPIC`, value your topic name.

About a day before each deadline you'll get one notification: captain and vice, squad injuries and rotation risks, and price falls. Tap it to open FPL Lab.

## Live gameweek

`.github/workflows/live.yml` runs every 15 minutes during usual match times and only does work while a gameweek is in progress. It writes `site/live.json` (your live points, provisional bonus and auto-subs), shown at the top of the Week tab.

## Install on your phone

Open the site in Safari, tap Share, then "Add to Home Screen". It opens full screen with its own icon and keeps the last data for offline use.
