# Possession labeling for Barcelona v Athletic Club

Team Data Kickers. Code: https://github.com/rajaxar/barca-possession

We label every frame of the match `H`, `A`, `D` or `X` with a deterministic pipeline built on the provided
tracking data. The submitted file has 199,319 rows, 370 possessions and 138 dead-ball stretches; its
sha256 is `ee7b5b6bdf7404e82696654ce01b9d823e5b54f149028dcfdeaa5626cf3646d8`.

Terms used below:

- **Practice set:** 10 public matches with frame-level possession labels (7 Bundesliga, 3 Metrica), their
  tracking degraded to look like ours. Every method choice was made on it. It is about .06 spellF1 easier
  than our match (the organizers' 2-second smoothing baseline scores .304 on it and .243 on the full match),
  so we trust its ranking of methods, not its levels.
- **Intervals** are 95% bootstrap intervals over whole matches for the difference between two methods
  scored on the same data. "0/10" means no practice match went negative.
- **Level with zero:** the degrader uses random seeds, and the spread between seeds exceeds what a
  match-level interval shows. A practice-set difference below .004 has no established sign, whatever its
  interval says.

## 1. Approach

Possession is the team of the last credited contact, held through passes, loose balls and missing-ball
stretches until the other team touches it. Our tracking cannot see contacts: the ball is smoothed over about
0.3 s and missing on 40.7% of graded frames, mostly in blocks of about 10 s. So the pipeline proposes
hand-overs from player proximity and puts most of its effort into the question the data answers well:
when is the ball dead?

Seven stages run in a fixed order; each rewrites only what the earlier ones left.

| stage | what it does | practice-set gain |
|---|---|---|
| Match structure | Published half boundaries and kickoffs. The 2,934 frames before the second-half kickoff are `D`; frames outside the halves are `X`. | (required) |
| Baseline+ | The owner is the team whose player is nearest the ball within 2.5 m, confirmed over 16 frames (0.64 s) and kept through ball-missing frames. Ball out of the pitch is `D`. Spells under 0.8 s are merged. | +.0222 [+.0058, +.0403] over the 2-second baseline; the gain does not survive a 0.5 s label shift |
| Missing-ball gate | A ball-missing run of 2 s or more whose players look stopped is `D` in full. A classifier on player motion alone gives the stoppage probability. | +.0728 [+.0649, +.0808], 0/10 |
| Grown stoppages | A visible ball at rest with slow players is grown backward and forward to the edges of its stoppage (at most 25 m of ball travel, 30 s each way). | +.0068 [+.0055, +.0083], 0/10 |
| In-gap decoder | Inside missing-ball runs the gate left live, a two-state model infers the team in possession from team shape (that team spreads wider). | level with zero: +.0017, seed-inclusive interval [−.0011, +.0041] |
| Sub-gap rule | In the same runs, any 2 s stretch with stoppage probability above .5 is `D`. | +.0045, 0/10 |
| Short-spell floor | Any possession under 1.5 s becomes `D`. | +.0113, 0/10 |

Each gain is measured on the stack the stage joined; gains are not additive. The final pipeline scores
.4247 on the practice set, +.0988 [+.0909, +.1068] over Baseline+ and +.0208 [+.0169, +.0250] over the gate
plus in-gap decoder, both 0/10.

**Why this design.** The missing ball is a dead-ball sensor. On the practice set about 85% of missing time
is a stoppage; an independent check against the broadcast camera's ball track puts our match at 84.5%. The
gate supplies about three quarters of the gain over Baseline+. Three alternative cores each lost:

- *Possessions from detected touches* lost .025 to .071, worse in 10 of 10 matches. In our data only 31.7%
  of real hand-overs have a touch by the new team within 0.5 s, and only 2.2% of touches by the opposing
  team are real hand-overs.
- *A hidden semi-Markov model* (duration-aware decoder) scored −.0090 [−.0163, −.0024] against the gate plus
  in-gap decoder, worse in 9 of 10. It produces too few possessions.
- *A supervised frame classifier* scored +.0084 [+.0056, +.0115] over the gate plus in-gap decoder, but was
  negative in all three conditions at the default player-stillness setting and would change 0.35% of our
  frames. We kept the simpler stack.

**Held-out checks.** No choice was made on these. On 20 SkillCorner matches (possession only) the final
pipeline beats Baseline+ by +.055 to +.057 across four ways of deriving dead ball, 0/20. On 28 PFF World
Cup 2022 matches it beats the gate plus in-gap decoder by +.0189 [+.0169, +.0212], 0/28.

**Public 5 minutes.** Scored once per pipeline version, for reporting only: spellF1 .4686, macroF1 .7199,
18 predicted against 17 true possessions. Nothing was chosen on it. With 17 possessions it is a sanity
check, not an estimate; we have no defensible point estimate for the full match.

## 2. Assets used and dropped

| asset | what we did | result | status |
|---|---|---|---|
| Tracking (players and ball, 25 fps) | The whole pipeline | Ball missing on 40.7% of graded frames; with the ball present, the nearest team matches the owner 78% of the time | used |
| Video audio | Referee-whistle detector | Leads stoppages by about 2 s but changed no decision when added as evidence | prints a status line; cannot change the file |
| Video image | CPU ball detection to fill gaps | Best detector found the ball on 36% of frames where tracking had it and fired on 82% where it did not. Long gaps are 98–100% stoppage-like, so even a perfect detector adds little | dropped |
| Broadcast ball track | Independent check of what the gate calls dead | Adds 3.7 percentage points of ball coverage; 84.5% of missing time is dead by this check | audit only, not in the build |
| Homography | Measured as a ball-position prior | Camera aim follows the ball at r = .83 with 12 m median error; the gate does not need ball position | not used |
| Smart Tagging | Tested as a possession and stoppage prior | Adds nothing over team shape: −.001 [−.023, +.023] | dropped |

## 3. Training data

Two models are trained at build time, only on the 10 practice matches. None of the data comes from the
provided game, and we labeled nothing by hand.

| dataset | games | license | labels: source and labeler | use |
|---|---|---|---|---|
| IDSSE / DFL Sportec (Bassek et al., 2025) | 7 Bundesliga | CC BY 4.0 | The provider's official per-frame ball-possession and ball-status flags | trains both models; practice set |
| Metrica Sports sample | 3 | none stated; attribution requested | Owner from the last on-ball event in Metrica's event data; dead ball from event types | trains both models; practice set; possession-length prior |
| SkillCorner open data | 20 A-League | MIT | SkillCorner's per-frame possession field; dead ball derived by our rule | held-out check, possession only |
| PFF FC World Cup 2022 | 28 of 64 | access form, no published license | Owner and dead ball derived by our rule from PFF's event and tracking data | held-out check only |
| StatsBomb open data | 30 La Liga | StatsBomb user agreement | Event-level possessions | checked the shape of the possession-length distribution |

The models are the stoppage classifier (gradient-boosted trees on player motion at 5 Hz, no ball features)
and the team-shape classifier used inside missing-ball gaps (87% accurate on matches it was not trained on).
The in-gap decoder's switching cost, 3.88 nats, comes from a Gamma fit to Metrica possession lengths and was
not tuned. Seeds are fixed.

**Making the practice set look like our match.** A degrader smooths each practice match's ball, drops it
where ours drops, and matches our gap lengths and player counts. Its first version hid the ball during live
play far more than our tracker does (42% of live frames against about 3%), which inflated the in-gap
decoder's gain from about +.005 to +.08. After the fix the practice set matches our match on gap profile
and ball-to-player geometry. Remaining blind spots: no cooling break or video-review stop, Metrica
over-calls dead ball around advantage play, and resting-ball detection is 2–5 times worse than on clean
data.

## 4. Possession rules

**Owner.** The team of the last credited contact. Ownership persists through passes in the air, loose balls,
deflections the team recovers, and stretches where the ball is not tracked. It does not change because the
other team is nearer, because the shape of play changed, or because time passed. On the practice matches
the last-contact team equals the true owner on 90.1% of live frames; the nearest team does so on 76.5%.

**Loose balls.** A loose ball stays with the last team to have it. With the ball visible, a hand-over
needs 16 consecutive frames in which an opposing player is nearest within 2.5 m. Missing-ball frames
neither advance nor cancel that count. Inside missing-ball runs the gate left live, the in-gap decoder may
change owner from team shape.

**Challenges.** One opposing contact is enough: no period of control, second touch or duel winner is
required. Touches inside a duel become real hand-overs six times as often as touches outside one (.298
against .048), so suppressing duels would delete steals. A deflection the old team answers at once is the
same play: if the old team is nearest again before the challenger has 16 frames, nothing changes. A
confirmed change is back-dated to the first of those frames, so it adds no delay. The cost is that a single
brief opposing contact is invisible.

**Restarts and dead ball.** A dead ball is a referee stoppage, from the moment play stops to the restart,
whether or not the ball left the pitch. `D` is written by the out-of-pitch rule, the missing-ball gate,
grown stoppages and the sub-gap rule, in that order. The 2,934 pre-kickoff frames of the second half are
`D` from the published kickoff frame. Every other stoppage, including kickoffs after goals, is found by
code. We do not pick the restart team from the stoppage type: that moved restart-team accuracy by −0.4
points and lowered the score. After a stoppage the owner is the first team to accumulate 16 frames of
evidence, so an opponent restart is credited about 0.64 s late.

**Short possessions.** Any possession under 1.5 s becomes `D`, never the neighboring team. This is a
scoring rule, not a football claim. On the practice set it removes about 380 false possessions and 84 real
ones (of 464 under 1.5 s). We found no label-free way to keep the real 84: the best separator reaches 20.5%
precision against an 18.1% base rate, and a rule built on it scored −.0031 [−.0044, −.0017].

**`X`.** Only outside the graded halves. The scorer fills `X` with the previous label, so `X` at a
hand-over acts as lag.

## 5. Limitations

- **Recall.** On the practice set, measured before the last three stages were added, we match about one
  real possession in five. Most misses are hand-overs we never see, and 44% of those have no tracked ball.
- **Hand-over timing.** 46.9% [43.9, 49.5] of matched hand-overs land more than 1 s from the true one.
  93.4% of that tail comes from the proximity cue; fixing it needs a contact detector at about 0.16 s
  resolution, which smoothed tracking cannot support.
- **Restart team.** The owner 2 s after a stoppage is right 77.1% of the time, because many predicted
  dead stretches bridge several stoppages.
- **Stoppage edges.** 13.4% of dead stretches end more than 1 s late and 28.3% more than 1 s early.
- **Stoppages with a visible ball** are found only in part; about two thirds of dead frames have a tracked
  ball.
- **Transfer.** All evidence comes from degraded external matches. Our match has a cooling break and the
  practice set does not, so our dead share (45.7% of graded frames) is above any practice match.

Tried after the pipeline was frozen, none shipped:

- Holding a hand-over while the ball closes on the receiver cut that error class by 84% but scored −.0016
  [−.0035, +.0002], level with zero.
- Splitting long dead stretches where the ball reappears scored +.0014 [+.0008, +.0022], level with zero,
  and would rarely fire on our match.
- Down-weighting the in-gap decoder's evidence changes no frame of our file.

With more time, in order of expected value: contact-level hand-over timing; refining both edges of each
dead stretch, onset first (a perfect version is worth +.056 on the practice set); a rule that keeps real
short possessions; and a second practice set to settle the supervised classifier.

## 6. Reproduction

Requirements: Python 3.13.5, the organizers' data bundle, and the 10 practice matches on disk (the build
retrains both models). The external data is not in the repository; `docs/REFERENCE.md` lists the sources.

From the repository root:

```sh
python3.13 -m venv .venv
.venv/bin/pip install --no-cache-dir -r requirements.txt   # 34 hash-pinned packages
.venv/bin/pip install -e . --no-deps
export MTAH_DATA_DIR=/path/to/data              # default ../more-than-a-hack-2027/data
export MTAH_EXTERNAL_DIR=/path/to/external      # default ../external
make submission                                 # writes outputs/submission.csv
shasum -a 256 outputs/submission.csv            # ee7b5b6b...46d8
make validate                                   # organizer validator: Valid
make freeze-check          # two builds compared byte for byte, sha, validator, rules guard
make test
```

The build has no manual steps and no network access. An independent reviewer rebuilt the identical file three
times from a fresh clone, fresh environment and cold caches. A rules guard in the test suite fails on frame
literals tied to this match, reads of the public labels, network imports, training on anything but the
practice matches, and committed data files.
