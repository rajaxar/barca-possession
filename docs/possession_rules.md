# Our possession rules

These are the rules we actually implement, written the way they appear in our technical document. Each
rule says what we do, where in the code it lives, and the measurement that made us choose it. The
measurements come from other professional matches with public answers, degraded to look like ours,
because the organizers' answers exist for five minutes of our match and those may not be used to
choose anything.

In one sentence: **the owner is the team of the last player credited with a deliberate contact; a
single opponent contact is enough, but a contact the old team answers within about half a second is
the same play.** Our tracking cannot see contacts reliably, so in the code this rule is used to place
and confirm a hand-over that the tracking already suggests. It is never used to invent possessions.

## 1. What counts as a possession

A possession is an unbroken run of one team with the ball alive. The organizers define no minimum
length, and dead-ball frames end a run even when the same team restarts. There is no "contested",
"transition" or "unknown" state: every live frame belongs to one of the two teams.

*In the code:* a possession is any maximal run of `H` or `A` in the final label array. No module
constructs possessions directly; they fall out of the label sequence written by
`possession/make_submission.py::build_labels`.

*Why:* real possessions can be very short — on the practice matches 12.5% of possessions last under a
second and the median is 3.7 s (S05), with the left tail R08 measures directly. Anything that needs a
possession to last several seconds deletes real ones.

## 2. The owner is the team that touched it last, not the team that kept it

Ownership stays with a team through passes in the air, loose balls, deflections it recovers and
stretches where the ball is not tracked. It changes on a contact by the other team. It does not change
merely because the other team is nearer the ball, because the shape of play changed, or because time
passed.

*In the code:* `possession/baseline_plus.py::_labels_half` holds the current owner and only accepts a
new one after 16 frames of evidence that the other team's player is the nearest to the ball within
2.5 m (`_evidence`, `r=2.5`, `k=16`). Ball-missing frames neither advance nor cancel that candidacy.

*Why:* on external matches the owner equals the team of the last detected contact on 90.1% of live
frames, against 76.5% for the nearest team (91.8% only when the ball is within a meter of a player).
80.5% of owner changes have a contact by the new team within half a second, at a median offset of
0.00 s. Proximity is a proposal generator; contact is the definition.

## 3. A single opposing contact is enough; one answered within about 0.5 s is the same play

We do not require the challenger to keep the ball, to touch it twice, or to win a duel. A deflection
that the old team immediately plays again is not a change of owner.

*In the code:* the same 16-frame confirmation in `baseline_plus.py::_labels_half` is our deflection
merge. If the old team is nearest again before the challenger accumulates 16 evidence frames, the
candidacy is discarded. A confirmed change is back-dated to the first frame of the run, so confirming
it adds no delay. The explicit half-second merge over a list of contacts
(`possession/spells_from_events.py::revert`, `merge_s<=0.5`) exists but is not imported by the build.

*Why:* single-contact possessions are a real population — 767 of 3,061 practice-match possessions
(25%) contain exactly one own-team touch, median 2.2 s, and 44% of those are 2 seconds or shorter.
Possessions started by an opposing touch include 12% of 1 second or less. The merge window is
deliberately shorter than the shortest possession R08's label-side bound says we must keep (1.0 s), so
it cannot erase a genuine single-touch possession.

## 4. We do not rebuild possessions from touches

Our ball data delivers proximity events, not contacts: local minima of the distance from the ball to
the nearest player. They can re-time a hand-over the tracking model already found. They cannot decide
which possessions exist.

*In the code:* no shipped module imports `possession/touches.py` or
`possession/spells_from_events.py`. `build_labels` takes its label sequence from Baseline+, the
dead-ball rules and the in-gap decoder. The one touch-based improvement we measured (move an existing
hand-over onto a nearby new-team touch) is tier 2 behind a default-off switch, not in the build.

*Why:* rebuilding from touches loses .025 to .071 of the score, negative in 10 of 10 practice matches,
because only 31.7% of real hand-overs have a new-team proximity event within half a second and only
2.2% of opposing proximity events are real hand-overs. The loss is the event set, not the rule: even
an oracle restart owner loses .13.

## 5. Dead ball, and what we do inside a ball-missing stretch

A dead ball is a referee stoppage, not a geometry fact. A frame is `D` from the moment play stops
(ball out, foul, offside, goal, injury, cooling break) until the restart is taken. The Law stops play for reasons that are not geometric — a foul, an offside, an injury — and the ball
is often still on the pitch; on the practice matches only about a third of dead-ball onsets coincide
with an out-of-bounds event (E25). We label `D`; we never label who owns a dead ball, because the
organizers do not ask.

*In the code:* `baseline_plus.py::_dead_mask` starts `D` on a debounced ball excursion beyond the
pitch (0.5 m margin, 6 frames) and ends it at the first sustained in-pitch ball movement (3 m/s for 10
frames). Everything else is added by the four rules below.

*Why:* dead stretches run from a few seconds to two minutes on the practice matches (E39), so their
edges are cheap and their existence is not: a missing dead stretch joins two possessions into one, and a spurious one
splits a possession in two and costs a false positive as well.

### 5a. A long ball-missing stretch is a stoppage

The ball is absent on 41% of graded frames, in blocks of about 10 seconds. A missing-ball run of at
least 2 seconds whose players look stopped is labeled `D` in full; live islands shorter than 4
seconds between two dead stretches are absorbed into `D`.

*In the code:* `possession/dgate.py::gate`, with the player-only classifier `build_model` (no ball
columns, 5 Hz), `TF=50` frames, `Q=0.5`, `ABSORB=100` frames. `make_submission.py` runs it first and
`check_footprint` fails the build if it adds less than 10% or more than 50% of graded frames as `D`.

*Why:* about 85% of missing time is a stoppage, and the rule adds +.073 [.065, .081] to the score on
the practice set, positive in 10 of 10 matches. It is the largest single gain in the pipeline and it
is not a touch gain.

### 5b. A visible ball at rest inside a stoppage is grown out to the whole stoppage

A tracked ball moving at 1 m/s or less for at least a second, with the players slow and no plausible
goalkeeper within 2 m, is a dead-ball anchor. The anchor is grown backward and forward until the
stoppage ends: the ball leaves 25 m of its rest point, the team speed jumps, or the team holds a
jogging speed for a second while the ball also moves. Each side grows at most 30 seconds.

*In the code:* `possession/growd.py::anchor` and `::grow` (`R_M=25`, `CAP_S=30`), applied by
`::apply` immediately after the gate. `check_footprint` fails the build if it changes more than 1.10x its
own eligible frames (measured 1.016 on our match), adds more than 5 possessions under a second, or
touches a frame inside a gate-fired gap; a 20% absolute cap on added `D` remains only as a runaway guard.

*Why:* the anchor alone covers only about 3% of the stoppage it sits in, and using it alone lost in
all seven variants tested. Grown frames are about 95% true stoppage, and the gain rises with the
radius: +.0032 at 4 m, +.0051 at 12 m, +.0055 at 25 m. We expect +.001 to +.005 on our match.

### 5c. Inside a gap, the owner is decoded from team shape

Inside a ball-missing run of at least 2 seconds, the frames the gate left live are re-decoded with a
two-state model over "how likely is the home team to have the ball", given the shape of both teams.
The state before the gap and the state after it are pinned, and a change of owner costs a fixed price
equal to the log of the average possession length.

*In the code:* `possession/ingap.py::gapdec`, run only after the gate
(`ingap.apply`), with `LAM=ln(0.743*13.06*5)=3.88` nats from the external possession-length prior,
`MIN_GAP=50` frames. `check_footprint` fails the build if more than 3% of graded frames change, if any
change lands in a gate-fired gap, if more than 5% of changes are on ball-present frames, or if the
possession count grows by more than 36%.

*Why:* the ball-free shape model names the team with the ball 87% of the time on unseen matches, and
the decoder added +.005 [.004, .007] on the practice set when it landed, positive in 10 of 10 matches.
**On the final stack its measured contribution is smaller: removing it changes the score by −.001
[−.003, +.001], negative in 8 of 10 matches.** The short-hidden-stoppage rule now claims most of the
stoppages this decoder used to find, and what remains re-decodes live play, adding about 730 very short
spells that break more real possessions than they recover. It stays because the effect is inside the
noise band and the ball-free evidence is the thing that makes the hidden-ball stretches legible at all —
not because it still earns five thousandths. On our match only 3.9% of graded frames are eligible, so
its reach was always small; with the gate in front it rewrites 1.2% of graded frames.

### 5d. Inside a gap the gate left alone, a sustained dead probability is a stoppage

Within a ball-missing run of at least 2 seconds that the gate did not fire on, any stretch of at least
2 seconds with a dead-ball probability above 0.5 becomes `D`; live islands shorter than 4 seconds
between dead stretches are absorbed into `D`.

*In the code:* `possession/subgap.py::overlay`, run after 5c, with `Q=0.5`, `TF=50`, `T2=50`,
`ABSORB=100`. It is an overlay on the shipped labels, never a re-decode.
`check_footprint` fails the build if it changes any frame in a gap the gate fired on, or adds `D`
outside [0.5%, 15%] of graded frames.

*Why:* the missed stoppages sit inside the stretches the whole-gap rule left alone. On the corrected
practice data the rule adds +.0044, positive in 10 of 10 matches; we expect about +.002 on our match.
Re-decoding inside those gaps instead of overlaying was negative in 24 of 24 variants.

## 6. Possessions shorter than 1.5 seconds are pruned to `D`

This is a scoring rule, not a claim about football. A spell under 1.5 seconds is much more often a false
positive than a real possession, but not always: on the practice set about 18% of them do match a real
possession, so the rule is a trade, not free pruning. It removes roughly 380 false positives at the cost
of roughly 84 real short possessions, without joining the two possessions around it. We turn it into `D`,
never into the neighboring team.

*In the code:* `possession/postprocess.py::absorb_short(min_frames=38, absorb="to_d")`, applied by
`postprocess.apply` as the last step of `build_labels` — one line, after the sub-gap rule, so it sees
the final labels. It works inside each half and never across the half-time hole. Baseline+ has a
weaker version of the same rule built in (`baseline_plus.py::_merge_short`, 20 frames = 0.8 s,
absorbed into the longest live neighbor or into `D`).

*Why:* on the practice set 464 predicted possessions per 10 matches are under 1.5 seconds and the
scorer's matcher credits 84 of them at IoU >= .3 (43 at .5). Turning them into `D` is +.0099 [.0063, .0138]
on the pre-add-on stack and +.0113 on the shipped one, positive in 10 of 10 matches and at both half-second
answer shifts; absorbing them into a neighbor instead is worth .0059 less. It costs .0024 of frame-level
score, which we accept because possessions are ranked and frames are not. (An earlier draft of this
section said the matcher credited only 2 of them; that was a diagnostic bug, corrected in E99.)

## 7. What we do not do

- **No control rule.** A challenger does not have to hold the ball, or touch it twice. Conditioning on
  "the new team touches next" does not select real hand-overs (.226) better than "the old team touches
  next" (.252), and a control rule halves recall (.279 against .616) for no precision gain.
- **No duel suppression.** We do not decline to change owner when both teams are within a meter of the
  ball. Touches inside a real duel become real hand-overs six times more often than touches outside
  one (.298 against .048), so suppressing them would delete the steals.
- **No rebuilding possessions from touches** (rule 4). Every rebuild loses .025 to .071, in 10 of 10
  practice matches.
- **No ball from video.** We do not run a detector on the video to fill ball gaps. The best CPU option
  found the ball on 36% of frames where tracking had it and fired on 82% of frames where it did not.
  Long gaps (15 seconds or more) are 30.4 percentage points of graded frames and 98-100% stoppage-like,
  so even a perfect detector buys at most 3 to 7 percentage points of ball coverage.
- **No restart team chosen by stoppage type.** A throw-in is not assumed to go to the other team, and a
  goal kick is not assumed to go to the goalkeeper's team. Choosing by stoppage type moved restart-team
  accuracy by −0.4 points and made the score worse in every variant; the rate at which the restart team
  differs from the pre-stoppage owner is 41% for every type.

## 8. Structure we are allowed to assume

The organizers publish the half boundaries, the frame alignment and both kick-off frames. We use those
and nothing else about when things happen in this match; every stoppage, restart and post-goal kick-off
is found by the rules above.

*In the code:* `possession/structure.py` — `HALF_BLOCKS = ((1, 11185, 90651), (2, 110808, 190197))`,
`KICKOFFS = {1: 11185, 2: 113742}`, and `structure().forced_dead_mask`, which marks the frames before
each half's kick-off as `D`. The second-half block 110808–113741 is 2,934 graded frames and is all `D`
(published in the organizers' FAQ and in their starter loader); the first half is graded from its
kick-off, so it has no leading dead block.

*Why:* those 2,934 frames are 1.8% of the graded match and their labels are published, so getting them
wrong would be a free loss. Everything outside the two half blocks is `X`, which is never scored.

## 9. When we write `X`

Only frames outside the graded halves. We never write `X` inside a live possession, and never across a
hand-over.

*In the code:* `baseline_plus.py::predict_arrays` initialises every frame to `X` and writes `H`/`A`/`D`
only where `half > 0`; no later module writes `X`. Two functions that would
(`postprocess.d_to_x`, `postprocess.x_low_conf`) are present as measured negative cases and are not
called.

*Why:* the scorer forward-fills `X` with the previous label inside a half, so `X` in the middle of a
possession is free but `X` at a hand-over is pure lag. Writing `X` for 12 frames after every true
hand-over costs .185 of the score (a perfect 1.000 falls to .815); at 25 frames it falls to .632.
Writing `X` at uncertain hand-overs was negative at the −0.5 s answer shift in 10 of 10 matches.
Writing `X` over dead ball is never better than writing `D`, and is worse on the frame-level tie-break.
