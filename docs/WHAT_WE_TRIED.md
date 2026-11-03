# What we tried

The data is one full match, Barcelona against Athletic Club: where every player and the ball are, 25 times a second, plus video. For every moment we give one answer: Barcelona has the ball, Athletic has the ball, or the ball is dead (a throw-in, a foul, a goal, an injury).

The score is spellF1. A possession is an unbroken stretch of one team having the ball. We are credited for each possession we find, give to the right team, and place with roughly the right start and end. Splitting one possession into fragments costs a wrong prediction and also damages the real possession it was cut from. Missing a possession costs less. Differences under about 0.04 on the full match are noise.

The organizers kept the answers. We have them for 5 public minutes only, and the rules forbid tuning anything on them. So we practice on other professional matches with public answers, after degrading their data to look like ours. The main practice set is 10 matches (7 German league, 3 Metrica). A second set of 20 SkillCorner matches is used only for who-has-the-ball questions. Our ball is missing 41% of the time and is smoothed. How we build and check the practice matches is in [HOW_WE_WORK.md](HOW_WE_WORK.md). Terms are in [GLOSSARY.md](GLOSSARY.md).

Legend: ✅ works or adopted. ❌ tried, does not work. ⚠️ partly, or with a caveat. 🔬 in progress. 📋 planned.

"Shipped" means the step is in the file we would submit. "Behind a switch" means the code exists and is checked but is off in that file. Numbers in square brackets are the range we are 95% sure the true gain lies in, from resampling whole matches.

## When is play stopped?

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Treat the ball vanishing from tracking, with players standing still, as a stoppage (the stoppage rule) | The ball is missing 41% of the time, in long blocks | About 85% of missing time is a stoppage, found by two independent routes (a trained classifier and a check against the broadcast camera). Shipped. On the corrected practice data it adds +.073 [.065, .081] over Baseline+, positive in 10 of 10 matches. We expect +.06 to +.08 on our match. On our match it moves the dead share from 11% to 41% |
| ⚠️ | Look at the organizers' 5 public minutes to see if a missing ball means a stoppage (report only) | A first look at our own data | Too little data: 56% with a range from 0 to 96%. Two long gaps in it were live play. The larger measurements above showed those gaps are short and easy to tell apart. The first reading was too pessimistic |
| ✅ | Check which long missing-ball stretches are live play (tracker dropouts) | If many are live, the rule above cuts real possessions | Only 0.9% of missing time looks stopped but is live |
| ⚠️ | Redo that check, excluding throw-ins (the ball travels but play is dead) | Video review showed successive throw-ins fooled the "ball moving means live" test | Provably live missing time fell from 4.1% to 3.0%. The earlier claim "no live gap longer than 11 seconds" had no evidence behind it, because no long gap had a usable second view. Withdrawn |
| ⚠️ | Test the stoppage rule across how often the ball stays visible during stoppages | We do not know that share for our match | The gain is never negative. It shrinks as more stoppages show the ball and is zero when all do. An early reading from this test, that Baseline+ was only level with the organizers' 2-second baseline in our range, was first withdrawn after the practice-data fixes below, then partly restored: see the retune row |
| ✅ | Narrow our match's visible-ball share without using the answers | It sets which practice settings count | Between 7% and 23% of stoppages keep the ball visible, about 4.5% of graded time at the center. Our tracker's ability to see a resting ball is the largest remaining uncertainty |
| ✅ | Detect the referee's whistle in the video's audio | Whistles mark fouls and offsides, which ball-out-of-play does not | 74 whistles found; 21 of 55 graded ones are discarded as fan whistles. Stillness 3 seconds after a whistle is 51% against 35% at random. It leads the tracking by about 2 seconds, so it helps find where a stoppage starts |
| ❌ | Replace the hand-built whistle detector with a public audio-event model (PANNs, run on a local CPU) | A trained model might hear what ours misses | It scores a synthetic whistle tone at .115 but our match's real whistles at about .001. It cannot hear them. Ours stays |
| ❌ | Use crowd loudness to find stoppages | Cheap extra signal | It tracks microphone placement, not the match. Weak, and it disappears once we account for the missing ball |
| ✅ | Treat a goalkeeper holding or handling the ball as live play | A keeper picking the ball up looks like a stoppage | 14 such episodes, 0.55% of graded time, and our stoppage rules score them as live. Receive-and-clear moments confirmed live on video |
| ⚠️ | Find stoppages where the ball is still visible (resting for a free kick, held for a throw-in) | About two thirds of dead-ball time has a visible ball | The resting-ball moment is found about 93% reliably and covers 2.8% of graded time. But no decision changed |
| ❌ | Use only that resting-ball detector as a stoppage marker | It is precise (95 to 98%) | It covers only about 3% of the stoppage it sits in. The fragments cut real possessions apart and the score fell in all seven variants (best −0.0003) |
| ✅ | Grow each resting-ball moment out into the whole stoppage | The detector's problem was size, not accuracy | Adopted. Grown frames are about 95% true stoppage. The wider the radius the better: on corrected practice data +.0032 at 4 m, +.0051 at 12 m with a 30-second cap, +.0055 at 25 m (all ranges above zero). We expect +.001 to +.005 on our match. Only safe on top of the stoppage rule: on raw Baseline+, 63.7% of the changed frames land in dead gaps. An independent re-implementation matched, and corrected one figure (55% became 63.7%) |
| ✅ | Add the sub-gap rule: inside missing-ball stretches the main rule left alone, call a stoppage when the stoppage probability stays above 50% for 2 seconds | Missed stoppages sit in those stretches, not in very short gaps as we first guessed (gaps under 2 seconds are 0.05% of our match) | Adopted. +.0044 on corrected practice data (was +.0160 before the fix), 0 of 10 matches negative. We expect about +.002 on our match. With the visible-ball dead time removed, the rest of our match's ball-in-play time lands in the normal football range |
| ❌ | Split long stoppage blocks into their separate stoppages | The fan's review thought blocks join several stoppages | On the old practice data +.0121. But those blocks had 13% to 18% live play only because the practice data hid the ball during live play. On our match the rule fires on 0.09% of graded time. After the fix it scores −.0011, negative in 10 of 10 matches. Dropped. Our earlier claim that a block holds "several stoppages" was wrong: it holds 1.0 to 1.8 |
| ❌ | Choose who restarts by stoppage type (a ball out over the touchline goes to the other team) | After a stoppage we had the right team only 70% of the time | Falsified as pre-registered. Restart-team accuracy 2 seconds after play resumes is 77.1% now (70% was raw Baseline+). Adding the stoppage type changed it by −0.4 points and made the score worse in every variant. The flip rate is 41% for every type. The cause: 73% of stoppage blocks before a restart are over 16 seconds and bridge several stoppages and live play |
| ❌ | Find stoppages where the ball is missing but players keep moving, using the whistle | About 6% of graded time | The whistle precedes only 5.8% of these stretches against 4.7% by chance. A perfect detector would add only +.005. Parked |
| ✅ | Catalog the cases that look live but are dead, and the reverse | To know what each stoppage rule costs | The ball must be nearly still: above 1.0 m/s of team speed the rule misfires. A cooling break in the first half was found from the data alone, so our share of dead ball is higher than in any practice match |
| ✅ | Ship the two small add-ons (grown stoppages, sub-gap rule) behind footprint checks | Both gains are positive but each is a few thousandths | Shipped and independently reproduced. On the corrected practice data grown stoppages add +.005 and the sub-gap rule +.004; together +.011 [.010, .013], 0 of 10 matches negative. They change 2.9% and 0.7% of our match's graded frames, none of it inside a dead gap. The checks still fail the build if an add-on touches frames outside the stoppage rule's footprint |
| 🔬 | Check whether our match's missing-ball stretches begin and end exactly at stoppage edges or are spread both ways | The practice data spreads them symmetrically by a fitted guess. The stoppage rule's gain depends on it: +.107 with no spread, +.036 with the full fitted spread on an earlier version of the data | Running |
| 📋 | Measure how often the stoppage rules fire on frames that are certainly live (ball above 6 m/s) | A rough probe: 3.1% on our match against 1.0 to 2.0% on practice matches | Not built |
| 📋 | Measure how far our stoppage blocks end from the real restart, and what the lag costs | Block ends decide who looks to restart | Not built |
| 📋 | Measure how well our tracker sees a ball at rest, against the practice matches | The largest uncertainty in the visible-ball share | Not built |
| 📋 | Grow stoppages past 25 m, and find where "the same stoppage" stops being true | The gain rose all the way to the edge of the tested radii | Not built |
| 📋 | Separate a held ball that is dead from one that is live, using player contact and formation | Held balls are hard for the resting-ball detector | Not built |
| 📋 | Anomaly detection for resting-ball moments; changepoint detection for stoppage onsets; using the restart whistle to mark the end of a stoppage | Alternative ways to find the whole stoppage | Not built |
| 📋 | Fuse every stoppage clue in one detector; a restart detector; after-goal restart chains from whistle and kickoff shape; re-pick the stoppage threshold when stillness is softened | The current rule uses few clues | Not built |
| 📋 | Choose the dead-ball share we target from outside evidence, with our match's share only as a diagnostic | Our dead share is 41% and ball-in-play time still looks high | Not decided |

## Who has the ball when we can't see it?

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Guess the team with the ball from team shape alone, no ball | The ball is missing 41% of the time | The team with the ball is wider. Correct 87% of the time on unseen matches. With the ball hidden for 5 to 10 seconds this beats "keep the last owner" (88% against 70%), though keeping the owner is better for the first 2 seconds |
| ✅ | Look at what our missed possessions are | We find only about 1 in 5 real possessions | Most misses are changes of owner we never saw, often with no ball in view. About 44% of those changes have no tracked ball at all, where any ball-based rule is blind |
| ⚠️ | Decode owner changes inside missing-ball gaps with the shape model (the in-gap decoder) | The clue fits the largest group of misses | Shipped, on top of the stoppage rule only. On the first practice data it added +.074 to +.102, reproduced independently. That data hid the ball in live play, so the gain was inflated. On corrected data it adds +.005 [.004, .007], positive in 10 of 10 matches. We expect +.002 to +.005 on our match. Run alone it rewrites 12.6% of our match, mostly inside stoppages. Behind the stoppage rule it touches 1.2% |
| ❌ | Run the same decoder when the ball is visible but nobody is within 3 meters of it (long kicks, loose balls) | Our nearest-player rule is blind there | Frame accuracy there rose from 77% to 84%, but the score fell: −.0031 and −.0111 at two visible-ball settings. More changes of owner broke more possessions than they fixed |
| ⚠️ | Use the organizers' tactical tags (Smart Tagging) as a possession prior | A second view of who attacks | Parked. The attacker code is reliable 60% to 90% of the time but adds nothing to team shape (−.001 [−.023, +.023]). It gives independent support for the stoppage rule: on ball-missing live frames the tags agree with Baseline+ 45% of the time (chance) and with the gated entry 64%. It leans against the in-gap decoder on our match (31 gaps: 3 better, 6 worse, range includes zero) |
| ⚠️ | Decode for the score itself: pick the segmentation with the best expected spellF1 | Per-frame accuracy and the score disagree | It does not replace the shipped stack (−.0074 to −.0159, because it produces too few possessions). It is the right objective, so it is kept as a switchable refinement. Under it the "nobody near the ball" clue turns positive (+.0057 [+.001, +.010]) |
| ❌ | Guess who wins a long ball from where it lands | Idea from the ideation round | Only 8% of gaps start with a fast ball. No signal |
| ❌ | Count how many players hide the ball at the start of a gap, as a sign of a contest | Idea from the ideation round | Same count as picking a random moment |
| ❌ | Trace contacts between players to spot ball transfers | Idea from the ideation round | Adds nothing beyond "who is nearest the ball" |
| 📋 | Settle how readily the in-gap decoder switches team, with a pre-registered sweep | The setting was left at its default | Not built |
| 📋 | Find a check on our match that can see who has the ball inside missing-ball gaps | No label-free instrument sees inside them | Not built |
| 📋 | Check the in-gap decoder's gain bound at our match's small eligible share | Only 3.9% of our graded frames are eligible, against 17% to 44% in practice | Not built |
| 📋 | Re-run the team-shape model on noisy practice matches and check team-width on our match | It was built on cleaner data | Not built |
| 📋 | Use expected-score decoding to refine the shipped segmentation instead of replacing it | The first test replaced it and lost | Not built |

## Spotting touches of the ball

| | What we tested | Why | What happened |
|---|---|---|---|
| ⚠️ | Detect kicks from sudden ball acceleration | The textbook method, and it works on clean public data | Recall 75 to 83% on clean data. It fails on our ball |
| ✅ | Measure the noise in our ball | To find out why that failed | Our ball path is smoothed over about 0.3 seconds, so a kick is spread out. Once practice matches get the same smoothing the method fails there too, which confirms the cause |
| ❌ | Track the ball as it travels with a player or flies between players (carry and pass evidence) | A different route to touches | Best case finds half of owner changes. Adding it to the decoder cost −0.002 to −0.011. The nearest-player rule already carries the same information |
| ❌ | Hear kicks in the audio | Gives exact kick times | Not above chance |
| ✅ | Count how many ball-visible changes of owner can be seen at all | It sets the ceiling for any touch method | On clean data the nearest-player rule catches 76% of changes. On our degraded copy it catches 48%, so 28 points are lost to the degrading, which our match shares. Of the misses on clean data, 32% are a visible touch, 37% are duels, 20% are tracking that contradicts the label and 9% show no sign. About 13.5% of all changes cannot be resolved from geometry. We stopped adding ball-visible evidence |
| ✅ | Measure how early or late our hand-overs land | Timing might cost score | They land slightly early: the middle hand-over is 0.32 seconds early on one source and 0.12 seconds early on the other. The scatter is the real cost: 41% to 45% land more than a second off. Scatter costs about .10 to .15 of a perfect score, the early bias about .02 to .035 |
| ⚠️ | Delay every hand-over by 3 frames | A simple fix for the early bias | +.0039 [+.0025, +.0051], but only one source drives it and 2 of 3 Metrica matches go negative. Kept behind a switch, sign on our match unknown |
| ✅ | Check that the two sources mark a possession change at the same moment | The delay result might be a clock difference | They do: both mark the new team's first touch, within one frame. The difference between sources is how our decoder behaves, not the labels |
| 📋 | Find out why about 43% of our hand-overs land more than a second from the real one | The scatter is the biggest timing cost | Not built |
| 📋 | Undo the smoothing mathematically; flag moments when the whole team turns together | Two more ways to date a change of owner | Low expectation. Not built |
| ✅ | Check whether the decoder's lag is real or a property of one source's clock; score with timing shifts built in | Guard against tuning a lag on one source | Both sources mark a possession change on the new team's first touch, within one frame of each other, so our early hand-overs are real behavior of our method. The scoring harness now runs a half-second shift check on every comparison |

## Making practice matches look like ours

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Find open matches with tracking, ball and possession answers; load them into one format | We cannot tune on our match | 7 matches from the German league (DFL), 3 from Metrica, 20 from SkillCorner. Licenses checked. A fourth source, the 2022 World Cup data, is cleared to use but not loaded |
| ✅ | Rebuild their answers to match the organizers' definitions | Each source defines "owner" and "dead" a little differently | Last-touch owner and a ball-out-of-play dead state match ours. SkillCorner's dead state is our own derived rule: about 45% of its stoppage runs never see the ball leave play, and most runs are cut off by tracking gaps. So SkillCorner is H/A only and never sets a dead-ball rule |
| ✅ | Degrade clean data to look like ours: drop the ball where ours drops, smooth it, match gaps and player counts | Our ball is missing and blurred | Matches the real data on gap lengths, player counts and smoothness. Hidden leaks removed: in the raw data the ball freezes in dead-ball frames and goes missing exactly then |
| ✅ | Add a setting for how often the ball stays visible during stoppages | We can't know the true share | Methods are tested at 5%, 12% and 23% as the main cells, plus 30% and 50% as stress. The method ranking holds across settings. The size of the stoppage rule's benefit does not |
| ✅ | Place the resting ball near the player who would restart | A resting ball in the degraded data sat far from any player, which gave it away | A classifier could tell resting balls apart at 0.80 (0.5 means no tell). That fell to 0.63. A bug that teleported the resting ball between players 100 to 560 times a match was fixed (now about 15) |
| ✅ | Check whether player stillness is exaggerated by the degrading | It is the signal the stoppage rule uses | It is not. Left alone, with a softer setting that we always report beside the default |
| ⚠️ | Practice-data flaw 1: the resting ball shook about 20 times more than a real one | Ball clues separated dead from live far better on practice data than on real data | Fixed in sign. The shake score separating dead from live fell from .984 to .49, where real data sits near .42. A weaker tell remains: a classifier on the ball's texture alone scores .92 against a target of .80. On 20 SkillCorner matches never used for tuning it scores .907. We still treat ball-motion clues as upper bounds. Ball speed alone was always sound |
| ✅ | Practice-data flaw 2: the degrader hid the ball at random, including during live play | Our match hides it almost only during stoppages | The share of live frames with no ball was .42 on practice data against about .03 on our match, and 35% to 39% of long gaps were live against 4% or less. Fixed with a cap on live-play gaps and a spread on gap edges: live-frame missing share now .044, long-gap live share .021, our match's gap profile matched on four measures. This moved every number: Baseline+ .216 to .326, the in-gap decoder +.082 to +.006, the stoppage rule's gain +.085. An independent check passed it on the 10 matches and on 20 held-out SkillCorner matches |
| ✅ | Check the practice data's ball-to-player geometry against our match | The 28 points lost in the ceiling check could have been a degrading artefact | It matches: nearest-player median 2.29 m against 2.35 m on ours, a player within 1 m .209 against .211, two-team contests .029 against .026. The clean data is the outlier (too sharp). Only the kinks are slightly over-blurred, by about 2 points |
| ✅ | Freeze the official practice set | Every later comparison needs one yardstick | 10 matches, 3 random seeds, the 6 main cells (3 visible-ball settings by 2 stillness settings), the stress cells, and a ±0.5 second answer-shift check. SkillCorner is a separate H/A-only set. Eight blind spots are recorded, among them: no cooling breaks or video-review stops, Metrica over-calls dead ball around advantage play, short sliver gaps our match lacks, and low resting-ball recall |
| ✅ | Compare the practice set's difficulty with the real match | To know what its levels mean | The organizers' 2-second baseline scores .304 on it against .243 on the real match, so it is about .06 too easy. We use rankings, never levels |
| ✅ | Test whether errors in answer timing change which method wins | Source labels have their own timing | Random jitter hardly matters (≤0.006). A systematic shift of a second does, so we never tune a lag on one source |
| ✅ | Clean obvious tracking errors | Duplicated players, ball spikes | Fixed. No change to the score of the entry |
| ✅ | Unit check across sources | The review found one source's speeds were in km/h | Fixed |
| ✅ | Make cache writes safe when several processes run at once | A failed test showed they could collide | Fixed. A four-process test now passes. The file did not change |
| 🔬 | Check whether SkillCorner's invented dead rule moves who-has-the-ball comparisons | SkillCorner only counts as a selector if it does not | Running |
| ✅ | Build one shared harness that scores and compares methods on the frozen practice set the same way | The method race needs identical scoring | Built. It also runs the half-second shift check, which is how the Baseline+ lead was found to be fragile |
| 📋 | Run the shipped stack and the finalists on the 20 SkillCorner matches | The second set confirms who-has-the-ball gains | Not done |
| 📋 | Test whether the method ranking flips if the practice data is made a bit cleaner or messier | Only one degrading level is fitted | Not done |
| 📋 | Remove the short sliver gaps the corrected data creates (about 10 a match, none on ours) | They inflate two small gains slightly | Not done |
| 📋 | Smaller degrading refinements: the far tail of ball-to-player distances, the leftover resting-ball texture, player-track noise, 10-frames-per-second sources, advantage-play labels in Metrica | Remaining differences from our data | Not done |
| 📋 | Audit feature distributions across our match, the degraded practice data and Metrica; tighten the degrader's leak tests; measure the stoppage proxy at each degrading level | Catch drift before it fools a decision | Not done |
| 📋 | Load the 2022 World Cup tracking with raw ball only | A fourth source | Not done |
| 📋 | Audit SkillCorner's derived dead state against phases of play | Remaining SkillCorner question | Not done |

## Tuning for the scoring rule

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | A simple first entry: nearest player within 2.5 m owns the ball, otherwise keep the last owner; ignore short flickers; ball out of play is dead (Baseline+) | Something valid to submit | 493 possessions, a valid file, identical on every rerun |
| ⚠️ | Retune that entry on degraded practice matches | The settings were placeholders | Longer holding of the current owner gave +.058 over the organizers' 2-second baseline on the first practice data. That gain shrank to about nothing with a more realistic visible-ball share, then came back as a smaller one: after both practice-data fixes Baseline+ is +.022 [.006, .040] ahead of the 2-second baseline. That lead does not survive the half-second answer-shift check (at −0.5 seconds it is +.0066, range −.010 to +.025). So Baseline+ is about level with, or slightly ahead of, the 2-second baseline. The orderings that matter survive every shift: stoppage rule above Baseline+, and the shipped entry above the stoppage rule |
| ✅ | Measure possession lengths on public matches | To set what a normal possession looks like | Median 4.6 seconds. About 12% are under 1 second, mostly a single touch. We aim for fewer short ones, but do not delete them |
| ✅ | Pick a method for splitting the match into whole possessions rather than frames | Cleaner possessions | A duration-aware decoder was chosen on paper. Not built |
| ❌ | A label-free score on our match to rank close methods | Insurance in case the practice set is weak | Failed its pre-registered bar: it picked the practice winner 47% of the time where we required 70%. It does separate Baseline+ from any gated stack (97%). Kept as a red-flag check only. The practice set stays the selector |
| ❌ | Optimize the score directly with reinforcement learning | Idea from the ideation round | Dropped as heavy on a CPU and risky. The expected-score decoding above covers it |
| 📋 | Joint search over hold time and merge settings | The first tune stopped at the edge of its grid | Not done |
| 📋 | Stop the merge rule from absorbing real long turnovers | Part of the missed possessions | Not done |
| 📋 | Statistical decoder with explicit possession lengths, combining all clues; choose its length family | Planned main method | Designed, not built |
| 📋 | A model trained on the practice matches | Everything it needs now exists | Not started |
| 📋 | Turn touch events into possessions; tune spell count and length for the metric; a spell verifier; ensemble the finalists | Cheaper alternatives and combiners | Not built |
| 📋 | Weight clues without labels; adapt clue weights per half; train on our own confident frames; a contest-gated length prior | Alternatives for combining clues | Not built |
| 📋 | Print the label-free red-flag checks in the build log | Make the check routine, never gate the file on it | Not built |
| 📋 | Race three approaches on the frozen practice set: touch-by-touch, a statistical decoder, a trained model | The core architecture choice | Harness ready |

## Better raw data

| | What we tested | Why | What happened |
|---|---|---|---|
| ❌ | Find the ball in the video ourselves | The tracker loses a visible ball | The ball is 8 to 10 pixels wide. The best detector found it on 36% of frames where tracking had it, and fired on 82% of frames where it did not. The best option also has a restrictive license. Dropped |
| ✅ | Check how much video detection would help even if it worked | | Long gaps are almost all stoppages we already infer. Best case adds 3 to 7 points of ball coverage at low trust |
| ⚠️ | Check whether the tracker loses balls that are visible on video | A human watching saw one | At least 4.5% of missing time is a lost visible ball |
| ❌ | Predict where the ball is during a live gap (particle filter) | Idea from the ideation round | Low value by the same coverage argument |
| ✅ | Use the broadcast camera's own ball track | A second view | Adds 3.7 points of ball coverage, found by code, and serves as a live-or-dead check |
| 📋 | Bridge short live gaps using video, with camera motion removed | Needs 80% reacquisition within 2 meters to continue | Not built |
| 📋 | Use broadcast cutaways and replays as a dead-ball cue; ball outside the pitch in broadcast view; camera pan as a sensor; the organizers' tactical tags as a prior | Other match-only clues | Not built |
| 📋 | Fuse the broadcast ball track into the entry for extra coverage and a replay cue | The second view is only a check today | Not built |
| 📋 | Train a ball detector on a GPU | Needs a GPU we do not rely on | Parked |

## Football-knowledge review

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Watch 14 short video clips, as understanding only, never as labels | Check the model against football | Broadly confirmed. Two corrections: a throw-in sequence is dead although the ball moves, and a keeper involvement is often receive-and-clear, which is live |
| ✅ | Walk through 40 match situations from the Laws (the fan's review) | Find what we overlooked | 25 covered, 9 too small to matter, 3 killed by tests, 3 open. The one real lead, who restarts after a stoppage, was then tested and falsified (see above) |
| 🔬 | A second round of video clips, on what is inside our long stoppage blocks | Tells us whether the blocks hide live play | Clips prepared. Answers not yet recorded |
| ❌ | Fill short dead-ball islands between two spells of the same team | Looked like noise | Score −0.001 to −0.004 |
| ❌ | Treat an airborne ball as live | A ball in the air cannot be resting | −0.018 |
| ❌ | End a stoppage when a ball retriever is running | Retrievers signal a restart | After the stoppage, play looks live already |
| ❌ | A second ball on the pitch as a stoppage clue | | No sudden ball jumps found |
| ❌ | Substitutions as a stoppage signature | | Extra peripheral players are tracked for a median of 3 seconds. Dropped |
| ✅ | Dismiss rare events: quick free kicks, penalties, red cards, a ball hitting the referee, pitch invaders | Each is under 0.5% of the score | Ignored on purpose |
| ✅ | Check keeper rule length | The Laws changed from 6 to 8 seconds | Noted |
| ⚠️ | Check that the number of ball-in-play minutes is plausible against typical matches | Sanity band of 53 to 55 minutes | Our entry has 62 minutes, still high. Removing about 4.4 minutes of visible-ball dead time, which the grown-stoppage add-on targets, would put it inside the band |

## Method and evaluation setup

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Compare two methods by resampling whole matches, not single possessions | Possession-level resampling called 30% of identical methods different, whole-match resampling 5% | Adopted as the only test. Detecting a 0.04 gain needs about 8 matches |
| ✅ | Keep levers that are positive but unproven behind a switch, off by default | Ten matches give little power | A two-tier policy, now explicit: a lever ships only if its range is above zero overall, it is non-negative in every main and stress cell, it survives a half-second answer shift, and SkillCorner does not contradict it |
| ✅ | Quote the expected gain on our match beside every practice number | Practice gains are near-deterministic, so a +.004 gain is real there and invisible on one match | Adopted. Each shipped lever lists its practice gain and its smaller expected gain |
| ✅ | Check every practice-data property against our match conditioned on the game state, not only overall | The ball-hiding flaw passed an overall check and failed a conditional one | Adopted |
| ✅ | Build the evaluation tools and reproduce the organizers' four baselines exactly | Needed before any claim | Matches to the digit |
| ✅ | Match structure: half boundaries, kickoffs, attack direction, the pre-kickoff block | Structure the organizers publish | Done, each constant cited to its source |

## Keeping ourselves honest

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Rules audit of the first entry | Rule breaches disqualify | Passed. One warning (a test reading the public answers) was fixed with a guard that fails on public-answer reads outside one script, frame-number constants, network imports and committed data |
| ✅ | Reproduce the shipped entry (Baseline+, stoppage rule, in-gap decoder) from a clean copy | One command must regenerate the file | An independent reviewer got the identical file from a clean clone, fresh install and cold caches, passed 5 of 5 checks, and the rules audit was clean |
| ⚠️ | Reproduce it across thread counts, a clean hash-locked install and a Linux container | The file must not depend on the machine | Identical every time, with dependency versions locked. The ARM architecture (Apple silicon) is untested |
| ⚠️ | Code review of features and loaders | Hidden leaks and bugs | No leaks. Found the km/h speed bug, fixed |
| ⚠️ | Independent check of the degrading | A leak would inflate every result | Several rounds. The smoothing and visible-ball setting each had a leak that was found and fixed (see the practice-data section). Ball-only clues still separate dead from live a little on practice data, so they are discounted |
| ✅ | Independent check of the resting-ball placement, resting-ball shake fix and live-play-gap fix | Each changes every later number | All three passed. The live-play-gap fix also held on 20 SkillCorner matches never used in tuning |
| ✅ | Independent check of the in-gap decoder | Its gain was the largest | Reproduced to the fourth decimal, no leak. It ships only after the stoppage rule |
| ✅ | Independent re-implementation of the grown stoppages and the visible-ball range | Both feed decisions | Passed, numbers reproduce |
| ⚠️ | Independent review of the whistle detector | Before anyone uses it | Detector sound, but its failure handling was not: partial audio decode passed as success, and events were one frame late. All nine faults fixed |
| ⚠️ | Independent review of SkillCorner | Catch leaks | One quoted figure used an unbalanced time window and is withdrawn. Its dead state differs from ours, hence H/A only |
| ✅ | Public-sample score, report only | Where we stand | See below. Never used to choose anything |
| ✅ | Setting one adoption rule for match-only clues (whistle, cutaways, tags) | Avoid choosing by eye on our match | Whistle adopted, crowd noise parked |
| 📋 | Re-run the rules audit and the from-scratch reproduction on the frozen final pipeline | The last step before submitting | Not done |

## Claims we withdrew or corrected

- "No live gap is longer than 11 seconds": no evidence behind it. Replaced by an upper bound of about 4% live in long gaps.
- "The first retune gives +.058 over the 2-second baseline": that held only at the extremes of the visible-ball share.
- "Baseline+ is only level with the 2-second baseline": withdrawn after the second practice-data fix, which showed +.022. That lead then failed the half-second shift check, so the fair statement is about level or slightly ahead.
- "The in-gap decoder adds +.08": it added +.005 once the practice data stopped hiding the ball in live play.
- "A stoppage block holds several stoppages": it holds 1.0 to 1.8.
- "55% of changed frames fall in dead gaps": 63.7%.
- The first choice of the visible-ball setting (50%) sat outside the range our match allows.
- One SkillCorner figure used an unbalanced window.
- The practice set is easier than the real match by about .06, which we did not know until the fixes.

## Where we stand

The shipped entry is Baseline+, then the stoppage rule for missing balls, then grown stoppages for visible-ball stoppages, then the in-gap decoder, then the sub-gap rule. It builds the same file on every rerun, across thread counts, a clean install and a Linux container. Independent reviewers reproduced it, including the two add-ons.

On the corrected practice data, averaged over the six main cells:

| method | score |
|---|---|
| Organizers' 2-second baseline | .304 |
| Baseline+ | .326 |
| Baseline+ and stoppage rule | .399 |
| Shipped entry before the short-spell floor (gate + in-gap + add-ons) | .404 |
| Shipped entry, frozen (adds the short-spell floor) | **.4247** |

The shipped entry is +.100 [.088, .113] over the organizers' 2-second baseline. The stoppage rule supplies +.073 of that, the in-gap decoder +.005, and the two add-ons about +.011 on top of the decoder alone. On our match we expect the stoppage rule to add +.06 to +.08 and the rest +.005 to +.012. The practice set is about .06 easier than the real match, so we trust the order of the methods, not the levels.

One add-on tested on that set was dropped: splitting long stoppage blocks, −.0011. The ordering we ship is not sensitive to how hard the practice data is degraded: making it 20% cleaner or 20% messier leaves the same order (our entry, then the stoppage rule, then Baseline+, then the organizers' 2-second baseline), with the entry ahead of the stoppage rule by +.020 to +.023 in both directions. The short-spell floor is in the shipped entry: on the final stack it is worth +.0113 on the practice set, positive in 10 of 10 matches. It is a trade, not free pruning — it deletes about 84 real short possessions to remove about 380 false ones. We tested whether the 84 could be kept and the 380 dropped: the best label-free separator only reaches 20.5% precision against an 18.1% base, and a rule built on it lost .0031, worse in 9 of 10 matches. Keeping exactly the right ones would be worth .0089, so the simple rule stays.

The biggest weakness is possessions: we catch about 1 in 5 well enough to count. Following the ball more closely did not help. About 13.5% of changes of owner cannot be seen from positions, and our hand-overs often land more than a second from the real one.

On the organizers' 5 public minutes the shipped entry scores **.4686**, against .326 for their 2-second baseline and .424 for their 5-second one. Our frame-level score there is **.7199** against .581 for their best. It is first on both. That sample holds only 17 real possessions and we report it without acting on it.

## What we plan to test next

In progress:

1. The SkillCorner check, which decides whether that second set can pick methods.
2. Shipping the two small add-ons.
3. The shape of our match's gap edges.
4. The second round of video clips.

Planned, grouped by what they test:

- **The race.** Race touch-by-touch, a statistical decoder and a trained model on the practice set. Then choose the core architecture, write down our possession rules, and fold in late improvements. Freeze the method, re-run the audit and reproduction, and write up what we learned.
- **Decoder refinements.** Joint search over hold time and merge settings; stop the merge absorbing long turnovers; settle the in-gap switching rate; refine the shipped segmentation with expected-score decoding; ensemble the finalists; the statistical decoder with explicit possession lengths.
- **Hand-over timing.** Find why 43% of hand-overs land more than a second off, and test one fix for the largest cause.
- **Stoppages.** Gate false alarms on certainly-live frames; how far block ends sit from the real restart; how well our tracker sees a resting ball; grow stoppages past 25 m; held ball dead versus live; one detector fusing every stoppage clue; a restart detector and after-goal restart chains; anomaly detection, changepoints and the restart whistle; the dead-ball share we target.
- **Who has the ball.** Re-run the shape model on noisy matches; a label-free instrument that sees inside gaps; the in-gap gain bound at our small eligible share; weight clues without labels; contest-gated length prior; spell verifier; a model trained on practice matches.
- **Practice set.** Run the finalists on the 20 SkillCorner matches; check the ranking against cleaner and messier data; remove sliver gaps; the far tail of ball distances and leftover resting-ball texture; player-track noise; 10-frames-per-second sources; advantage-play labels; feature-distribution audit; tighter leak tests; load the 2022 World Cup data; audit SkillCorner's dead state; label-free red-flag checks in the build log.
- **Raw data.** Fuse the broadcast ball track; broadcast cutaways, camera pan and ball-outside-pitch cues; bridge short gaps with video; the organizers' tags as a prior; a GPU ball detector (parked).
- **Team tooling.** Shared cache keys, a cue registry, a progress board view, local match and clip viewers, a dependency prune after the final method lands.
