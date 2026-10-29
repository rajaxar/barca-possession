# What we tried

The data is one full match, Barcelona against Athletic Club: where every player and the ball are, 25 times a second, plus video. For every moment we give one answer: Barcelona has the ball, Athletic has the ball, or the ball is dead (a throw-in, a foul, a goal, an injury).

The score is spellF1. A possession is an unbroken stretch of one team having the ball. We are credited for each possession we find, give to the right team, and place with roughly the right start and end. Splitting one possession into fragments costs a wrong prediction and also damages the real possession it was cut from. Missing a possession costs less. Differences under about 0.04 on the full match are noise.

The organisers kept the answers. We have them for 5 public minutes only, and the rules forbid tuning anything on them. So we practise on 30 other professional matches with public answers, after degrading their data to look like ours. Our ball is missing 41% of the time and is smoothed. How we build and check those practice matches is in [HOW_WE_WORK.md](HOW_WE_WORK.md). Terms are in [GLOSSARY.md](GLOSSARY.md).

Legend: ✅ works or adopted. ❌ tried, does not work. ⚠️ partly, or with a caveat. 🔬 in progress. 📋 planned.

"Not switched on" means the code exists and is checked but is not yet in the entry we would submit.

## When is play stopped?

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Treat the ball vanishing from tracking, with players standing still, as a stoppage | The ball is missing 41% of the time, in long blocks | About 85% of missing time is a stoppage, found by two independent routes (a trained classifier and a check against the broadcast camera). Practice score +0.038. Not switched on |
| ⚠️ | Look at the organisers' 5 public minutes to see if a missing ball means a stoppage (report only) | A first look at our own data | Too little data: 56% with a range from 0 to 96%. Two long gaps in it were live play. Later, the two larger measurements above showed those gaps are short and easy to tell apart. The first reading was too pessimistic |
| ✅ | Check which long missing-ball stretches are live play (tracker dropouts) | If many are live, the rule above cuts real possessions | Only 0.9% of missing time looks stopped but is live |
| ⚠️ | Redo that check, excluding throw-ins (the ball travels but play is dead) | Video review showed successive throw-ins fooled the "ball moving means live" test | Provably live missing time fell from 4.1% to 3.0%. The earlier claim "no live gap longer than 11 seconds" had no evidence behind it, because no long gap had a usable second view |
| ⚠️ | Test the stoppage rule across how often the ball stays visible during stoppages | We do not know that share for our match | The gain is always positive, biggest when stoppages hide the ball, zero when they never do. In the share range we estimate for our match, our shipped entry is only level with the organisers' simple baseline (−0.005, range −0.021 to +0.011). The stoppage rule still beats that baseline by 0.05 to 0.08 |
| ✅ | Detect the referee's whistle in the video's audio | Whistles mark fouls and offsides, which ball-out-of-play does not | 74 whistles found; 21 of 55 graded ones are discarded as fan whistles. Stillness 3 seconds after a whistle is 51% against 35% at random. It leads the tracking by about 2 seconds, so it helps find where a stoppage starts |
| ❌ | Use crowd loudness to find stoppages | Cheap extra signal | It tracks microphone placement, not the match. Weak, and it disappears once we account for the missing ball |
| ✅ | Treat a goalkeeper holding or handling the ball as live play | A keeper picking the ball up looks like a stoppage | 14 such episodes, 0.55% of graded time, and our stoppage rules score them as live. Receive-and-clear moments confirmed live on video |
| ⚠️ | Find stoppages where the ball is still visible (resting for a free kick, held for a throw-in) | About two thirds of dead-ball time has a visible ball | The resting-ball moment is found about 93% reliably and covers 2.8% of graded time. But no decision changed |
| ❌ | Use only that resting-ball detector as a stoppage marker | It is precise (95 to 98%) | It covers only about 3% of the stoppage it sits in. The fragments cut real possessions apart and the score fell in all seven variants (best −0.0003). Growing the marker to the full stoppage would gain +0.019 to +0.029 |
| ❌ | Find stoppages where the ball is missing but players keep moving, using the whistle | About 6% of graded time | The whistle precedes only 5.8% of these stretches against 4.7% by chance. A perfect detector would add only +0.005. Parked |
| ✅ | Catalogue the cases that look live but are dead, and the reverse | To know what each stoppage rule costs | The ball must be nearly still: above 1.0 m/s of team speed the rule misfires. A cooling break in the first half was found from the data alone, so our share of dead ball is higher than in any practice match |
| 🔬 | Grow a detected resting-ball moment into the whole stoppage | The largest unclaimed gain (+0.07 to +0.10 if all dead-ball frames were right) | Running |
| 📋 | Choose who restarts by stoppage type (a ball out over the touchline goes to the other team) | After a stoppage we have the right team only 70% of the time, 2 seconds after play resumes | Not built |
| 📋 | Measure how often the stoppage rules fire on frames that are certainly live (ball above 6 m/s) | A rough probe: 3.1% on our match against 1.0 to 2.0% on practice matches | Not built |
| 📋 | Anomaly detection for resting-ball moments; changepoint detection for stoppage onsets; using the restart whistle to mark the end of a stoppage | Alternative ways to find the whole stoppage | Not built |

## Who has the ball when we can't see it?

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Guess the team with the ball from team shape alone, no ball | The ball is missing 41% of the time | The team with the ball is wider. Correct 87% of the time on unseen matches. With the ball hidden for 5 to 10 seconds this beats "keep the last owner" (88% against 70%), though keeping the owner is better for the first 2 seconds |
| ✅ | Look at what our missed possessions are, then decode owner changes inside missing-ball gaps with the shape model | We find only about 1 in 5 real possessions | Most misses are changes of owner we never saw, often with no ball in view. Decoding inside gaps adds +0.074 to +0.102 on practice matches, reproduced independently. Only safe on top of the stoppage rule. Not switched on |
| 🔬 | Run the same decoder when the ball is visible but nobody is within 3 metres of it (long kicks, loose balls) | Our nearest-player rule is blind there | Running |
| 📋 | Ship the in-gap decoder behind a switch, after the stoppage rule | The largest single practice gain since the first retune | Next code change |
| ❌ | Guess who wins a long ball from where it lands | Idea from the ideation round | Only 8% of gaps start with a fast ball. No signal |
| ❌ | Count how many players hide the ball at the start of a gap, as a sign of a contest | Idea from the ideation round | Same count as picking a random moment |
| ❌ | Trace contacts between players to spot ball transfers | Idea from the ideation round | Adds nothing beyond "who is nearest the ball" |

## Spotting touches of the ball

| | What we tested | Why | What happened |
|---|---|---|---|
| ⚠️ | Detect kicks from sudden ball acceleration | The textbook method, and it works on clean public data | Recall 75 to 83% on clean data. It fails on our ball |
| ✅ | Measure the noise in our ball | To find out why that failed | Our ball path is smoothed over about 0.3 seconds, so a kick is spread out. Once practice matches get the same smoothing the method fails there too, which confirms the cause |
| ❌ | Track the ball as it travels with a player or flies between players (carry and pass evidence) | A different route to touches | Best case finds half of owner changes. Adding it to the decoder cost −0.002 to −0.011. The nearest-player rule already carries the same information |
| ❌ | Hear kicks in the audio | Gives exact kick times | Not above chance |
| 📋 | Undo the smoothing mathematically; flag moments when the whole team turns together | Two more ways to date a change of owner | Low expectation. Not built |
| 📋 | Count how many ball-visible changes of owner can be seen in the data at all | Would tell us the ceiling for any touch method | Not built |

## Making practice matches look like ours

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Find open matches with tracking, ball and possession answers; load them into one format | We cannot tune on our match | 7 matches from the German league (DFL), 3 from Metrica, 20 from SkillCorner. Licences checked. A fourth source, the 2022 World Cup data, is cleared to use but not loaded |
| ✅ | Rebuild their answers to match the organisers' definitions | Each source defines "owner" and "dead" a little differently | Last-touch owner and a ball-out-of-play dead state match ours. SkillCorner's dead state starts at the last kick and most of its runs are cut off by tracking gaps, so it is used for H/A only |
| ✅ | Degrade clean data to look like ours: drop the ball where ours drops, smooth it, match gaps and player counts | Our ball is missing and blurred | Matches the real data on gap lengths, player counts and smoothness. Hidden leaks removed: in the raw data the ball freezes in dead-ball frames and goes missing exactly then |
| ⚠️ | Add a setting for how often the ball stays visible during stoppages | We can't know the true share | The method ranking holds across settings, but the stoppage rule's benefit does not |
| ✅ | Place the resting ball near the player who would restart | A resting ball in the degraded data sat far from any player, which gave it away | A classifier could tell resting balls apart at 0.80 (0.5 means no tell). That fell to 0.63, close to the real data |
| ✅ | Check whether player stillness is exaggerated by the degrading | It is the signal the stoppage rule uses | It is not. Left alone, with an optional setting to weaken it |
| ⚠️ | Test how ball-based clues behave across practice and real data | To catch inflated results | Ball speed behaves the same on both. Other ball clues separate dead from live 0.06 to 0.125 better on practice matches (on a 0.5 to 1 scale), because the degraded resting ball shakes about 20 times more than a real one. Stoppage rules that use ball speed are safe; the others are discounted |
| 🔬 | Make the degraded resting ball as steady as a real one | Fixes the point above | Running |
| 🔬 | Independent re-check of the "how often the ball stays visible" range | It sets which practice setting we select on | Running |
| ✅ | Test whether errors in answer timing change which method wins | Source labels have their own timing | Random jitter hardly matters (≤0.006). A systematic shift of a second does, so we never tune a lag on one source |
| ✅ | Clean obvious tracking errors | Duplicated players, ball spikes | Fixed. No change to the score of the current entry |
| ✅ | Unit check across sources | The review found one source's speeds were in km/h | Fixed |
| 📋 | Sign off the official practice set | Waits on the two fixes above | Not done |
| 📋 | Smaller degrading refinements: where gaps start relative to stoppages, the far tail of ball-to-player distances, player-track noise, 10-frames-per-second sources | Remaining differences from our data | Not done |

## Tuning for the scoring rule

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | A simple first entry: nearest player within 2.5 m owns the ball, otherwise keep the last owner; ignore short flickers; ball out of play is dead | Something valid to submit | 493 possessions, a valid file, identical on every rerun |
| ✅ | Retune that entry on degraded practice matches | The settings were placeholders | Longer holding of the current owner: +0.058 over the organisers' 2-second baseline, 10 of 10 matches. Fewer fragments |
| ⚠️ | Check that gain again later, with a more realistic share of visible stoppage balls | The practice setting had changed | The gain shrinks to about nothing. This is the "level with the organisers" result above |
| ✅ | Measure possession lengths on public matches | To set what a normal possession looks like | Median 4.6 seconds. About 12% are under 1 second, mostly a single touch. We aim for fewer short ones, but do not delete them |
| ✅ | Pick a method for splitting the match into whole possessions rather than frames | Cleaner possessions | A duration-aware decoder was chosen on paper. Not built |
| 📋 | Joint search over hold time and merge settings | The first tune stopped at the edge of its grid | Not done |
| 📋 | Stop the merge rule from absorbing real long turnovers | Part of the missed possessions | Not done |
| 📋 | Statistical decoder that combines all clues and uses possession lengths | Planned main method | Designed, not built |
| 📋 | A model trained on the practice matches | Everything it needs now exists | Not started |
| 📋 | Choose the answer that maximises expected score; scrambles get a different length prior; a second check that removes unlikely spells | Cheaper alternatives from the ideation rounds | Not built |
| 📋 | Weight clues without labels; adapt clue weights per half; train on our own confident frames | Alternatives for combining clues | Not built |
| ❌ | Optimise the score directly with reinforcement learning | Idea from the ideation round | Dropped as heavy on a CPU and risky. The cheaper expected-score idea above covers it |

## Better raw data

| | What we tested | Why | What happened |
|---|---|---|---|
| ❌ | Find the ball in the video ourselves | The tracker loses a visible ball | The ball is 8 to 10 pixels wide. The best detector found it on 36% of frames where tracking had it, and fired on 82% of frames where it did not. The best option also has a restrictive licence. Dropped |
| ✅ | Check how much video detection would help even if it worked | | Long gaps are almost all stoppages we already infer. Best case adds 3 to 7 points of ball coverage at low trust |
| ⚠️ | Check whether the tracker loses balls that are visible on video | A human watching saw one | At least 4.5% of missing time is a lost visible ball |
| ❌ | Predict where the ball is during a live gap (particle filter) | Idea from the ideation round | Low value by the same coverage argument |
| ✅ | Use the broadcast camera's own ball track | A second view | Adds 3.7 points of ball coverage, found by code, and serves as a live-or-dead check |
| 📋 | Bridge short live gaps using video, with camera motion removed | Needs 80% reacquisition within 2 metres to continue | Not built |
| 📋 | Use broadcast cutaways and replays as a dead-ball cue; ball outside the pitch in broadcast view; camera pan as a sensor; the organisers' tactical tags as a prior | Other match-only clues | Not built |
| 📋 | Train a ball detector on a GPU | Needs a GPU we do not rely on | Parked |

## Football-knowledge review

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Watch 14 short video clips, as understanding only, never as labels | Check the model against football | Broadly confirmed. Two corrections: a throw-in sequence is dead although the ball moves, and a keeper involvement is often receive-and-clear, which is live |
| ✅ | Walk through 40 match situations from the Laws | Find what we overlooked | 25 covered, 9 too small to matter, 3 killed by tests, 3 open. The one real gap is who restarts after a stoppage |
| ❌ | Fill short dead-ball islands between two spells of the same team | Looked like noise | Score −0.001 to −0.004 |
| ❌ | Treat an airborne ball as live | A ball in the air cannot be resting | −0.018 |
| ❌ | End a stoppage when a ball retriever is running | Retrievers signal a restart | After the stoppage, play looks live already |
| ❌ | A second ball on the pitch as a stoppage clue | | No sudden ball jumps found |
| ❌ | Substitutions as a stoppage signature | | Extra peripheral players are tracked for a median of 3 seconds. Dropped |
| ✅ | Dismiss rare events: quick free kicks, penalties, red cards, a ball hitting the referee, pitch invaders | Each is under 0.5% of the score | Ignored on purpose |
| ✅ | Check keeper rule length | The Laws changed from 6 to 8 seconds | Noted |
| 📋 | Check that the number of ball-in-play minutes is plausible against typical matches | Sanity band | Waits on the dead-ball prior decision |

## Method and evaluation setup

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Compare two methods by resampling whole matches, not single possessions | Possession-level resampling called 30% of identical methods different, whole-match resampling 5% | Adopted as the only test. Detecting a 0.04 gain needs about 8 matches |
| ✅ | Keep levers that are positive but unproven behind a switch, off by default | Ten matches give little power | A two-tier policy |
| ✅ | Build the evaluation tools and reproduce the organisers' four baselines exactly | Needed before any claim | Matches to the digit |
| ✅ | Match structure: half boundaries, kickoffs, attack direction, the pre-kickoff block | Structure the organisers publish | Done, each constant cited to its source |

## Keeping ourselves honest

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Rules audit of the first entry | Rule breaches disqualify | Passed. One warning (a test reading the public answers) was fixed with a guard that fails on public-answer reads outside one script, frame-number constants, network imports and committed data |
| ✅ | Reproduce the first entry from a fresh copy and a fresh install | One command must regenerate the file | Identical file twice, on a different Python version. Again after the retune |
| ⚠️ | Code review of features and loaders | Hidden leaks and bugs | No leaks. Found the km/h speed bug, fixed |
| ⚠️ | Independent check of the degrading | A leak would inflate every result | Conditional pass. Ball-only clues still separate dead from live on practice data (a separation score of 0.80 to 0.85, where 0.5 is none), which led to the resting-ball fixes above |
| ⚠️ | Independent check of the ball-visibility setting | Its first version had a leak | Found the resting ball was twice as far from players as in real data, which led to the placement fix. Also found the first setting choice (0.5) was outside our label-free range |
| ✅ | Independent check of the placement fix | | Passed, numbers reproduce |
| ✅ | Independent check of the in-gap decoder | Its gain was the largest | Reproduced to the fourth decimal, no leak. On our match it rewrites 12.6% of frames if run alone, mostly inside stoppages, so it must run only after the stoppage rule |
| ⚠️ | Independent review of the whistle detector | Before anyone uses it | Detector sound, but its failure handling was not: partial audio decode passed as success, and events were one frame late. All nine faults fixed |
| ⚠️ | Independent review of SkillCorner | Catch leaks | One quoted figure used an unbalanced time window and is withdrawn. Its dead state differs from ours, hence H/A only |
| ✅ | Public-sample score, report only | Where we stand | See below. Never used to choose anything |
| ✅ | Setting one adoption rule for match-only clues (whistle, cutaways, tags) | Avoid choosing by eye on our match | Whistle adopted, crowd noise parked |

## Where we stand

Our current entry scores about level with the organisers' best simple baseline in realistic practice conditions. It was clearly ahead at the two extreme practice settings (every stoppage hides the ball, or none does), and that lead went away at the more realistic settings.

The stoppage rule for missing balls, built and checked, puts us clearly ahead: +0.05 to +0.08 over the organisers' baseline at every setting we tried. It is not switched on yet. The in-gap team-shape decoder adds +0.08 to +0.10 on top, also not switched on.

The biggest weakness is possessions: we catch about 1 in 5 well enough to count. Following the ball more closely did not help, so the next attempts read team shape.

On the organisers' 5 public minutes the current entry scores 0.423, against 0.326 for their 2-second baseline and 0.424 for their 5-second one. That sample holds only 17 real possessions and we report it without acting on it. Our frame-level score on the same sample is the lowest of the five.

## What we plan to test next

In order:

1. Fix the resting-ball shake in the practice data, then sign off the practice set.
2. The independent re-check of the ball-visibility range.
3. Grow a resting-ball moment into the whole stoppage, and measure false alarms on certainly-live frames.
4. Decode owner changes when nobody is near the ball.
5. Switch on the stoppage rule and the in-gap decoder, on practice matches, and search hold time and merge settings jointly.
6. Choose the restart team from the stoppage type.
7. Race three approaches on the practice set: touch-by-touch, a statistical decoder, and a model trained on practice matches.
8. Narrow the estimate of how often a stoppage shows the ball, using only our match's label-free data.
9. Freeze the method, re-run the rules audit and the from-scratch reproduction.

Lower priority and open: broadcast cutaways, camera pan, tags as a prior, bridging short gaps with video, the 2022 World Cup data.
