# What we tried

The data is one full match, Barcelona against Athletic Club: where every player and the ball are, 25 times a second, plus video. For every moment we give one answer: Barcelona has the ball, Athletic has the ball, or the ball is dead (a throw-in, a foul, a goal, an injury).

The score is spellF1. A possession is an unbroken stretch of one team having the ball. We are credited for each possession we find, give to the right team, and place with roughly the right start and end. Splitting one possession into fragments costs a wrong prediction and also damages the real possession it was cut from. Missing a possession costs less. Differences under about 0.04 on the full match are noise.

We have answers for 5 public minutes only, and the rules forbid tuning on them. So we practice on other professional matches with public answers, degraded to look like ours ([HOW_WE_WORK.md](HOW_WE_WORK.md)). Terms are in [GLOSSARY.md](GLOSSARY.md).

Legend: ✅ works or adopted. ❌ does not work. ⚠️ partly, or with a caveat. Ideas we listed but never tested are left out.

"Shipped" means the step is in the submitted file. "Behind a switch" means the code exists and is tested but is off in that file. Numbers in square brackets are the range we are 95% sure the true gain lies in, from resampling whole matches.

## When is play stopped?

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Treat the ball vanishing from tracking, with players standing still, as a stoppage (the stoppage rule) | The ball is missing 41% of the time, in long blocks | About 85% of missing time is a stoppage, found by two independent routes (a trained classifier and a check against the broadcast camera). Shipped. On the corrected practice data it adds +.073 [.065, .081] over Baseline+, positive in 10 of 10 matches. We expect +.06 to +.08 on our match. On our match it moves the dead share from 11% to 41% |
| ⚠️ | Check on the 5 public minutes whether a missing ball means a stoppage (report only) | A first look at our own data | Too little data: 56%, with a range from 0 to 96%. Two long gaps in it were live play. The larger measurements above showed such gaps are short and easy to tell apart, so this first reading was too pessimistic |
| ✅ | Check which long missing-ball stretches are live play (tracker dropouts) | If many are live, the rule above cuts real possessions | Only 0.9% of missing time looks stopped but is live |
| ⚠️ | Redo that check, excluding throw-ins (the ball travels but play is dead) | Video review showed successive throw-ins fooled the "ball moving means live" test | Provably live missing time fell from 4.1% to 3.0%. The earlier claim "no live gap longer than 11 seconds" had no evidence behind it, because no long gap had a usable second view. Withdrawn |
| ⚠️ | Test the stoppage rule across how often the ball stays visible during stoppages | We do not know that share for our match | The gain is never negative. It shrinks as more stoppages show the ball and is zero when all do |
| ✅ | Narrow our match's visible-ball share without using the answers | It sets which practice settings count | Between 7% and 23% of stoppages keep the ball visible, about 4.5% of graded time at the center. Our tracker's ability to see a resting ball is the largest remaining uncertainty |
| ✅ | Detect the referee's whistle in the video's audio | Whistles mark fouls and offsides, which ball-out-of-play does not | 74 whistles found; 21 of 55 graded ones are discarded as fan whistles. Stillness 3 seconds after a whistle is 51% against 35% at random. It leads the tracking by about 2 seconds, so it helps find where a stoppage starts |
| ❌ | Replace our whistle detector with a public audio-event model (PANNs, run on a local CPU) | A trained model might hear what ours misses | It scores a synthetic whistle tone at .115 but our match's real whistles at about .001. It cannot hear them. Ours stays |
| ❌ | Use crowd loudness to find stoppages | Cheap extra signal | It tracks microphone placement, not the match. Weak, and it disappears once we account for the missing ball |
| ✅ | Treat a goalkeeper holding or handling the ball as live play | A keeper picking the ball up looks like a stoppage | 14 such episodes, 0.55% of graded time, and our stoppage rules score them as live. The broadcast tracker, an independent automatic source, puts the ball at that keeper on 0.91 of 754 frames against 0.03 for a control, and no episode follows a whistle or a ball out of play |
| ⚠️ | Find stoppages where the ball is still visible (resting for a free kick, held for a throw-in) | About two thirds of dead-ball time has a visible ball | The resting-ball moment is found about 93% reliably and covers 2.8% of graded time. But no decision changed |
| ❌ | Use only that resting-ball detector as a stoppage marker | It is precise (95 to 98%) | It covers only about 3% of the stoppage it sits in. The fragments cut real possessions apart and the score fell in all seven variants (best −0.0003) |
| ✅ | Grow each resting-ball moment out into the whole stoppage | The detector's problem was size, not accuracy | Adopted. Grown frames are about 95% true stoppage. The wider the radius the better: on corrected practice data +.0032 at 4 m, +.0051 at 12 m with a 30-second cap, +.0055 at 25 m (all ranges above zero). We expect +.001 to +.005 on our match. Only safe on top of the stoppage rule: on raw Baseline+, 63.7% of the changed frames land in dead gaps. An independent re-implementation matched, and corrected one figure (55% became 63.7%) |
| ✅ | Add the sub-gap rule: inside missing-ball stretches the main rule left alone, call a stoppage when the stoppage probability stays above 50% for 2 seconds | Missed stoppages sit in those stretches, not in very short gaps as we first guessed (gaps under 2 seconds are 0.05% of our match) | Adopted. +.0044 on corrected practice data (was +.0160 before the fix), 0 of 10 matches negative. We expect about +.002 on our match. With the visible-ball dead time removed, the rest of our match's ball-in-play time lands in the normal football range |
| ❌ | Split long stoppage blocks into their separate stoppages | Our football-knowledge review suggested blocks join several stoppages | On the old practice data +.0121. But those blocks had 13% to 18% live play only because the practice data hid the ball during live play. On our match the rule fires on 0.09% of graded time. After the fix it scores −.0011, negative in 10 of 10 matches. Dropped. A block holds 1.0 to 1.8 stoppages, not "several" |
| ❌ | Choose who restarts by stoppage type (a ball out over the touchline goes to the other team) | After a stoppage we had the right team only 70% of the time | Falsified as pre-registered. Restart-team accuracy 2 seconds after play resumes is 77.1% now (70% was raw Baseline+). Adding the stoppage type changed it by −0.4 points and made the score worse in every variant. The flip rate is 41% for every type. The cause: 73% of stoppage blocks before a restart are over 16 seconds and bridge several stoppages and live play |
| ❌ | Find stoppages where the ball is missing but players keep moving, using the whistle | About 6% of graded time | The whistle precedes only 5.8% of these stretches against 4.7% by chance. A perfect detector would add only +.005. Parked |
| ✅ | Catalog the cases that look live but are dead, and the reverse | To know what each stoppage rule costs | The ball must be nearly still: above 1.0 m/s of team speed the rule misfires. A cooling break in the first half was found from the data alone, so our share of dead ball is higher than in any practice match |
| ✅ | Ship the two small add-ons (grown stoppages, sub-gap rule) | Both gains are positive but each is a few thousandths | Shipped and independently reproduced. On the corrected practice data grown stoppages add +.005 and the sub-gap rule +.004; together +.011 [.010, .013], 0 of 10 matches negative. They change 2.9% and 0.7% of our match's graded frames, none of it inside a dead gap. The build fails if an add-on touches frames outside the stoppage rule's footprint |

## Who has the ball when we can't see it?

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Guess the team with the ball from team shape alone, no ball | The ball is missing 41% of the time | The team with the ball is wider. Correct 87% of the time on unseen matches. With the ball hidden for 5 to 10 seconds this beats "keep the last owner" (88% against 70%), though keeping the owner is better for the first 2 seconds |
| ✅ | Look at what our missed possessions are | We find only about 1 in 5 real possessions | Most misses are changes of owner we never saw, often with no ball in view. About 44% of those changes have no tracked ball at all, where any ball-based rule is blind |
| ⚠️ | Decode owner changes inside missing-ball gaps with the shape model (the in-gap decoder) | The clue fits the largest group of misses | Shipped, after the stoppage rule only. On the first practice data it added +.074 to +.102, reproduced independently, but that data hid the ball in live play, so the gain was inflated. On corrected data, on top of the stoppage rule alone, it adds +.005 [.004, .007], positive in 10 of 10 matches. On the final pipeline (with grown stoppages and the short-spell floor) it adds +.0017 [+.0004, +.0030], or [−.0011, +.0041] once the degrader's random seeds are included, negative in 3 of 10 matches: level with zero ([possession_rules.md](possession_rules.md), 5c). We expect +.002 to +.005 on our match. Run alone it rewrites 12.6% of our match, mostly inside stoppages. After the stoppage rule it touches 1.2% |
| ❌ | Run the same decoder when the ball is visible but nobody is within 3 meters of it (long kicks, loose balls) | Our nearest-player rule is blind there | Frame accuracy there rose from 77% to 84%, but the score fell: −.0031 and −.0111 at two visible-ball settings. More changes of owner broke more possessions than they fixed |
| ⚠️ | Use the organizers' tactical tags (Smart Tagging) as a possession prior | A second view of who attacks | Parked. The attacker code is reliable 60% to 90% of the time but adds nothing to team shape (−.001 [−.023, +.023]). It supports the stoppage rule: on ball-missing live frames the tags agree with Baseline+ 45% of the time (chance) and with the stoppage-rule entry 64%. It leans against the in-gap decoder on our match (31 gaps: 3 better, 6 worse, range includes zero) |
| ⚠️ | Decode for the score itself: pick the segmentation with the best expected spellF1 | Per-frame accuracy and the score disagree | It does not replace the shipped pipeline (−.0074 to −.0159, because it produces too few possessions). It is the right objective, so it is kept behind a switch. Under it the "nobody near the ball" clue turns positive (+.0057 [+.001, +.010]) |
| ❌ | Guess who wins a long ball from where it lands | Brainstorm idea | Only 8% of gaps start with a fast ball. No signal |
| ❌ | Count how many players hide the ball at the start of a gap, as a sign of a contest | Brainstorm idea | Same count as picking a random moment |
| ❌ | Trace contacts between players to spot ball transfers | Brainstorm idea | Adds nothing beyond "who is nearest the ball" |

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
| ✅ | Check that the two sources mark a possession change at the same moment | The delay result might be a clock difference | They do: both mark the new team's first touch, within one frame. So our early hand-overs are how our method behaves, not a labeling difference |

## Making practice matches look like ours

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Find open matches with tracking, ball and possession answers; load them into one format | We cannot tune on our match | 7 matches from the German league (DFL), 3 from Metrica, 20 from SkillCorner. Licenses checked. A fourth source, the 2022 World Cup data, was granted by an access form with no published licence; it is **not used** in the build or the training |
| ✅ | Rebuild their answers to match the organizers' definitions | Each source defines "owner" and "dead" a little differently | Last-touch owner and a ball-out-of-play dead state match ours. SkillCorner's dead state is our own derived rule: about 45% of its stoppage runs never see the ball leave play, and most runs are cut off by tracking gaps. So SkillCorner is H/A only and never sets a dead-ball rule |
| ✅ | Degrade clean data to look like ours: drop the ball where ours drops, smooth it, match gaps and player counts | Our ball is missing and blurred | Matches the real data on gap lengths, player counts and smoothness. Hidden leaks removed: in the raw data the ball freezes in dead-ball frames and goes missing exactly then |
| ✅ | Add a setting for how often the ball stays visible during stoppages | We can't know the true share | Methods are tested at 5%, 12% and 23% as the main settings, plus 30% and 50% as stress tests. The method ranking holds across settings. The size of the stoppage rule's benefit does not |
| ✅ | Place the resting ball near the player who would restart | A resting ball in the degraded data sat far from any player, which gave it away | A classifier could tell resting balls apart at 0.80 (0.5 means no tell). That fell to 0.63. A bug that teleported the resting ball between players 100 to 560 times a match was fixed (now about 15) |
| ✅ | Check whether player stillness is exaggerated by the degrading | It is the signal the stoppage rule uses | It is not. Left alone, with a softer setting that we always report beside the default |
| ⚠️ | Practice-data flaw 1: the resting ball shook about 20 times more than a real one | Ball clues separated dead from live far better on practice data than on real data | Fixed in sign. The shake score separating dead from live fell from .984 to .49, where real data sits near .42. A weaker tell remains: a classifier on the ball's texture alone scores .92 against a target of .80. On 20 SkillCorner matches never used for tuning it scores .907. We still treat ball-motion clues as upper bounds. Ball speed alone was always sound |
| ✅ | Practice-data flaw 2: the degrader hid the ball at random, including during live play | Our match hides it almost only during stoppages | The share of live frames with no ball was .42 on practice data against about .03 on our match, and 35% to 39% of long gaps were live against 4% or less. Fixed with a cap on live-play gaps and a spread on gap edges: live-frame missing share now .044, long-gap live share .021, our match's gap profile matched on four measures. This moved every number: Baseline+ .216 to .326, the in-gap decoder +.082 to +.006, the stoppage rule's gain +.085. An independent check passed it on the 10 matches and on 20 held-out SkillCorner matches |
| ✅ | Check the practice data's ball-to-player geometry against our match | The 28 points lost in the ceiling check could have been a degrading artefact | It matches: nearest-player median 2.29 m against 2.35 m on ours, a player within 1 m .209 against .211, two-team contests .029 against .026. The clean data is the outlier (too sharp). Only the kinks are slightly over-blurred, by about 2 points |
| ✅ | Freeze the official practice set | Every later comparison needs one yardstick | 10 matches, 3 random seeds, 6 main settings (3 visible-ball shares by 2 stillness settings), the stress settings, and a ±0.5 second answer-shift check. SkillCorner is a separate H/A-only set. Eight blind spots are recorded, among them: no cooling breaks or video-review stops, Metrica over-calls dead ball around advantage play, short sliver gaps our match lacks, and low resting-ball recall |
| ✅ | Compare the practice set's difficulty with the real match | To know what its levels mean | The organizers' 2-second baseline scores .304 on it against .243 on the real match, so it is about .06 too easy. We use rankings, never levels |
| ✅ | Test whether errors in answer timing change which method wins | Source labels have their own timing | Random jitter hardly matters (≤0.006). A systematic shift of a second does, so we never tune a lag on one source |
| ✅ | Clean obvious tracking errors | Duplicated players, ball spikes | Fixed. No change to the score of the entry |
| ✅ | Unit check across sources | The review found one source's speeds were in km/h | Fixed |
| ✅ | Make cache writes safe when several processes run at once | A failed test showed they could collide | Fixed. A four-process test now passes. The file did not change |
| ✅ | Build one shared harness that scores and compares methods on the frozen practice set the same way | Comparing methods needs identical scoring | Built. It also runs the half-second shift check, which is how the Baseline+ lead was found to be fragile |

## Tuning for the scoring rule

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | A simple first entry: nearest player within 2.5 m owns the ball, otherwise keep the last owner; ignore short flickers; ball out of play is dead (Baseline+) | Something valid to submit | 493 possessions, a valid file, identical on every rerun |
| ⚠️ | Retune that entry on degraded practice matches | The settings were placeholders | Longer holding of the current owner gave +.058 over the organizers' 2-second baseline on the first practice data. That gain shrank to about nothing with a more realistic visible-ball share, then came back smaller: after both practice-data fixes Baseline+ is +.022 [.006, .040] ahead of the 2-second baseline. That lead does not survive the half-second answer-shift check (at −0.5 seconds it is +.0066, range −.010 to +.025), so Baseline+ is about level with, or slightly ahead of, the 2-second baseline. The orderings that matter survive every shift: stoppage rule above Baseline+, and the shipped entry above the stoppage rule |
| ✅ | Measure possession lengths on public matches | To set what a normal possession looks like | Median 4.6 seconds. About 12% are under 1 second, mostly a single touch |
| ✅ | Pick a method for splitting the match into whole possessions rather than frames | Cleaner possessions | A duration-aware decoder was chosen on paper. Built later, it lost to the shipped pipeline ([TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md), section 1) |
| ❌ | A label-free score on our match to rank close methods | Insurance in case the practice set is weak | Failed its pre-registered bar: it picked the practice winner 47% of the time where we required 70%. It does separate Baseline+ from any entry with the stoppage rule (97%). Kept as a red-flag check only. The practice set stays the selector |
| ❌ | Optimize the score directly with reinforcement learning | Brainstorm idea | Dropped as heavy on a CPU and risky. Expected-score decoding covers it |

## Better raw data

| | What we tested | Why | What happened |
|---|---|---|---|
| ❌ | Find the ball in the video ourselves | The tracker loses a visible ball | The ball is 8 to 10 pixels wide. The best detector found it on 36% of frames where tracking had it, and fired on 82% of frames where it did not. The best option also has a restrictive license. Dropped |
| ✅ | Check how much video detection would help even if it worked | | Long gaps are almost all stoppages we already infer. Best case adds 3 to 7 points of ball coverage at low trust |
| ⚠️ | Check whether the tracker loses balls that are still visible | Some of the missing 41% may be the tracker rather than the game | At least 4.5% of missing time is a lost visible ball, measured with the broadcast tracker as an independent automatic source. No human observation enters the number |
| ❌ | Predict where the ball is during a live gap (particle filter) | Brainstorm idea | Low value by the same coverage argument |
| ✅ | Use the broadcast camera's own ball track | A second view | Adds 3.7 points of ball coverage, found by code, and serves as a live-or-dead check |

## Football-knowledge review

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Watch short clips to understand the game, never to check our output | Reading the Laws is not the same as seeing them applied | Understanding only: no label, threshold, parameter or score comes from it. Two questions it raised were answered automatically: whether a moving ball can be dead (by the broadcast ball track, which now excludes motion starting off the pitch) and what counts as keeper involvement (the 0.91-of-754 broadcast check above) |
| ✅ | Walk through 40 match situations from the Laws | Find what we overlooked | 25 covered, 9 too small to matter, 3 killed by tests, 3 open. The one real lead, who restarts after a stoppage, was then tested and falsified (see above) |
| ✅ | Look at what sits inside our long stoppage blocks, to frame a question | Reading our own output is not evidence, so the question has to be answered by measurement | Is play resuming inside a block? Measured automatically: the median block end is −0.12 s, with play resuming inside only a 14–20% tail, and the match's live share is inside the external range once the clock is normalised |
| ❌ | Fill short dead-ball islands between two spells of the same team | Looked like noise | Score −0.001 to −0.004 |
| ❌ | Treat an airborne ball as live | A ball in the air cannot be resting | −0.018 |
| ❌ | End a stoppage when a ball retriever is running | Retrievers signal a restart | After the stoppage, play looks live already |
| ❌ | A second ball on the pitch as a stoppage clue | | No sudden ball jumps found |
| ❌ | Substitutions as a stoppage signature | | Extra peripheral players are tracked for a median of 3 seconds. Dropped |
| ✅ | Dismiss rare events: quick free kicks, penalties, red cards, a ball hitting the referee, pitch invaders | Each is under 0.5% of the score | Ignored on purpose |
| ✅ | Check keeper rule length | The Laws changed from 6 to 8 seconds | Noted |
| ✅ | Check that the amount of ball in play is plausible against typical matches | The written sanity band is 53 to 55 minutes, an absolute figure for a ~98-minute match | As a share of the playing clock, our entry is **55.3%** against 58.2 ± 6.4% for the ten practice matches, inside the noise (z −0.46). The old "62 minutes, still high" reading compared a 104-minute clock against a 98-minute band. At most **0.9 minute** of our live time has any dead-ball evidence |

## Method and evaluation setup

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Compare two methods by resampling whole matches, not single possessions | Possession-level resampling called 30% of identical methods different, whole-match resampling 5% | Adopted as the only test. Detecting a 0.04 gain needs about 8 matches |
| ✅ | Keep changes that are positive but unproven behind a switch, off by default | Ten matches give little power | A change ships only if its range is above zero overall, it is non-negative in every main and stress setting, it survives a half-second answer shift, and SkillCorner does not contradict it |
| ✅ | Quote the expected gain on our match beside every practice number | Practice gains are near-deterministic, so a +.004 gain is real there and invisible on one match | Adopted |
| ✅ | Check every practice-data property against our match separately for live play and stoppages, not only overall | The ball-hiding flaw passed an overall check and failed a conditional one | Adopted |
| ✅ | Build the evaluation tools and reproduce the organizers' four baselines exactly | Needed before any claim | Matches to the digit |
| ✅ | Match structure: half boundaries, kickoffs, attack direction, the pre-kickoff block | Structure the organizers publish | Done, each constant cited to its source |

## Keeping ourselves honest

| | What we tested | Why | What happened |
|---|---|---|---|
| ✅ | Rules audit of the first entry | Rule breaches disqualify | Passed. One warning (a test reading the public answers) was fixed with a guard that fails on public-answer reads outside one script, frame-number constants, network imports and committed data |
| ✅ | Reproduce the entry of the time (Baseline+, stoppage rule, in-gap decoder) from a clean copy | One command must regenerate the file | An independent reviewer got the identical file from a clean clone, fresh install and cold caches, passed 5 of 5 checks, and the rules audit was clean |
| ⚠️ | Reproduce it across thread counts, a clean hash-locked install and a Linux container | The file must not depend on the machine | Identical every time, with dependency versions locked. The ARM architecture (Apple silicon) is untested |
| ⚠️ | Code review of features and loaders | Hidden leaks and bugs | No leaks. Found the km/h speed bug, fixed |
| ⚠️ | Independent checks of the degrading | A leak would inflate every result | The smoothing and the visible-ball setting each had a leak, found and fixed. The resting-ball placement, shake fix and live-play-gap fix each passed; the last also held on 20 SkillCorner matches never used in tuning. Ball-only clues still separate dead from live a little on practice data, so they are discounted |
| ✅ | Independent check of the in-gap decoder | Its gain was the largest | Reproduced to the fourth decimal, no leak. It ships only after the stoppage rule |
| ✅ | Independent re-implementation of the grown stoppages and the visible-ball range | Both feed decisions | Passed, numbers reproduce |
| ⚠️ | Independent review of the whistle detector | Before anyone uses it | Detector sound, but its failure handling was not: partial audio decode passed as success, and events were one frame late. All nine faults fixed |
| ⚠️ | Independent review of SkillCorner | Catch leaks | One quoted figure used an unbalanced time window and is withdrawn. Its dead state differs from ours, hence H/A only |
| ✅ | One adoption rule for match-only clues (whistle, cutaways, tags) | Avoid choosing by eye on our match | Whistle adopted, crowd noise parked |

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

The shipped entry is Baseline+, then the stoppage rule for missing balls, then grown stoppages for visible-ball stoppages, then the in-gap decoder, then the sub-gap rule, then the short-spell floor. It builds the same file on every rerun, across thread counts, a clean install and a Linux container. Independent reviewers reproduced it, including the two add-ons.

On the corrected practice data, averaged over the six main settings:

| method | score |
|---|---|
| Organizers' 2-second baseline | .304 |
| Baseline+ | .326 |
| Baseline+ and stoppage rule | .399 |
| Shipped entry before the short-spell floor | .404 |
| Shipped entry, frozen | **.4247** |

The shipped entry is +.100 [.088, .113] over the organizers' 2-second baseline. The stoppage rule supplies +.073 of that, the in-gap decoder +.005 on top of the stoppage rule alone (+.0017 on the final pipeline, level with zero), and the two add-ons about +.011 on top of the decoder. On our match we expect the stoppage rule to add +.06 to +.08 and the rest +.005 to +.012. The practice set is about .06 easier than the real match, so we trust the order of the methods, not the levels.

The ordering is not sensitive to how hard the practice data is degraded: 20% cleaner or 20% messier data gives the same order (our entry, the stoppage rule, Baseline+, the organizers' 2-second baseline), with the entry ahead of the stoppage rule by +.020 to +.023 both ways. The short-spell floor is worth +.0113 on the final pipeline, positive in 10 of 10 matches. It is a trade, not free pruning: it deletes about 84 real short possessions to remove about 380 false ones. The best label-free separator of the two reaches only 20.5% precision against an 18.1% base, and a rule built on it lost .0031, worse in 9 of 10 matches. Keeping exactly the right ones would be worth .0089, so the simple rule stays.

The biggest weakness is possessions: we catch about 1 in 5 well enough to count. Following the ball more closely did not help. About 13.5% of changes of owner cannot be seen from positions, and our hand-overs often land more than a second from the real one.

On the organizers' 5 public minutes the shipped entry scores **.4686**, against .326 for their 2-second baseline and .424 for their 5-second one. Our frame-level score there is **.7199** against .581 for their best. It is first on both. That sample holds only 17 real possessions and we report it without acting on it.

The alternative cores raced against this pipeline, and the attempts after the method was frozen, are in [TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md), sections 1 and 5.
