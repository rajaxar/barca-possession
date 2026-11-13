# How we work

## Why we practice on other matches

We have the true answers for 5 public minutes of our match, and the rules forbid training or tuning on them. Even if they were allowed, 5 minutes hold only 17 real possessions. The organizers show that tuning a one-setting baseline on this sample picks the wrong setting. So every choice is made on other public matches that have answers.

The main practice set is frozen at 10 matches: 7 from the German league (DFL) and 3 from Metrica. Anything trained is trained with one match held out at a time. A second set of 20 SkillCorner matches is used only for who-has-the-ball questions, never pooled with the 10. Its dead-ball labels are our own derived rule, so it never sets a dead-ball threshold, a stoppage length or a dead share. A dead-ball conclusion must hold on the 10 alone. A who-has-the-ball conclusion must not reverse on the 20.

## How we make them look like ours

Our tracking differs from clean public data in three ways. The ball is missing 41% of the time, in blocks of several seconds. The ball path is smoothed over about 0.3 seconds, which hides the moment of a kick. A few player tracks jump or duplicate.

We copy those properties onto the practice matches: we drop the ball where ours drops, smooth it the same way, and match gap lengths and player counts. We also remove leaks, meaning clues that exist only because the data is clean. In raw public data the ball freezes during a stoppage and goes missing exactly then. A method trained on that would learn the leak. After each change to the copying code, a separate reviewer tests it with a planted leak to confirm the test can detect one.

Two flaws got past this process and were found later.

- **The resting ball.** A ball lying still for a restart shook about 20 times more in the practice data than in a real one. Ball clues therefore looked better on practice matches than they are. We fixed the sign of the shake. A smaller difference in texture remains, so ball-motion clues are still treated as upper bounds.
- **Where the ball hides.** The practice data dropped the ball at random, including during live play. In our match it is missing almost only during stoppages. We had matched how long the gaps are, but not where they fall. Fixing it changed the numbers: Baseline+ went from .216 to .326 and the in-gap decoder's gain fell from +.08 to +.005.

The lesson: every property of the copied data is now checked against our match separately for live play and for stoppages, not only overall. Another check confirmed the practice data's ball-to-player distances match our match, so the copied data is not too harsh.

One setting cannot be measured on our match: how often the ball stays visible during a stoppage. We therefore test every method across a range of that setting and trust only gains that hold across it. The range is estimated from our match without using the answers: between 7% and 23% of stoppages keep the ball visible.

The practice set is about .06 easier than the real match. The organizers' 2-second baseline scores .304 on it and .243 on the real match. We use it to rank methods and never to predict a level. Eight known blind spots are written down with it, such as no cooling breaks or video-review stops in the practice matches.

## How we choose what ships

Every candidate is scored on the frozen practice set with whole-match resampling. A candidate ships by default only if all of these hold:

- its range of gain is above zero over the six main settings (three visible-ball shares by two stillness settings);
- it is not negative in any main or stress setting;
- it survives shifting the answers by half a second;
- the SkillCorner set does not contradict it.

Otherwise it stays behind a switch with a named re-test. We write the pass marks before looking at the result. When a pass mark turned out to measure the wrong quantity, we replaced it and wrote the new one before looking at our match's numbers again.

Practice gains are near-deterministic, so a +.004 gain is real on the practice set but invisible on one match, where noise is about .04. Every practice number is therefore quoted next to the size we expect on our match, scaled by how much of our match the rule touches.

## Why we never tune on the 5 public minutes

The score can be run against them, and we do so once per version, as a report. The number never decides which method we keep. A constant that is justified only by "it scored better on the public sample" is not allowed. Sanity checks use it for alignment, file validity and what the labels mean.

## Why every important result is re-checked

The practice-set comparison quoted in the README is reproducible with one command: `make headline` runs the ten practice matches, three seeds and all twelve settings in one pass, prints the expected rows beside the measured ones, and treats a near miss as a failure. A quick run (`make headline SEEDS=0 CELLS=1:0.05`) is labelled as a smoke test and claims nothing.

Any result that changes a decision goes to a reviewer who did not run it. The reviewer reruns it with their own script, looks for leaks and checks that the numbers match. Several first results changed at this step: a figure was withdrawn for using an unbalanced window, a unit bug showed up in one source, the first visible-ball setting was found to be outside the range our data allows, and a re-implementation of the grown-stoppage code corrected one figure. All are listed in [WHAT_WE_TRIED.md](WHAT_WE_TRIED.md).

When we compare two methods, we resample whole matches rather than single possessions. Possession-level resampling called 30% of identical methods different. Whole-match resampling called 5%, which is the right rate.

## Rules we hold ourselves to

- No hand labels on this match, no hand-corrected tracking, and no rule tied to a specific frame. Structure the organizers publish (half boundaries, the two kickoffs) is used as a constant with its source cited. Everything else about when things happen is found by code.
- No outside data about this match: no event feeds, no corrected tracking, no possession statistics. Data about other matches is fine, and its source is recorded.
- No match data sent to outside services or published. Tracking, frames, audio and video stay on local machines, and so does anything drawn from them. Public audio models run locally.
- The video is watched only to understand football: the Laws, what a stoppage looks like, what a keeper does. Nothing watched becomes a label, a threshold, a parameter or a score, and no claim here rests on it. Every quantity is measured automatically on our own data or on external matches. Where watching raised a question, the answer was re-derived by code.
- One command rebuilds the final file byte for byte, with fixed random seeds and no manual steps. It does so across thread counts, a clean hash-locked install and a Linux container. A test fails on public-answer reads, frame-number constants, network imports and committed data.
- Everything shipped is inspectable: public models and datasets only.
