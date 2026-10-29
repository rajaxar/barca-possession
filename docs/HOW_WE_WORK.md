# How we work

## Why we practise on other matches

We have the true answers for 5 public minutes of our match, and the rules forbid training or tuning on them. Even if they were allowed, 5 minutes hold 17 real possessions. The organisers show that tuning a one-setting baseline on this sample picks the wrong setting. So every choice is made on other public matches that have answers: 7 from the German league (DFL), 3 from Metrica and 20 from SkillCorner.

## How we make them look like ours

Our tracking differs from clean public data in three ways. The ball is missing 41% of the time, in blocks of several seconds. The ball path is smoothed over about 0.3 seconds, which hides the moment of a kick. A few player tracks jump or duplicate.

We copy those properties onto the practice matches: we drop the ball where ours drops, smooth it the same way, and match gap lengths and player counts. We also remove leaks, meaning clues that exist only because the data is clean. In raw public data the ball freezes during a stoppage and goes missing exactly then. A method trained on that would learn the leak. After each change to the copying code, a separate reviewer tests it with a planted leak to confirm the test can detect one.

One setting cannot be measured on our match: how often the ball stays visible during a stoppage. We therefore test every method across a range of that setting and trust only gains that hold across it. The range itself is estimated from our match without using the answers.

## Why we never tune on the 5 public minutes

The score can be run against them, and we do so once per version, as a report. The number never decides which method we keep. A constant that is justified only by "it scored better on the public sample" is not allowed. Sanity checks use it for alignment, file validity and what the labels mean.

## Why every important result is re-checked

Any result that changes a decision goes to a reviewer who did not run it. The reviewer reruns it with their own script, looks for leaks and checks that the numbers match. Several first results changed at this step: a figure was withdrawn for using an unbalanced window, a unit bug showed up in one source, and the first practice-match setting was found to be outside the range our data allows. We say so in [WHAT_WE_TRIED.md](WHAT_WE_TRIED.md).

When we compare two methods, we resample whole matches rather than single possessions. Possession-level resampling called 30% of identical methods different. Whole-match resampling called 5%, which is the right rate.

## Rules we hold ourselves to

- No hand labels on this match, no hand-corrected tracking, and no rule tied to a specific frame. Structure the organisers publish (half boundaries, the two kickoffs) is used as a constant with its source cited. Everything else about when things happen is found by code.
- No outside data about this match: no event feeds, no corrected tracking, no possession statistics. Data about other matches is fine, and its source is recorded.
- No match data sent to outside services or published. Tracking, frames and video stay on local machines, and so does anything drawn from them.
- One command rebuilds the final file byte for byte, with fixed random seeds and no manual steps. A test fails on public-answer reads, frame-number constants, network imports and committed data.
- Everything shipped is inspectable: public models and datasets only.
