# Glossary

**Possession.** An unbroken stretch of one team having the ball. A dead-ball stretch ends it. The same team having the ball again afterward is a new possession.

**Owner.** The team that last touched the ball in play. It stays with that team through passes in the air and scrambles until the other team gains control.

**Hand-over.** The moment the owner changes from one team to the other.

**Dead ball, stoppage.** Play is stopped: ball out for a throw-in or corner, a foul, a goal, an injury, a cooling break. Labeled `D`. We are not asked who owns a dead ball.

**The score (spellF1).** Our predicted possessions are matched one to one with the true ones by how much they overlap in time. The F-score of that matching is averaged over overlaps of 30% to 70%. A fragmented possession loses on both sides. Differences below about 0.04 on the full match are noise.

**Frame-level score (macroF1).** The usual score over every single frame, averaged over H, A and D. The organizers rank on spellF1, not this.

**Organizers' baselines.** Take the player nearest the ball and vote over a window. Four versions: no window, 1 second, 2 seconds, 5 seconds. "The 2-second baseline" is the one we compare against most.

**Baseline+.** Our first entry. Nearest player within 2.5 meters owns the ball, otherwise the last owner keeps it. Short flickers are held back and ball-out-of-play is dead.

**Shipped entry.** The file we would submit now: Baseline+, then the stoppage rule, then the in-gap decoder.

**Practice matches.** Other professional matches with public answers, used to select methods. The frozen practice set is 10 matches (German league and Metrica). A second set of 20 SkillCorner matches is used only for who-has-the-ball questions.

**Degrading.** Changing a clean practice match so it looks like our data: ball dropped where ours drops, smoothed, with matching gaps. Leaks that only exist in clean data are removed at the same time.

**Ball-visible setting.** The share of stoppages in which the ball stays visible in a practice match. We cannot measure it exactly for our match, so methods are tested across a range, 5% to 23% for the main cells.

**Gap.** A stretch where the tracking has no ball.

**Stoppage rule.** Mark a stretch dead when the ball has been missing for a while and the players are standing still. Also called the gate.

**Resting-ball moment.** The short stretch inside a stoppage when the ball is visible and not moving.

**Grown stoppage.** A resting-ball moment extended to the whole stoppage around it, out to a set radius from the ball.

**Sub-gap rule.** A second stoppage rule that works inside missing-ball gaps the main rule left alone.

**In-gap decoder.** Reads which team has the ball from team shape inside a gap, and so finds changes of owner the ball would have shown.

**Decoder.** The step that turns per-frame evidence into a sequence of H, A and D stretches.

**Expected-score decoding.** Choosing the sequence with the best expected score over the plausible sequences, rather than the single most likely one.

**Leak.** A clue in the practice data that would not exist in ours and would inflate a result.

**Range in brackets.** The range a gain should lie in with about 95% confidence, from resampling whole matches. If it excludes zero, the gain is not luck.

**Switch, tier.** A gain that is positive but not proven enough ships behind a switch, off by default. A gain that passes every check ships on.

**Pre-registered.** The pass mark and test are written down before the result is seen.

**Public sample.** The 5 minutes of our match with published answers. We report a score on it and never tune on it.
