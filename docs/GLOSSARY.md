# Glossary

**Possession.** An unbroken stretch of one team having the ball. A dead-ball stretch ends it. The same team having the ball again afterwards is a new possession.

**Owner.** The team that last touched the ball in play. It stays with that team through passes in the air and scrambles until the other team gains control.

**Dead ball, stoppage.** Play is stopped: ball out for a throw-in or corner, a foul, a goal, an injury, a cooling break. Labelled `D`. We are not asked who owns a dead ball.

**The score (spellF1).** Our predicted possessions are matched one to one with the true ones by how much they overlap in time. The F-score of that matching is averaged over overlaps of 30% to 70%. A fragmented possession loses on both sides. Differences below about 0.04 on the full match are noise.

**Frame-level score (macroF1).** The usual score over every single frame, averaged over H, A and D. The organisers rank on spellF1, not this.

**Organisers' baseline.** Take the player nearest the ball and vote over a window. Four versions: no window, 1 second, 2 seconds, 5 seconds.

**Baseline+.** Our first entry. Nearest player within 2.5 metres owns the ball, otherwise the last owner keeps it. Short flickers are held back and ball-out-of-play is dead.

**Practice matches.** Other professional matches with public answers, used to select methods. Also "practice set" when we mean the agreed collection.

**Degrading.** Changing a clean practice match so it looks like our data: ball dropped where ours drops, smoothed, with matching gaps. Leaks that only exist in clean data are removed at the same time.

**Ball-visible setting.** The share of stoppages in which the ball stays visible in a practice match. We cannot measure it exactly for our match, so methods are tested across a range.

**Stoppage rule.** Mark a stretch dead when the ball has been missing for a while and the players are standing still. Also called the gate.

**Resting-ball moment.** The short stretch inside a stoppage when the ball is visible and not moving.

**Leak.** A clue in the practice data that would not exist in ours and would inflate a result.

**Public sample.** The 5 minutes of our match with published answers. We report a score on it and never tune on it.

**Switched on.** Part of the entry we would submit. Several checked improvements are not switched on yet.

**Decoder.** The step that turns per-frame evidence into a sequence of H, A and D stretches. The in-gap decoder does this inside stretches where the ball is missing.
