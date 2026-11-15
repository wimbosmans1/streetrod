# Sound development guidelines
## Physics and modelling
1. Every sound comes from its own real physical source at a realistic level. No source stands in for another; no artificial boosts or filters to hit a score.
2. One generic sound model; a car is only a parameter set. New physics enters as a general building block whose defaults leave every other car bit-identical.
3. Engine specs and dimensions stay true to the real car. Look up real values first (web, parts data, manuals); never run guessing grids for values that can be found. Estimate only when no source exists, and mark it with its basis.
4. Diagnose before tuning: trace a missing/extra sound stage by stage from source to listener and fix that stage.
5. Stay iPhone/Tesla-light; every change reports its CPU cost.
## Samples (rpm × throttle grid)
6. At least 8–10 s of clean steady audio per cell per listening position (cabin, outside rear, tailpipe), from 2 or more independent videos/cars, with no single clip over 5 s.
7. Tune on one video, validate on the other.
8. Steady stretches only, with known rpm and throttle, and no wind/road/tyre/music noise. Log rpm, throttle, position and source link per clip.
9. Thin or empty cells are reported as gaps, not scored, and filled before tuning in that range.
## Scoring by ear
10. Besides the per-cell frequency balance: engine-order curves (1.5/3/4.5/6 rel. to order 3 across rpm); fail any narrow tone more than about 6 dB above the real recording; match the 1.5–6 kHz noise floor and crest (no steady hiss); 2–8 kHz no higher than real; no start-up clicks or bursts.
11. A candidate becomes the default only after the owner listens. The score narrows, the ear decides.
## Process
12. Cheap, time-boxed steps. Diagnose, report, and get approval before any bigger build.
13. New sounds go live as numbered options "v11 · car pN"; the current default stays until approved. Default changes come with the version number and an info-page note.
