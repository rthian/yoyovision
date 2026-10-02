# Human judging and AI-judge roadmap

YoYoVision now covers the first useful human-in-the-loop workflow: an admin
creates a video entry, issues isolated judge links, judges record timestamped
positive or negative technical clicks and Freestyle Evaluation, and a head judge
reviews raw trails and panel aggregates. Analysis Review can use those records as
reference evidence without changing the AI score.

The `yoyo-pwa` reference provides broader event operations: judge accounts,
division assignments and judge types, participant queues, division locks,
score revisions, live leaderboards, and head-judge visibility. YoYoVision adds a
video-synchronised evidence layer that the reference application does not make
central. The strongest product direction is to connect those two ideas while
preserving an auditable boundary between human decisions and model output.

## Current gaps

| Gap | Competition impact | Recommended change |
| --- | --- | --- |
| Clicks require a live network request | A venue Wi-Fi failure can lose scoring actions | Store clicks immediately in IndexedDB with client-generated IDs and sequence numbers; sync idempotently in the background |
| One assignment performs TE and FE | Real panels commonly separate technical and evaluation roles | Add `judge_role` (`technical`, `evaluation`, `head`, `shadow`) and require only the fields appropriate to that role |
| Net clicks are averaged directly | Near-simultaneous clicks from different judges are not recognised as one element | Cluster clicks in a configurable time tolerance, expose consensus and outliers, and let the head judge inspect the source clicks |
| Positive/negative clicks have no ruleset meaning | A raw net count is not yet an official technical score | Version click semantics by division and ruleset; preserve raw input and calculate official values in a separate deterministic projection |
| No revision ledger after submission | Corrections cannot be explained during an appeal | Add append-only score revisions with actor, reason, before/after payload, and head-judge approval |
| Invite links identify a session, not a rostered judge | Harder to operate a full event and rotate devices safely | Keep quick links for training, then add judge accounts, event/division assignments, device sessions, and least-privilege access |
| No queue, competitor order, or stage state | The module handles isolated videos rather than a running contest | Add event, division, competitor, performance-order, now-playing, and locked/final states |
| No automated disagreement alerts | Head judges must manually find suspicious panels | Flag click-count spread, FE category range, missing submissions, unusual click cadence, and late edits |
| Human clicks are only displayed beside AI events | They do not yet measure detector quality | Match click clusters to AI event intervals, label matched/missed/extra events, and export calibration metrics by division |

## Recommended build order

1. Make scoring resilient: offline click capture, idempotent sync, visible sync
   state, reconnect testing, and a local recovery export.
2. Model real panel roles and submission requirements. Add explicit head-judge
   include/exclude controls and immutable revision history.
3. Add contest operations: events, divisions, competitors, performance order,
   judge assignments, locks, and result publication controls.
4. Build consensus review. Cluster human click timestamps, show judge agreement,
   and let the head judge resolve outliers while retaining every raw action.
5. Use adjudicated consensus as model evaluation data. Compare AI event intervals
   with human clusters on held-out players and report precision, recall, timing
   error, and performance by division. Do not train directly on unresolved raw
   clicks.
6. Only after prospective shadow trials show stable agreement should AI assist a
   live panel. Start with alerts and replay navigation; keep a human head judge as
   the authority for official results.

## Data boundary

Keep four layers separate: immutable raw judge actions, adjudicated human truth,
model predictions, and official score projections under a named ruleset version.
This separation makes appeals reproducible and prevents a model update from
rewriting historical results.
