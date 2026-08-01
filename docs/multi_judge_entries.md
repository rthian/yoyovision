# Multi-Judge Video Entries (Design)

**Status:** Draft design — implement on `feature/multi-judge-entries`, merge to `main` when v1 is ready.  
**Related:** [`yoyo-pwa`](../../yoyo-pwa) (live panel UX reference), `ml/src/yoyovision_ml/scoring/judges.py` (aggregation / click types), [`ruleset.md`](ruleset.md), [`data_model.md`](data_model.md).

## 1. Goal

Let an **admin** create a **Judging Entry** that references **one or more videos**. Each assigned **judge** receives a **private invite link** (and QR) to watch those videos, enter Freestyle Evaluation scores, and submit — without seeing other judges. Admin aggregates the panel, optionally compares AI / shadow analyses, and can configure how AI participates in the official total.

Owner **Review** (single-user analysis review) stays unchanged as a separate workflow.

## 2. Locked product decisions

| Topic | Decision |
| --- | --- |
| Use cases | **Both** training panels and contest panels (`entry.mode`) |
| Owner review | **Separate** from Judging Entry |
| Judge identity | **Name-only** — no judge accounts in v1 |
| Invite | Unique per-judge token URL + **Share** + **QR** on admin dashboard |
| Isolation | Judges never see other judges’ names, scores, panel average, or (by default) AI results |
| Token lifetime | **2-day expiry** from issue / last rotate; admin can revoke or re-issue |
| Entry contents | **Multiple videos** per entry (ordered list) |
| Clicker | **v2** — selectable training-only vs TE-driving modes |
| AI in official total | Per-entry switchable profile **A / B / C** (see §6) |
| v1 scope | FE + invites + aggregation + AI profiles; **no clicker** |
| Aggregation | `auto` default sized for **5–10** judges (see §7) |
| Branching | Develop on `feature/multi-judge-entries`; merge to `main` when ready |

## 3. Non-goals (v1)

- Official IYYF-certified rulesets or learned final scores.
- Judge login / password accounts (token link is auth).
- Public leaderboard visible to judges.
- Timestamp clicker / TE click scoring (v2).
- Replacing owner Review UI.
- Email delivery of invites (admin copies Share / shows QR; email is optional later).

## 4. Personas & access

| Role | Capabilities |
| --- | --- |
| **Admin** | Create/edit/lock entries; add/revoke judges; Share/QR; see all scores; configure AI mix + aggregation; attach official/shadow analyses |
| **Judge** | Open valid invite link; watch assigned videos; draft/submit own FE only |
| **Owner** (existing) | Unchanged upload / analyze / review / corpus path |

Judges authenticate solely via `invite_token`. Tokens are secrets (treat like passwords). QR is per-judge only — never project a single panel QR.

## 5. Domain model

### 5.1 JudgingEntry

| Field | Notes |
| --- | --- |
| `id` | UUID |
| `title` | Human label |
| `mode` | `training` \| `contest` |
| `status` | `draft` \| `open` \| `locked` |
| `ruleset_version` | Scoring ruleset stamp |
| `ai_mix_profile` | `A` \| `B` \| `C` (§6) |
| `aggregation_mode` | `auto` \| `simple_mean` \| `trim_1` \| `trim_2` (§7) |
| `created_by` | Admin user id |
| `due_at` | Optional deadline (display only in v1) |
| `created_at` / `updated_at` | Timestamps |

### 5.2 JudgingEntryVideo (ordered)

| Field | Notes |
| --- | --- |
| `entry_id` | FK |
| `video_id` | FK to existing `VideoAsset` |
| `sort_order` | 0-based play / score order |
| `official_analysis_id` | Optional completed non-shadow analysis |
| `shadow_analysis_id` | Optional completed shadow analysis |

One entry → N videos. Judges score **each video** separately (one FE submission per judge × video).

### 5.3 JudgeAssignment

| Field | Notes |
| --- | --- |
| `entry_id` | FK |
| `display_name` | Name-only identity |
| `invite_token_hash` | Store hash only; never store raw token at rest |
| `token_prefix` | Short prefix for admin UI (e.g. first 8 chars) |
| `token_expires_at` | Issued_at + **48 hours**; updated on rotate |
| `include_in_results` | Default true |
| `is_shadow` | If true, excluded from panel aggregate (training / spare) |
| `status` | `pending` \| `in_progress` \| `submitted` (derived or stored) |
| `revoked_at` | Null unless revoked |

Raw invite URL shape:

```text
/judge/{raw_token}
```

Token grants access to **all videos** on that entry for that judge only.

### 5.4 JudgeFreestyleScore (v1 scoring unit)

| Field | Notes |
| --- | --- |
| `assignment_id` | FK |
| `entry_video_id` | FK to JudgingEntryVideo |
| `execution` … `showmanship` | Eight 0–10 nullable FE fields (same as product FE) |
| `notes` | Optional |
| `is_submitted` | Draft vs submitted |
| `submitted_at` | Set on submit |
| Unique | `(assignment_id, entry_video_id)` |

Maps to existing ML type `JudgeFreestyleScore` / domain `FreestyleEvaluation`.

### 5.5 v2 (documented, not built)

- `JudgeClick` rows: `assignment_id`, `entry_video_id`, `timestamp_ms`, optional `label`.
- Entry flag `click_mode`: `training_only` \| `technical_score`.
- Click → TE path only when `technical_score`.

## 6. AI mix profiles (per entry)

| Profile | Official panel total | Admin UI |
| --- | --- | --- |
| **A — Compare only** (contest default) | Humans only | Side-by-side AI / shadow vs panel |
| **B — Gap-fill** | Panel FE; AI fills blank categories only | Show which categories were AI-filled |
| **C — Equal vote** | AI FE acts as one virtual judge in aggregation | Label “AI (virtual judge)” in admin table |

Training entries may use B/C for experiments. Contests default to **A**.

Shadow analyses remain **non-official** comparison peers; they never become the entry’s sole score without an explicit profile that includes them.

**Product principle:** Final arithmetic still goes through `DeterministicScoringEngine` (or panel FE aggregation feeding it). No opaque learned final score.

## 7. Panel aggregation (5–10 judges)

Apply **per FE category** (not only final total).

| `aggregation_mode` | Rule |
| --- | --- |
| `simple_mean` | Mean of included, non-shadow submitted values |
| `trim_1` | Drop 1 high + 1 low, mean of rest (requires ≥3 values) |
| `trim_2` | Drop 2 high + 2 low, mean of rest (requires ≥5 values) |
| `auto` (default) | n &lt; 5 → simple mean; 5–6 → `trim_1`; 7–10 → `trim_2` |

Excluded judges (`include_in_results=false` or `is_shadow`) never enter the mean.

Warnings when category range ≥ 3.0 points (existing `_DISAGREEMENT_THRESHOLD` in `scoring/judges.py`). Prefer extending that module with trim modes rather than duplicating logic in the API.

## 8. Invite, Share, QR, expiry

### 8.1 Lifecycle

1. Admin adds judge name → server generates raw token, stores hash, sets `token_expires_at = now + 48h`.
2. Dashboard shows **Share** (copy URL + short message) and **QR** (modal encoding the same URL).
3. Judge opens link before expiry → FE UI for each video.
4. Expired / revoked → 401/410 with “Ask admin for a new link.”
5. Admin **Rotate** → new token, new 48h window; old token invalid immediately.
6. Admin **Revoke** → token unusable; assignment retained for audit.

### 8.2 Security

- Rate-limit token resolution.
- HTTPS only.
- Do not log raw tokens.
- QR is convenient but forwardable — rotate if leaked.
- Judges must not receive panel or peer data from any API used by the judge surface.

## 9. UX flows

### 9.1 Admin

1. Create entry (title, mode, AI profile, aggregation, ruleset).
2. Attach videos (orderable); optionally link official/shadow analyses per video.
3. Add judges (name) → Share / QR / Rotate / Revoke.
4. Monitor progress (pending / draft / submitted per judge × video).
5. View results: per-judge FE, panel aggregate, AI compare (profile-aware).
6. Lock entry → freeze edits; allow export / calibration later.

### 9.2 Judge

1. Open private link (mobile-first).
2. See entry title + video list only (no other judges).
3. Per video: player + FE form; Save draft / Submit (confirm).
4. After submit for a video: read-only for that video.
5. No panel average, no peer scores, no AI totals unless a future “assist” flag is added (default off).

Reuse from yoyo-pwa where useful: draft vs submit confirm, large touch targets, progress indicators. Video player comes from YoYoVision patterns, not PWA.

## 10. API sketch (v1)

Admin (authenticated admin role):

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/judging-entries` | Create entry |
| `GET` | `/judging-entries` | List |
| `GET` | `/judging-entries/{id}` | Detail + videos + judge progress |
| `PATCH` | `/judging-entries/{id}` | Update config / lock |
| `POST` | `/judging-entries/{id}/videos` | Attach videos |
| `POST` | `/judging-entries/{id}/judges` | Add judge → returns **raw token once** |
| `POST` | `/judging-entries/{id}/judges/{jid}/rotate` | New token + 48h |
| `POST` | `/judging-entries/{id}/judges/{jid}/revoke` | Revoke |
| `GET` | `/judging-entries/{id}/results` | Aggregated + per-judge (admin only) |

Judge (token auth, e.g. `Authorization: Bearer <token>` or path token):

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/judge-access/{token}` | Entry + videos metadata (no peers) |
| `GET` | `/judge-access/{token}/videos/{entry_video_id}/stream` | Authorized stream |
| `PUT` | `/judge-access/{token}/videos/{entry_video_id}/fe` | Upsert draft FE |
| `POST` | `/judge-access/{token}/videos/{entry_video_id}/submit` | Submit FE |

All judge endpoints must filter to the assignment bound to that token.

## 11. Frontend surfaces

| Route | Audience |
| --- | --- |
| `/admin/judging-entries` | List / create |
| `/admin/judging-entries/[id]` | Videos, judges (Share/QR), results |
| `/judge/[token]` | Judge shell: video list → FE |

QR: client-side encode of invite URL (e.g. `qrcode` library) in admin modal; no need to persist image.

## 12. Auth / roles

Extend users with a role (minimum): `user` \| `admin`.  
v1 admin gate: `role == admin` (single-org is enough).  
Judge routes do not use user JWT.

Migration path: existing owner accounts remain `user`; bootstrap one admin for the operator.

## 13. Implementation phases

### Phase A — Design (this doc) ✅

### Phase B — Foundation (branch)

1. Alembic: tables in §5.
2. Admin role + guards.
3. Entry + videos CRUD.
4. JudgeAssignment + token issue / rotate / revoke (48h).

### Phase C — Judge FE

1. Token-auth judge API + video stream ACL.
2. Judge UI (FE form, draft/submit, isolation tests).
3. Admin Share + QR modal.

### Phase D — Aggregation + AI profiles

1. Extend `aggregate_judge_scores` with trim modes / `auto`.
2. Results endpoint + admin results UI.
3. Wire AI profiles A/B/C using attached analyses + existing FE estimators for B.

### Phase E — Hardening

1. Expiry/revoke tests, rate limits, isolation integration tests.
2. Docs: CreatorManual section + link from README.
3. Optional: export panel FE into corpus / calibration CLI.

### Phase F — v2 (later)

Clicker, click modes, event–click matching UI, richer calibration dashboard.

## 14. Testing plan

- Unit: trim aggregation edge cases (n=2,5,7,10); expired token; revoked token.
- API: judge cannot read another assignment’s FE; admin can.
- UI: QR encodes exact invite URL; Share copies same URL.
- Regression: owner Review + shadow flow unchanged.

## 15. Open items deferred

- Email invite delivery.
- Device binding / single-session enforcement.
- Multi-org / events / divisions (PWA-scale); v1 is flat entries.
- Public results page for spectators.

## 16. Success criteria (v1 merge)

- Admin can create a multi-video entry, add ≥2 name-only judges, share via link + QR.
- Each judge completes FE in isolation within 48h token window.
- Admin sees panel aggregate under `auto` trim rules and can switch AI profile A/B/C.
- Owner Review path unaffected.
- Feature merged from `feature/multi-judge-entries` → `main` after review.
