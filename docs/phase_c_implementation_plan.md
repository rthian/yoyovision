# Phase C — Judge FE (Implementation Plan)

See Grok 4.5 planning output in chat / agent transcript. Build brief for Composer 2.5.

## C1 API — done
- `/judge-access/{token}` routes
- Integration tests in `test_judge_access_api.py`

## C2 Judge FE
- `/judge/[token]` page
- `JudgeFreestyleForm`, hooks, token api-client

## C3 Admin Share/QR
- `/admin/judging-entries` list + detail
- `JudgeInviteShareModal` with QR
