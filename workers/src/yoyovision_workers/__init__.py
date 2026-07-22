"""YoYoVision Celery workers: independently deployable pipeline execution
service. Depends only on `yoyovision-ml` plus infrastructure libraries
(Celery, SQLAlchemy, Redis) -- never on `yoyovision-api` -- and shares state
with the API exclusively through the Postgres schema (mirrored in
`yoyovision_workers.schema`) and the Redis broker task-name contract
(`yoyovision_workers.tasks`)."""

from __future__ import annotations
