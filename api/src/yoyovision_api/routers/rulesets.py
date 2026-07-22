"""Read-only transparency endpoints exposing the versioned scoring ruleset(s).

Per product principle #8 ("Keep the rule set versioned and configurable")
and the disclaimer requirements, every ruleset a score could have been
computed against must be inspectable by users -- nothing about scoring is
opaque.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from yoyovision_ml.ruleset import Ruleset, get_ruleset_by_version, list_available_rulesets

from yoyovision_api.deps import CurrentUser

router = APIRouter(prefix="/rulesets", tags=["rulesets"])


@router.get("", response_model=list[Ruleset])
async def list_rulesets(current_user: CurrentUser) -> list[Ruleset]:
    return list_available_rulesets()


@router.get("/{version}", response_model=Ruleset)
async def get_ruleset(version: str, current_user: CurrentUser) -> Ruleset:
    ruleset = get_ruleset_by_version(version)
    if ruleset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ruleset not found.")
    return ruleset
