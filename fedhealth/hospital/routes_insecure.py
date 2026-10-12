"""!!! INSECURE PLAIN BASELINES !!!

/insecure/match         returns the national IDs of matching patients in the clear, so the coordinator can
                        do an ordinary join (Phase 1 baseline).
/insecure/match_linked  returns, per matching patient, its position in the PSI-linked index plus value /
                        category. Linkage is private, but the per-patient predicate results are shipped to
                        the coordinator in the clear (pseudonymised by index). It exists to measure the cost
                        of PSI on its own ("+PSI" stage of the evaluation) and to test PSI end to end.

They exist ONLY so tests and the evaluation can compare the private pipeline against "what you'd do without
privacy engineering". Mounted only when ALLOW_INSECURE_BASELINE=1 (never in the Compose defaults); both still
go through the same whitelisted LocalQuery validation as every other path.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from sqlalchemy.orm import Session

from fedhealth.common.schemas import LocalMatchResponse, LocalMatchRow, LocalQuery
from fedhealth.common.wire import LINKAGE_ID_PATTERN, LinkedMatchRequest, LinkedMatchResponse, LinkedRow
from fedhealth.hospital import security
from fedhealth.hospital.predicate_runner import run_local_query


def make_router(hospital_id: str, get_session) -> APIRouter:
    router = APIRouter(prefix="/insecure", tags=["INSECURE baseline"], dependencies=[Depends(security.control_only)])

    @router.post("/match", response_model=LocalMatchResponse)
    def match(q: LocalQuery, session: Session = Depends(get_session)) -> LocalMatchResponse:
        if q.hospital != hospital_id:
            raise HTTPException(400, f"this is hospital {hospital_id}, not {q.hospital}")
        rows = run_local_query(session, q)
        return LocalMatchResponse(
            hospital=hospital_id,
            rows=[LocalMatchRow(national_id=r.national_id, value=r.value, category=r.category) for r in rows],
        )

    @router.post("/match_linked/{lid}", response_model=LinkedMatchResponse)
    def match_linked(body: LinkedMatchRequest, request: Request, lid: Annotated[str, Path(pattern=LINKAGE_ID_PATTERN)],
                     session: Session = Depends(get_session)) -> LinkedMatchResponse:
        q = body.query
        if q.hospital != hospital_id:
            raise HTTPException(400, f"this is hospital {hospital_id}, not {q.hospital}")
        linkage = request.app.state.psi.linkages.get(lid)
        if linkage is None:
            raise HTTPException(404, "unknown or unfinished linkage")
        rows = [LinkedRow(idx=linkage.mapping[r.national_id], value=r.value, category=r.category)
                for r in run_local_query(session, q) if r.national_id in linkage.mapping]
        return LinkedMatchResponse(hospital=hospital_id, rows=rows)

    return router
