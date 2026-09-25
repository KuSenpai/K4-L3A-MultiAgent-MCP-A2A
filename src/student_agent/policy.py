"""Policy agent: deterministic rules over evidence-backed findings.

RULES_VERSION identifies the rule/rounding/confidence set used in a run. The rules
only read findings that carry evidence refs; claim topics are hypotheses to test.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from .a2a import (
    POLICY_AGENT,
    AgentContext,
    AgentResult,
    AgentTask,
    DataConflict,
    Finding,
    PolicyDecision,
    PolicyTaskPayload,
    SupportLink,
)
from .agents import money
from .ledger import EvidenceRecord
from .mcp_gateway import GatewayError, GatewayFatalError

RULES_VERSION = "l3a-rules-v1"
CENT = Decimal("0.01")

CANCELED_STATUSES = {"canceled", "cancelled"}
UNAVAILABLE_STATUSES = {"unavailable"}
REFUND_DONE = {"completed", "succeeded", "success", "refunded", "processed", "done"}
REFUND_PENDING = {"pending", "requested", "processing", "in_progress", "created", "initiated"}
REFUND_FAILED = {"failed", "rejected", "declined", "error", "reversed"}
CHARGE_EVENTS = {"captured", "charged", "charge", "capture", "settled", "paid", "payment_captured"}


def _round(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


class PolicyAgent:
    actor = POLICY_AGENT

    async def handle(self, task: AgentTask, context: AgentContext) -> AgentResult:
        payload = task.payload
        assert isinstance(payload, PolicyTaskPayload)
        policy_record = await self._policy(payload, context)
        decision, decision_code = decide(payload, policy_record)
        if (
            context.gateway is not None
            and policy_record is not None
            and not context.repair_reason_codes
            and policy_record.evidence_ref in decision.evidence_refs
        ):
            context.gateway.consume(self.actor, [policy_record])
        context.trace.emit(
            event_type="policy_decided",
            actor=self.actor,
            decision_code=decision_code,
            evidence_refs=list(decision.evidence_refs[:20]) or None,
            attributes={
                "task_id": task.task_id,
                "attempt": task.attempt,
                "rules_version": RULES_VERSION,
                "primary_issue": decision.assessment["primary_issue"],
                "case_status": decision.assessment["case_status"],
            },
        )
        return AgentResult(
            task.local_run_id, task.case_id, task.task_id, self.actor, "completed", decision
        )

    async def _policy(
        self, payload: PolicyTaskPayload, context: AgentContext
    ) -> EvidenceRecord | None:
        if not payload.policy_version or context.gateway is None:
            return None
        try:
            return await context.gateway.fetch(
                self.actor,
                "get_policy",
                deadline=context.deadline,
                policy_version=payload.policy_version,
            )
        except GatewayFatalError:
            raise
        except GatewayError:
            return None


def _by_code(findings: tuple[Finding, ...], code: str) -> list[Finding]:
    return [f for f in findings if f.finding_code == code]


def decide(payload: PolicyTaskPayload, policy: EvidenceRecord | None) -> tuple[PolicyDecision, str]:
    findings = payload.findings
    status_f = _by_code(findings, "ORDER_STATUS")
    items_f = _by_code(findings, "ORDER_ITEMS")
    pay_f = _by_code(findings, "PAYMENTS")
    events_f = _by_code(findings, "PAYMENT_EVENTS")
    refund_f = _by_code(findings, "REFUND_EVENTS")
    ship_f = _by_code(findings, "SHIPMENT_SUMMARY")

    order_status = status_f[0].value if status_f else None
    paid = money(pay_f[0].value.get("total_paid")) if pay_f else None
    order_total = money(items_f[0].value.get("order_total")) if items_f else None
    refunds = refund_f[0].value if refund_f else None
    shipment = ship_f[0].value if ship_f else None
    item_lines = items_f[0].value.get("lines", []) if items_f else []
    sellers = sorted({line["seller_id"] for line in item_lines if line.get("seller_id")})
    order_id = next(iter(payload.affected_entities.order_ids), None)

    refunded = Decimal("0")
    pending = Decimal("0")
    failed = Decimal("0")
    refund_states: set[str] = set()
    for event in refunds or []:
        state = event.get("status") or ""
        amount = money(event.get("amount")) or Decimal("0")
        refund_states.add(state)
        if state in REFUND_DONE:
            refunded += amount
        elif state in REFUND_PENDING:
            pending += amount
        elif state in REFUND_FAILED:
            failed += amount

    charges: list[tuple[str, Decimal]] = []
    for event in events_f[0].value if events_f else []:
        if event.get("event") in CHARGE_EVENTS and money(event.get("amount")):
            charges.append((str(event.get("payment_reference") or ""), money(event["amount"])))
    duplicate_amount = Decimal("0")
    seen: dict[Decimal, int] = {}
    for _, amount in charges:
        seen[amount] = seen.get(amount, 0) + 1
    for amount, count in seen.items():
        if count > 1:
            duplicate_amount += amount * (count - 1)
    if not duplicate_amount and pay_f:
        rows = pay_f[0].value.get("rows", [])
        statuses = {r.get("status") for r in rows}
        values = [money(r.get("value")) for r in rows]
        if len(rows) > 1 and "duplicate" in statuses:
            duplicate_amount = sum(
                (
                    v
                    for r, v in zip(rows, values, strict=False)
                    if r.get("status") == "duplicate" and v
                ),
                Decimal("0"),
            )

    used: list[Finding] = []
    issue = "insufficient_evidence"
    status = "needs_investigation"
    causes: list[str] = []
    parties: list[dict[str, Any]] = []
    refund_lines: list[dict[str, Any]] = []
    actions: list[str] = []
    confidence = 0.55
    decision_code = "INSUFFICIENT_EVIDENCE"
    remaining = None
    if paid is not None:
        remaining = max(Decimal("0"), paid - refunded - pending)

    if order_status in CANCELED_STATUSES | UNAVAILABLE_STATUSES and paid and paid > 0:
        issue = (
            "canceled_order_paid" if order_status in CANCELED_STATUSES else "unavailable_order_paid"
        )
        used += status_f + pay_f + refund_f
        if pending > 0 and remaining == 0:
            issue, status = "refund_pending", "needs_investigation"
            causes, actions = ["REFUND_IN_PROGRESS"], ["monitor_pending_refund"]
            parties = [{"party_type": "payment_provider", "party_id": None}]
        elif remaining and remaining > 0:
            status = "action_required"
            causes = [
                "ORDER_CANCELED_AFTER_PAYMENT"
                if issue == "canceled_order_paid"
                else "ITEM_UNAVAILABLE_AFTER_PAYMENT"
            ]
            parties = (
                [{"party_type": "seller", "party_id": s} for s in sellers]
                if issue == "unavailable_order_paid" and sellers
                else [{"party_type": "platform", "party_id": None}]
            )
            refund_lines = [
                {
                    "reason_code": issue.upper(),
                    "amount_brl": _round(remaining),
                    "entity_id": order_id,
                }
            ]
            actions = ["issue_full_refund" if refunded == 0 else "refund_remaining_balance"]
        else:
            status = "no_action"
            causes = ["REFUND_ALREADY_COMPLETED"]
            parties = [{"party_type": "platform", "party_id": None}]
        confidence = 0.85
        decision_code = "REFUND_FOR_UNFULFILLED_ORDER"
    elif refund_states & REFUND_FAILED and failed > 0:
        issue, status = "refund_failed", "action_required"
        used += refund_f + pay_f
        causes = ["REFUND_FAILED"]
        parties = [{"party_type": "payment_provider", "party_id": None}]
        refund_lines = [
            {"reason_code": "REFUND_RETRY", "amount_brl": _round(failed), "entity_id": order_id}
        ]
        actions = ["retry_failed_refund"]
        confidence = 0.8
        decision_code = "REFUND_FAILED"
    elif refund_states & REFUND_PENDING and pending > 0:
        issue, status = "refund_pending", "needs_investigation"
        used += refund_f + pay_f
        causes = ["REFUND_IN_PROGRESS"]
        parties = [{"party_type": "payment_provider", "party_id": None}]
        actions = ["monitor_pending_refund"]
        confidence = 0.8
        decision_code = "REFUND_PENDING"
    elif duplicate_amount > 0:
        issue, status = "duplicate_charge", "action_required"
        used += pay_f + events_f + refund_f
        causes = ["DUPLICATE_CAPTURE"]
        parties = [{"party_type": "payment_provider", "party_id": None}]
        refund_lines = [
            {
                "reason_code": "DUPLICATE_CHARGE",
                "amount_brl": _round(duplicate_amount),
                "entity_id": order_id,
            }
        ]
        actions = ["refund_duplicate_charge"]
        confidence = 0.8
        decision_code = "DUPLICATE_CHARGE"
    elif paid is not None and order_total is not None and abs(paid - order_total) > CENT:
        issue = "payment_mismatch"
        used += pay_f + items_f
        causes = ["PAYMENT_AMOUNT_MISMATCH"]
        parties = [{"party_type": "payment_provider", "party_id": None}]
        if paid > order_total:
            status = "action_required"
            refund_lines = [
                {
                    "reason_code": "OVERCHARGE",
                    "amount_brl": _round(paid - order_total),
                    "entity_id": order_id,
                }
            ]
            actions = ["refund_overcharge"]
        else:
            status = "needs_investigation"
            actions = ["review_payment_records"]
        confidence = 0.7
        decision_code = "PAYMENT_MISMATCH"
    elif shipment and shipment.get("delivered_late"):
        used += ship_f + items_f
        if shipment.get("seller_handoff_late"):
            issue = "late_delivery_seller"
            causes = ["SELLER_LATE_HANDOFF"]
            parties = [{"party_type": "seller", "party_id": s} for s in sellers] or [
                {"party_type": "seller", "party_id": None}
            ]
        else:
            issue = "late_delivery_logistics"
            causes = ["CARRIER_TRANSIT_DELAY"]
            parties = [{"party_type": "logistics_provider", "party_id": None}]
        status = "action_required"
        actions = ["compensate_late_delivery"]
        confidence = 0.75
        decision_code = "LATE_DELIVERY"
    elif (
        pay_f
        and len(pay_f[0].value.get("rows", [])) > 1
        and paid is not None
        and (order_total is None or abs(paid - order_total) <= CENT)
    ):
        issue, status = "valid_split_payment", "no_action"
        used += pay_f + items_f
        causes = ["SPLIT_PAYMENT_MATCHES_ORDER"]
        confidence = 0.75
        decision_code = "VALID_SPLIT_PAYMENT"
    elif status_f and (pay_f or ship_f):
        issue, status = "unsupported_claim", "no_action"
        used += status_f + pay_f + ship_f + refund_f
        causes = ["NO_DISCREPANCY_FOUND"]
        confidence = 0.6
        decision_code = "CLAIM_NOT_SUPPORTED"
    else:
        used += status_f + pay_f
        actions = ["request_manual_review"]
        confidence = 0.6 if payload.coverage_gaps else 0.5

    refs = list(dict.fromkeys(ref for f in used for ref in f.evidence_refs))
    if policy is not None and refs:
        refs.append(policy.evidence_ref)
    total_refund = sum((line["amount_brl"] for line in refund_lines), Decimal("0"))
    if payload.conflicts:
        confidence = min(confidence, 0.6)

    claim_assessments = []
    for claim in payload.claims:
        verdict, claim_conf = _claim_verdict(claim.text, issue, total_refund, paid, bool(refs))
        claim_assessments.append(
            {
                "claim_id": claim.claim_id[:64],
                "verdict": verdict,
                "confidence": claim_conf,
                "evidence_refs": refs[:30] if verdict != "insufficient_evidence" else [],
            }
        )

    decision = PolicyDecision(
        assessment={"primary_issue": issue, "case_status": status, "confidence": confidence},
        root_cause_analysis={
            "ranked_causes": [{"cause_code": c, "rank": i} for i, c in enumerate(causes, start=1)],
            "responsible_parties": _unique_parties(parties),
        },
        financial_resolution={
            "currency": "BRL",
            "recommended_refund_brl": _round(total_refund),
            "refund_lines": refund_lines,
        },
        resolution_actions=tuple(actions),
        data_conflicts=tuple(
            DataConflict(c.field, c.sources, c.selected_source, c.resolution_code)
            for c in payload.conflicts[:5]
        ),
        evidence_refs=tuple(refs[:30]),
        support_links=(
            SupportLink(
                "/assessment/primary_issue", tuple(f.finding_id for f in used), tuple(refs[:30])
            ),
        ),
        claim_assessments=tuple(claim_assessments),
    )
    return decision, decision_code


def _claim_verdict(
    topic: str, issue: str, refund: Decimal, paid: Decimal | None, has_evidence: bool
) -> tuple[str, float]:
    if not has_evidence or issue == "insufficient_evidence":
        return "insufficient_evidence", 0.6
    if topic == "requested_full_refund":
        if refund > 0 and paid is not None and refund >= paid:
            return "supported", 0.8
        if refund > 0:
            return "partially_supported", 0.7
        return "unsupported", 0.7
    if topic == issue:
        return "supported", 0.8
    return "unsupported", 0.7


def _unique_parties(parties: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, result = set(), []
    for party in parties:
        key = (party["party_type"], party["party_id"])
        if key not in seen:
            seen.add(key)
            result.append(party)
    return result[:5]
