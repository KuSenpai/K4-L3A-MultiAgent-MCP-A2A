"""Domain specialists: order/item, payment/refund and shipment agents.

Each agent only calls the MCP tools of its own domain (``ledger.TOOL_MAPPING``), turns
validated evidence into findings that carry the supporting ``evidence_ref`` and emits
``tool_result_consumed`` for evidence it actually used. A failed or missing tool call is
a coverage gap, never negative evidence.

MCP ``data`` payloads are read defensively (Olist-style field names, several aliases)
because their business structure is not part of the public contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from .a2a import (
    ORDER_AGENT,
    PAYMENT_AGENT,
    SHIPMENT_AGENT,
    AgentContext,
    AgentResult,
    AgentTask,
    DataConflict,
    DomainResult,
    DomainTaskPayload,
    EntityIds,
    Finding,
    HandoffRequest,
)
from .ledger import EvidenceRecord, thaw
from .mcp_gateway import GatewayError, GatewayFatalError

LIST_KEYS = (
    "rows",
    "items",
    "order_items",
    "payments",
    "events",
    "timeline",
    "refunds",
    "sellers",
    "products",
    "records",
    "data",
    "history",
    "orders",
    "shipments",
)


# ----------------------------------------------------------------- data helpers
def rows(data: Any) -> list[dict[str, Any]]:
    """Return the list of row objects inside an evidence ``data`` payload."""
    value = thaw(data)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in LIST_KEYS:
            inner = value.get(key)
            if isinstance(inner, list) and all(isinstance(row, dict) for row in inner):
                return inner
        return [value]
    return []


def nested_rows(data: Any, *keys: str) -> list[dict[str, Any]]:
    value = thaw(data)
    if isinstance(value, dict):
        for key in keys:
            inner = value.get(key)
            if isinstance(inner, list):
                return [row for row in inner if isinstance(row, dict)]
    return []


def get(row: Mapping[str, Any] | None, *names: str) -> Any:
    if not row:
        return None
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def money(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def when(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    for candidate in (text, text.replace(" ", "T")):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed.replace(tzinfo=None)
        except ValueError:
            continue
    return None


def lower(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


class _Specialist:
    actor: str = ""

    def __init__(self) -> None:
        self._counter = 0

    async def _fetch(
        self, context: AgentContext, gaps: list[str], tool: str, **arguments: str
    ) -> EvidenceRecord | None:
        evidence = context.gateway
        if evidence is None:
            gaps.append(f"{tool}:NO_GATEWAY")
            return None
        try:
            return await evidence.fetch(self.actor, tool, deadline=context.deadline, **arguments)
        except GatewayFatalError:
            raise
        except GatewayError as exc:
            gaps.append(f"{tool}:{exc.code}")
            return None

    def _finding(
        self,
        task: AgentTask,
        code: str,
        value: Any,
        records: Iterable[EvidenceRecord],
        entity_ids: EntityIds | None = None,
    ) -> Finding:
        self._counter += 1
        payload = task.payload
        claim_ids = tuple(c.claim_id for c in getattr(payload, "claims", ()))
        refs = tuple(dict.fromkeys(r.evidence_ref for r in records))
        return Finding(
            finding_id=f"{task.task_id}-{self.actor.split('-')[0].upper()}-{self._counter:02d}",
            finding_code=code,
            value=value,
            entity_ids=entity_ids or EntityIds(),
            claim_ids=claim_ids,
            evidence_refs=refs,
        )

    def _result(
        self,
        task: AgentTask,
        context: AgentContext,
        findings: list[Finding],
        entities: EntityIds,
        gaps: list[str],
        used: list[EvidenceRecord],
        conflicts: tuple[DataConflict, ...] = (),
        handoffs: tuple[HandoffRequest, ...] = (),
    ) -> AgentResult:
        if used and context.gateway is not None and not context.repair_reason_codes:
            context.gateway.consume(self.actor, used)
        refs = tuple(dict.fromkeys(ref for f in findings for ref in f.evidence_refs))
        if handoffs:
            status = "needs_handoff"
        elif findings:
            status = "completed"
        else:
            status = "insufficient_evidence"
        payload = DomainResult(
            findings=tuple(findings),
            affected_entities=entities,
            evidence_refs=refs,
            conflicts=conflicts,
            warnings=tuple(sorted(set(gaps))),
            handoff_requests=handoffs,
        )
        return AgentResult(
            task.local_run_id, task.case_id, task.task_id, self.actor, status, payload
        )

    @staticmethod
    def _order_ids(task: AgentTask) -> tuple[str, ...]:
        payload = task.payload
        assert isinstance(payload, DomainTaskPayload)
        return payload.entity_ids.order_ids


# ---------------------------------------------------------------- order agent
class OrderAgent(_Specialist):
    actor = ORDER_AGENT

    async def handle(self, task: AgentTask, context: AgentContext) -> AgentResult:
        self._counter = 0
        findings: list[Finding] = []
        gaps: list[str] = []
        used: list[EvidenceRecord] = []
        entities = EntityIds()
        for order_id in self._order_ids(task):
            order = await self._fetch(context, gaps, "get_order", order_id=order_id)
            if order is not None:
                row = rows(order.data)[0] if rows(order.data) else {}
                status = lower(get(row, "order_status", "status"))
                if status or row:
                    used.append(order)
                    entities = entities.union(EntityIds(order_ids=(order_id,)))
                    findings.append(
                        self._finding(
                            task,
                            "ORDER_STATUS",
                            status or None,
                            [order],
                            EntityIds(order_ids=(order_id,)),
                        )
                    )
                    dates = {
                        key: get(row, key)
                        for key in (
                            "order_purchase_timestamp",
                            "order_approved_at",
                            "order_delivered_carrier_date",
                            "order_delivered_customer_date",
                            "order_estimated_delivery_date",
                        )
                        if get(row, key)
                    }
                    if dates:
                        findings.append(
                            self._finding(
                                task,
                                "ORDER_TIMELINE",
                                dates,
                                [order],
                                EntityIds(order_ids=(order_id,)),
                            )
                        )
                    customer = get(row, "customer_unique_id", "customer_id")
                    if customer:
                        findings.append(
                            self._finding(task, "ORDER_CUSTOMER", str(customer), [order])
                        )
            items = await self._fetch(context, gaps, "get_order_items", order_id=order_id)
            if items is not None:
                item_rows = rows(items.data)
                total_price = Decimal("0")
                total_freight = Decimal("0")
                item_ids: list[str] = []
                seller_ids: list[str] = []
                lines = []
                for row in item_rows:
                    price = money(get(row, "price", "item_price")) or Decimal("0")
                    freight = money(get(row, "freight_value", "freight")) or Decimal("0")
                    total_price += price
                    total_freight += freight
                    item_id = get(row, "item_id", "order_item_key", "order_item_uid")
                    if item_id is None and get(row, "order_item_id") is not None:
                        item_id = get(row, "order_item_id")
                    seller = get(row, "seller_id")
                    if item_id is not None:
                        item_ids.append(str(item_id))
                    if seller:
                        seller_ids.append(str(seller))
                    lines.append(
                        {
                            "item_id": None if item_id is None else str(item_id),
                            "seller_id": seller,
                            "price": str(price),
                            "freight": str(freight),
                            "status": lower(get(row, "item_status", "status", "availability"))
                            or None,
                            "shipping_limit_date": get(row, "shipping_limit_date"),
                        }
                    )
                if item_rows:
                    used.append(items)
                    item_entities = EntityIds(
                        order_ids=(order_id,),
                        item_ids=tuple(item_ids),
                        seller_ids=tuple(seller_ids),
                    )
                    entities = entities.union(item_entities)
                    findings.append(
                        self._finding(
                            task,
                            "ORDER_ITEMS",
                            {
                                "lines": lines,
                                "items_total": str(total_price),
                                "freight_total": str(total_freight),
                                "order_total": str(total_price + total_freight),
                            },
                            [items],
                            item_entities,
                        )
                    )
        return self._result(task, context, findings, entities, gaps, used)


# -------------------------------------------------------------- payment agent
class PaymentAgent(_Specialist):
    actor = PAYMENT_AGENT

    async def handle(self, task: AgentTask, context: AgentContext) -> AgentResult:
        self._counter = 0
        findings: list[Finding] = []
        gaps: list[str] = []
        used: list[EvidenceRecord] = []
        entities = EntityIds()
        for order_id in self._order_ids(task):
            payments = await self._fetch(context, gaps, "get_order_payments", order_id=order_id)
            timeline = await self._fetch(context, gaps, "get_payment_timeline", order_id=order_id)
            refunds = await self._fetch(context, gaps, "get_refund_timeline", order_id=order_id)

            base_rows: list[dict[str, Any]] = []
            base_record = None
            for record in (payments, timeline):
                if record is None:
                    continue
                candidate = nested_rows(record.data, "payments", "base_payments") or rows(
                    record.data
                )
                candidate = [
                    r for r in candidate if get(r, "payment_value", "amount", "value") is not None
                ]
                if candidate:
                    base_rows, base_record = candidate, record
                    break
            if base_record is not None:
                refs: list[str] = []
                total = Decimal("0")
                summary = []
                for row in base_rows:
                    value = money(get(row, "payment_value", "amount", "value")) or Decimal("0")
                    total += value
                    ref = get(row, "payment_reference", "payment_id", "transaction_id")
                    if ref:
                        refs.append(str(ref))
                    summary.append(
                        {
                            "payment_reference": None if ref is None else str(ref),
                            "sequential": get(row, "payment_sequential", "sequence"),
                            "type": get(row, "payment_type", "method"),
                            "installments": get(row, "payment_installments", "installments"),
                            "value": str(value),
                            "status": lower(get(row, "status", "payment_status")) or None,
                        }
                    )
                used.append(base_record)
                pay_entities = EntityIds(order_ids=(order_id,), payment_references=tuple(refs))
                entities = entities.union(pay_entities)
                findings.append(
                    self._finding(
                        task,
                        "PAYMENTS",
                        {
                            "count": len(base_rows),
                            "total_paid": str(total),
                            "rows": summary,
                        },
                        [base_record],
                        pay_entities,
                    )
                )

            if timeline is not None:
                events = nested_rows(timeline.data, "events", "lifecycle_events", "timeline")
                if not events and timeline is not base_record:
                    events = [
                        r for r in rows(timeline.data) if get(r, "event_type", "event", "type")
                    ]
                if events:
                    used.append(timeline)
                    compact = [
                        {
                            "event": lower(get(e, "event_type", "event", "type", "status")),
                            "amount": str(money(get(e, "amount", "value", "payment_value")) or ""),
                            "payment_reference": get(
                                e, "payment_reference", "payment_id", "transaction_id"
                            ),
                            "at": get(e, "occurred_at", "timestamp", "at", "created_at"),
                        }
                        for e in events
                    ]
                    findings.append(self._finding(task, "PAYMENT_EVENTS", compact, [timeline]))

            if refunds is not None:
                events = nested_rows(refunds.data, "events", "refunds", "lifecycle_events") or rows(
                    refunds.data
                )
                events = [
                    e
                    for e in events
                    if get(e, "status", "refund_status", "event_type", "event", "amount")
                ]
                used.append(refunds)
                compact = [
                    {
                        "status": lower(get(e, "status", "refund_status", "event_type", "event")),
                        "amount": str(money(get(e, "amount", "refund_amount", "value")) or ""),
                        "refund_id": get(e, "refund_id", "refund_reference", "id"),
                        "payment_reference": get(e, "payment_reference", "payment_id"),
                        "at": get(e, "occurred_at", "timestamp", "at", "created_at"),
                    }
                    for e in events
                ]
                # An empty refund timeline from a valid envelope is negative evidence.
                findings.append(
                    self._finding(
                        task, "REFUND_EVENTS", compact, [refunds], EntityIds(order_ids=(order_id,))
                    )
                )
        return self._result(task, context, findings, entities, gaps, used)


# ------------------------------------------------------------- shipment agent
class ShipmentAgent(_Specialist):
    actor = SHIPMENT_AGENT

    async def handle(self, task: AgentTask, context: AgentContext) -> AgentResult:
        self._counter = 0
        findings: list[Finding] = []
        gaps: list[str] = []
        used: list[EvidenceRecord] = []
        entities = EntityIds()
        for order_id in self._order_ids(task):
            record = await self._fetch(context, gaps, "get_shipment_summary", order_id=order_id)
            if record is None:
                continue
            data = thaw(record.data)
            base = data if isinstance(data, dict) else (rows(data)[0] if rows(data) else {})
            delivered = when(
                get(
                    base, "order_delivered_customer_date", "delivered_customer_date", "delivered_at"
                )
            )
            estimated = when(
                get(
                    base,
                    "order_estimated_delivery_date",
                    "estimated_delivery_date",
                    "promised_date",
                )
            )
            carrier = when(
                get(
                    base,
                    "order_delivered_carrier_date",
                    "delivered_carrier_date",
                    "carrier_handoff_at",
                )
            )
            limits = [
                when(get(r, "shipping_limit_date"))
                for r in nested_rows(data, "items", "seller_handoff_limits", "handoff_limits")
            ]
            limit = when(get(base, "shipping_limit_date", "seller_handoff_limit")) or max(
                (x for x in limits if x), default=None
            )
            shipment_ids = [str(x) for x in (get(base, "shipment_id", "tracking_id"),) if x]
            events = nested_rows(data, "events", "shipment_events")
            summary = {
                "delivered_customer": delivered.isoformat() if delivered else None,
                "estimated_delivery": estimated.isoformat() if estimated else None,
                "delivered_carrier": carrier.isoformat() if carrier else None,
                "seller_handoff_limit": limit.isoformat() if limit else None,
                "delivered_late": bool(
                    delivered and estimated and delivered.date() > estimated.date()
                ),
                "late_days": (delivered.date() - estimated.date()).days
                if delivered and estimated
                else None,
                "seller_handoff_late": bool(carrier and limit and carrier > limit),
                "event_count": len(events),
                "flags": {k: v for k, v in base.items() if isinstance(v, bool)}
                if isinstance(base, dict)
                else {},
            }
            used.append(record)
            ship_entities = EntityIds(order_ids=(order_id,), shipment_ids=tuple(shipment_ids))
            entities = entities.union(ship_entities)
            findings.append(
                self._finding(task, "SHIPMENT_SUMMARY", summary, [record], ship_entities)
            )
        return self._result(task, context, findings, entities, gaps, used)
