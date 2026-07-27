from __future__ import annotations

from typing import Sequence

from ..domain.models import ContextFrame


ALL_LABELS = (
    "PREFIX", "PERSON", "GENDER", "AGE", "BIRTHDATE", "PHONE", "EMAIL", "LOCATION",
    "ADDRESS", "ZIP_CODE", "COORDINATE", "USERNAME", "ACCOUNT_ID", "TICKET_ID",
    "PASSWORD", "PIN", "API_KEY", "URL", "IP", "BANK_ACCOUNT", "MONEY", "CARD_ISSUER",
    "CARD_NUMBER", "CVV", "IBAN", "SWIFT", "WALLET", "JOB_TITLE", "ORGANIZATION",
    "EMPLOYEE_ID", "NATIONAL_ID", "PASSPORT", "LICENSE", "PLATE", "TIN", "DATE", "TIME",
    "MARITAL", "RELIGION", "ETHNICITY", "TRADE_UNION", "NATIONALITY", "INSURANCE_ID",
    "MEDICAL_INFO",
)


def _frame(
    frame_id: str,
    domain: str,
    document_type: str,
    tone: str,
    labels: Sequence[str],
    max_sentences: int = 3,
) -> ContextFrame:
    return ContextFrame(
        frame_id=frame_id,
        domain=domain,
        document_type=document_type,
        tone=tone,
        max_sentences=max_sentences,
        supported_labels=list(labels),
    )


CONTEXT_FRAMES: tuple[ContextFrame, ...] = (
    _frame("technical_support_visit", "technical_support", "support_ticket", "formal",
           ("ADDRESS", "DATE", "TIME", "EMAIL", "PHONE", "PERSON", "IP", "URL", "TICKET_ID")),
    _frame("delivery_request", "delivery", "delivery_note", "neutral",
           ("ADDRESS", "LOCATION", "DATE", "TIME", "PHONE", "PERSON", "MONEY")),
    _frame("employee_onboarding", "human_resources", "internal_email", "formal",
           ("PERSON", "DATE", "EMAIL", "PHONE", "ADDRESS", "EMPLOYEE_ID", "NATIONAL_ID",
            "JOB_TITLE", "ORGANIZATION", "BANK_ACCOUNT")),
    _frame("travel_booking", "travel", "booking_request", "neutral",
           ("PERSON", "PASSPORT", "NATIONALITY", "DATE", "TIME", "MONEY", "PHONE", "EMAIL",
            "LOCATION", "ADDRESS")),
    _frame("payment_dispute", "finance", "transaction_note", "formal",
           ("PERSON", "DATE", "TIME", "MONEY", "BANK_ACCOUNT", "CARD_ISSUER", "CARD_NUMBER",
            "CVV", "IBAN", "SWIFT", "WALLET", "EMAIL", "PHONE", "ORGANIZATION")),
    _frame("medical_follow_up", "healthcare", "follow_up_note", "professional",
           ("PERSON", "MEDICAL_INFO", "INSURANCE_ID", "DATE", "TIME", "PHONE", "EMAIL", "ADDRESS")),
    _frame("account_security_review", "cybersecurity", "security_alert", "urgent",
           ("PERSON", "USERNAME", "ACCOUNT_ID", "PASSWORD", "PIN", "API_KEY", "URL", "IP",
            "DATE", "TIME", "EMAIL", "PHONE", "ORGANIZATION", "EMPLOYEE_ID", "TICKET_ID")),
    _frame("vehicle_service_record", "transport", "service_record", "neutral",
           ("PERSON", "PLATE", "LICENSE", "DATE", "TIME", "MONEY", "ADDRESS", "PHONE")),
    _frame("customer_profile_update", "customer_service", "profile_change_request", "conversational",
           ("PREFIX", "PERSON", "GENDER", "AGE", "BIRTHDATE", "PHONE", "EMAIL", "LOCATION",
            "ADDRESS", "ZIP_CODE", "USERNAME", "ACCOUNT_ID", "DATE", "TIME")),
    _frame("controlled_administrative_record", "administration", "controlled_test_record", "formal",
           ALL_LABELS, max_sentences=4),
    _frame("generic_internal_note", "customer_support", "internal_note", "neutral",
           ALL_LABELS, max_sentences=2),
)


def compatible_context_frames(
    focus_labels: Sequence[str],
    excluded_frame_ids: Sequence[str] = (),
) -> list[ContextFrame]:
    required = set(focus_labels)
    excluded = set(excluded_frame_ids)
    return [
        frame
        for frame in CONTEXT_FRAMES
        if frame.frame_id not in excluded and required.issubset(set(frame.supported_labels))
    ]
