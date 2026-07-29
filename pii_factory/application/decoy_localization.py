from __future__ import annotations

from ..domain.models import DecoySeed


_REQUIRED_CUES: dict[str, dict[str, tuple[str, str]]] = {
    "en": {
        "analytics_code": ("reporting metric", "analytics dashboard"),
        "business_identifier": ("inventory code", "business catalog"),
        "catalog_code": ("catalog code", "classification table"),
        "configuration_code": ("configuration key", "test configuration"),
        "document_section": ("document section", "policy section"),
        "invalid_value": ("test value", "error-handling test"),
        "module_identifier": ("module identifier", "system module"),
        "schema_field": ("schema field", "data field"),
        "template_identifier": ("template identifier", "document template"),
        "test_identifier": ("test fixture", "test suite"),
    },
    "de": {
        "analytics_code": ("Berichtskennzahl", "Analyse-Dashboard"),
        "business_identifier": ("Inventarcode", "Geschäftskatalog"),
        "catalog_code": ("Katalogcode", "Klassifikationstabelle"),
        "configuration_code": (
            "Konfigurationsschlüssel",
            "Testkonfiguration",
        ),
        "document_section": ("Dokumentabschnitt", "Richtlinienabschnitt"),
        "invalid_value": ("Testwert", "Fehlerbehandlungstest"),
        "module_identifier": ("Modulkennung", "Systemmodul"),
        "schema_field": ("Schemafeld", "Datenfeld"),
        "template_identifier": ("Vorlagenkennung", "Dokumentvorlage"),
        "test_identifier": ("Testdatensatz", "Testsuite"),
    },
}

_DEFAULT_REQUIRED_CUES = {
    "en": ("internal system code", "test data"),
    "de": ("interner Systemcode", "Testdaten"),
}

_GERMAN_LABEL_CUES = {
    "PERSON": "Person",
    "PHONE": "Telefonnummer",
    "ADDRESS": "Adresse",
    "LOCATION": "Ort",
    "BIRTHDATE": "Geburtsdatum",
    "PASSWORD": "Passwort",
    "BANK_ACCOUNT": "Bankkonto",
    "MONEY": "Geldbetrag",
    "JOB_TITLE": "Berufsbezeichnung",
    "ORGANIZATION": "Organisation",
    "EMPLOYEE_ID": "Mitarbeiternummer",
    "NATIONAL_ID": "Personalausweisnummer",
    "LICENSE": "Führerscheinnummer",
    "PLATE": "Kennzeichen",
    "DATE": "Datum",
    "TIME": "Uhrzeit",
    "MARITAL": "Familienstand",
    "RELIGION": "Religion",
    "NATIONALITY": "Staatsangehörigkeit",
    "MEDICAL_INFO": "Gesundheitsinformation",
}


def localize_decoy(decoy: DecoySeed, language: str) -> DecoySeed:
    """Return language-matched hard-negative cues without changing its value."""

    code = _language_code(language)
    if code == "vi":
        return decoy
    language_cues = _REQUIRED_CUES.get(code)
    if language_cues is None:
        return decoy
    required = language_cues.get(
        decoy.family,
        _DEFAULT_REQUIRED_CUES[code],
    )
    label_cue = decoy.target_label.casefold().replace("_", " ")
    if code == "de":
        label_cue = _GERMAN_LABEL_CUES.get(
            decoy.target_label,
            label_cue,
        )
    forbidden = [label_cue] if decoy.forbidden_context_cues else []
    return decoy.copy(update={
        "required_context_cues": list(required),
        "forbidden_context_cues": forbidden,
    })


def _language_code(language: str) -> str:
    normalized = str(language).strip().casefold().replace("_", "-")
    aliases = {
        "vietnamese": "vi",
        "english": "en",
        "german": "de",
    }
    return aliases.get(normalized, normalized.split("-", 1)[0])
