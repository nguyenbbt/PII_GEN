from __future__ import annotations

import random
from typing import Dict, List, Protocol, Sequence

from .context_catalog import ALL_LABELS, compatible_context_frames
from .content_vocabulary import build_content_seeds
from .decoy_localization import localize_decoy
from .value_bank import ValueBankEntityProvider
from .hard_negative_base import DecoyStrategy, digits as _digits, strategy as _strategy
from .hard_negative_variants import ADDITIONAL_HARD_NEGATIVE_STRATEGIES
from ..domain.models import (
    ContextFrame,
    DecoySeed,
    ValueBankConfig,
    GenerationTask,
    HardNegativeConfig,
    PositiveEntitySeed,
    SampleType,
    SeedPack,
    TaxonomyLabel,
)


class SeedPackFactory(Protocol):
    def build(
        self,
        task: GenerationTask,
        taxonomy: Sequence[TaxonomyLabel],
        rng: random.Random,
    ) -> SeedPack: ...


class ContextSelectionError(ValueError):
    scope = "CONTEXT"


class UnsupportedDecoyError(ValueError):
    scope = "SEEDS"


_ALL_LABELS = ALL_LABELS


SEMANTIC_ROLES: Dict[str, str] = {
    "ADDRESS": "street_address", "LOCATION": "administrative_location",
    "ZIP_CODE": "postal_code", "DATE": "appointment_date", "TIME": "appointment_time",
    "EMAIL": "contact_email", "PHONE": "contact_phone", "PERSON": "requester_name",
    "IP": "device_ip", "URL": "support_portal", "TICKET_ID": "support_ticket_id",
    "EMPLOYEE_ID": "employee_record_id", "NATIONAL_ID": "identity_document_number",
    "MONEY": "transaction_amount", "ORGANIZATION": "responsible_organization",
}


class ContextFrameSelector:
    def select(
        self,
        focus_labels: Sequence[str],
        rng: random.Random,
        *,
        decoy_semantic_types: Sequence[str] = (),
        excluded_frame_ids: Sequence[str] = (),
        preferred_frame_id: str | None = None,
    ) -> ContextFrame:
        candidates = compatible_context_frames(focus_labels, excluded_frame_ids)
        if not candidates:
            raise ContextSelectionError(f"no context frame supports labels: {sorted(set(focus_labels))}")
        if preferred_frame_id:
            preferred = next((frame for frame in candidates if frame.frame_id == preferred_frame_id), None)
            if preferred is not None:
                return preferred.copy(deep=True)
        return rng.choice(candidates).copy(deep=True)


HARD_NEGATIVE_STRATEGIES: Dict[str, tuple[DecoyStrategy, ...]] = {
    "PREFIX": _strategy(
        "prefix_as_customer_group", "PREFIX", "catalog_code", "customer_group_code",
        ("mã nhóm khách hàng", "nhóm chiến dịch"), ("danh xưng", "xưng hô"),
        lambda rng: f"MR-GROUP-{rng.randint(1, 99):02d}",
        lambda value: value.startswith("MR-GROUP-") and value[-2:].isdigit(),
    ),
    "PERSON": _strategy(
        "person_as_schema_field", "PERSON", "schema_field", "database_column_name",
        ("tên trường dữ liệu", "cột trong schema"), ("họ tên", "khách hàng tên"),
        lambda rng: f"person_name_field_v{rng.randint(2, 9)}",
        lambda value: value.startswith("person_name_field_v"),
    ),
    "GENDER": _strategy(
        "gender_as_report_group", "GENDER", "catalog_code", "anonymous_report_group",
        ("nhãn nhóm thống kê", "báo cáo tổng hợp"), ("giới tính của", "bản dạng của"),
        lambda rng: f"GENDER-GROUP-{rng.choice('ABC')}",
        lambda value: value.startswith("GENDER-GROUP-"),
    ),
    "AGE": _strategy(
        "age_as_validation_error", "AGE", "invalid_value", "validation_error_code",
        ("mã lỗi kiểm thử", "giá trị lỗi"), ("tuổi của", "năm tuổi"),
        lambda rng: f"ERR-AGE-{rng.choice(('ABOVE-MAX', 'BELOW-MIN', 'NOT-SET'))}",
        lambda value: value.startswith("ERR-AGE-") and value.rsplit("-", 1)[-1] in {"MAX", "MIN", "SET"},
    ),
    "BIRTHDATE": _strategy(
        "birthdate_as_schema_field", "BIRTHDATE", "schema_field", "database_column_name",
        ("tên cột schema", "trường dữ liệu"), ("ngày sinh", "sinh ngày"),
        lambda rng: f"birthdate_column_v{rng.randint(2, 9)}",
        lambda value: value.startswith("birthdate_column_v"),
    ),
    "PHONE": _strategy(
        "phone_as_product_code", "PHONE", "business_identifier", "product_code",
        ("mã sản phẩm", "danh mục kho"), ("gọi", "nhắn tin", "liên hệ"),
        lambda rng: f"TEL-SKU-{rng.choice('ABC')}{rng.randint(10, 99)}",
        lambda value: value.startswith("TEL-SKU-") and value[-3].isalpha() and value[-2:].isdigit(),
    ),
    "EMAIL": _strategy(
        "email_as_schema_field", "EMAIL", "schema_field", "database_column_name",
        ("trường cấu hình", "cột dữ liệu"), ("gửi thư", "hộp thư", "liên hệ qua"),
        lambda rng: f"email_field_v{rng.randint(2, 9)}",
        lambda value: value.startswith("email_field_v") and "@" not in value,
    ),
    "LOCATION": _strategy(
        "location_as_warehouse_zone", "LOCATION", "business_identifier", "warehouse_zone_code",
        ("mã khu vực kho", "phân loại hàng hóa"), ("tỉnh", "thành phố", "quốc gia"),
        lambda rng: f"ZONE-{rng.choice('ABC')}{rng.randint(1, 99):02d}",
        lambda value: value.startswith("ZONE-"),
    ),
    "ADDRESS": _strategy(
        "address_as_shelf_code", "ADDRESS", "business_identifier", "warehouse_shelf_code",
        ("vị trí kệ hàng", "hệ thống quản lý kho"), ("số nhà", "đường", "căn hộ"),
        lambda rng: f"SHELF-{rng.choice('ABC')}-{rng.randint(1, 99):02d}",
        lambda value: value.startswith("SHELF-"),
    ),
    "ZIP_CODE": _strategy(
        "zip_as_processing_lane", "ZIP_CODE", "business_identifier", "sorting_lane_code",
        ("mã tuyến xử lý", "trung tâm phân loại"), ("mã bưu chính", "bưu điện"),
        lambda rng: f"ZIP-LANE-{rng.randint(1, 99):02d}",
        lambda value: value.startswith("ZIP-LANE-"),
    ),
    "COORDINATE": _strategy(
        "coordinate_as_invalid_test_value", "COORDINATE", "invalid_value", "validation_test_value",
        ("kiểm tra lỗi tọa độ", "test validate"), ("vĩ độ", "kinh độ", "vị trí thực"),
        lambda rng: f"LAT-{rng.choice('ABC')},LON-{rng.choice('XYZ')}",
        lambda value: value.startswith("LAT-") and ",LON-" in value,
    ),
    "USERNAME": _strategy(
        "username_as_schema_field", "USERNAME", "schema_field", "database_column_name",
        ("tên trường schema", "cột đăng nhập"), ("đăng nhập bằng", "tên người dùng của"),
        lambda rng: f"username_field_v{rng.randint(2, 9)}",
        lambda value: value.startswith("username_field_v"),
    ),
    "ACCOUNT_ID": _strategy(
        "account_as_schema_field", "ACCOUNT_ID", "schema_field", "database_column_name",
        ("tên cột tài khoản", "schema dữ liệu"), ("tài khoản được cấp", "định danh tài khoản"),
        lambda rng: f"account_id_column_v{rng.randint(2, 9)}",
        lambda value: value.startswith("account_id_column_v"),
    ),
    "TICKET_ID": _strategy(
        "ticket_as_schema_field", "TICKET_ID", "schema_field", "database_column_name",
        ("tên trường ticket", "schema hỗ trợ"), ("yêu cầu hỗ trợ số", "ticket được tạo"),
        lambda rng: f"ticket_id_field_v{rng.randint(2, 9)}",
        lambda value: value.startswith("ticket_id_field_v"),
    ),
    "PASSWORD": _strategy(
        "password_as_config_flag", "PASSWORD", "configuration_code", "test_configuration_key",
        ("khóa cấu hình", "chế độ kiểm thử"), ("mật khẩu", "đăng nhập", "xác thực"),
        lambda rng: f"CFG-pass-{rng.randint(1000, 9999)}",
        lambda value: value.startswith("CFG-pass-"),
    ),
    "PIN": _strategy(
        "pin_as_product_family", "PIN", "catalog_code", "product_family_code",
        ("mã dòng sản phẩm", "linh kiện"), ("mã bí mật", "xác thực", "mở khóa"),
        lambda rng: f"PIN-PRODUCT-{rng.randint(1000, 9999)}",
        lambda value: value.startswith("PIN-PRODUCT-"),
    ),
    "API_KEY": _strategy(
        "api_key_as_schema_field", "API_KEY", "schema_field", "configuration_field_name",
        ("tên trường cấu hình", "schema tích hợp"), ("secret", "token truy cập", "xác thực API"),
        lambda rng: f"api_key_field_v{rng.randint(2, 9)}",
        lambda value: value.startswith("api_key_field_v"),
    ),
    "URL": _strategy(
        "url_as_module_name", "URL", "module_identifier", "documentation_module_name",
        ("tên module", "hệ thống build tài liệu"), ("truy cập", "đường dẫn web", "liên kết"),
        lambda rng: f"docs{rng.randint(2, 9)}.example.module",
        lambda value: value.endswith(".example.module") and "://" not in value,
    ),
    "IP": _strategy(
        "ip_as_invalid_test_value", "IP", "invalid_value", "network_validation_test",
        ("test case", "kiểm tra lỗi nhập IP"), ("máy chủ", "kết nối tới", "địa chỉ mạng"),
        lambda rng: f"999.abc.{rng.randint(1, 9)}.x",
        lambda value: value.startswith("999.abc.") and value.endswith(".x"),
    ),
    "BANK_ACCOUNT": _strategy(
        "bank_account_as_schema_field", "BANK_ACCOUNT", "schema_field", "database_column_name",
        ("tên trường tài chính", "schema thanh toán"), ("chuyển khoản", "tài khoản ngân hàng"),
        lambda rng: f"bank_account_field_v{rng.randint(2, 9)}",
        lambda value: value.startswith("bank_account_field_v"),
    ),
    "MONEY": _strategy(
        "money_as_price_tier", "MONEY", "catalog_code", "subscription_price_tier",
        ("mã tier giá", "bảng cấu hình"), ("thanh toán", "số tiền", "VND", "USD"),
        lambda rng: f"PRICE-TIER-{rng.randint(100, 9999)}",
        lambda value: value.startswith("PRICE-TIER-") and not any(c in value for c in ("VND", "USD", "$")),
    ),
    "CARD_ISSUER": _strategy(
        "card_issuer_as_config_flag", "CARD_ISSUER", "configuration_code", "payment_feature_flag",
        ("tùy chọn cấu hình", "file cấu hình checkout"), ("thẻ được phát hành", "mạng thẻ của"),
        lambda rng: f"PAYMENT-{rng.choice(('VISA', 'NAPAS'))}-ENABLED",
        lambda value: value.startswith("PAYMENT-") and value.endswith("-ENABLED"),
    ),
    "CARD_NUMBER": _strategy(
        "card_number_as_inventory_tag", "CARD_NUMBER", "business_identifier", "card_reader_inventory_tag",
        ("tem kho", "thiết bị đọc thẻ"), ("thẻ thanh toán", "số thẻ", "PAN"),
        lambda rng: f"CARD-BIN-{rng.randint(1000, 9999)}-TEST",
        lambda value: value.startswith("CARD-BIN-") and value.endswith("-TEST"),
    ),
    "CVV": _strategy(
        "cvv_as_invalid_test_value", "CVV", "invalid_value", "payment_validation_test",
        ("form kiểm thử", "xác nhận báo lỗi"), ("thẻ thanh toán", "mã bảo mật"),
        lambda rng: f"CVV-{rng.choice(('ABC', 'XYZ'))}",
        lambda value: value.startswith("CVV-") and not value[4:].isdigit(),
    ),
    "IBAN": _strategy(
        "iban_as_invalid_test_value", "IBAN", "invalid_value", "bank_validation_test",
        ("bộ kiểm thử", "kiểm tra nhánh xử lý lỗi"), ("chuyển tiền", "tài khoản quốc tế"),
        lambda rng: f"IBAN-TEST-INVALID-{rng.randint(1, 9)}",
        lambda value: value.startswith("IBAN-TEST-INVALID-"),
    ),
    "SWIFT": _strategy(
        "swift_as_invalid_test_value", "SWIFT", "invalid_value", "bank_code_validation_test",
        ("kiểm thử validate", "trường mã ngân hàng"), ("ngân hàng nhận", "giao dịch quốc tế"),
        lambda rng: f"SWIFT-CODE-{rng.choice(('XYZ', 'TEST'))}-{rng.randint(1, 9)}",
        lambda value: value.startswith("SWIFT-CODE-") and len(value) not in {8, 11},
    ),
    "WALLET": _strategy(
        "wallet_as_invalid_test_value", "WALLET", "invalid_value", "blockchain_validation_test",
        ("module blockchain", "kiểm tra thông báo lỗi"), ("gửi tài sản", "nhận token", "ví của"),
        lambda rng: f"CRYPTO-WALLET-TEST-{rng.randint(1, 99):02d}",
        lambda value: value.startswith("CRYPTO-WALLET-TEST-"),
    ),
    "JOB_TITLE": _strategy(
        "job_as_catalog_code", "JOB_TITLE", "catalog_code", "recruitment_category_code",
        ("mã danh mục tuyển dụng", "nhóm vị trí"), ("đang làm", "nghề nghiệp của", "chức danh"),
        lambda rng: f"JOB-{rng.choice(('SOFTWARE', 'FINANCE', 'SALES'))}-GROUP-{rng.randint(1, 9)}",
        lambda value: value.startswith("JOB-") and "-GROUP-" in value,
    ),
    "ORGANIZATION": _strategy(
        "organization_as_schema_field", "ORGANIZATION", "schema_field", "database_column_name",
        ("tên trường tổ chức", "schema dữ liệu"), ("công ty", "đơn vị công tác", "cơ quan"),
        lambda rng: f"organization_field_v{rng.randint(2, 9)}",
        lambda value: value.startswith("organization_field_v"),
    ),
    "EMPLOYEE_ID": _strategy(
        "employee_as_schema_field", "EMPLOYEE_ID", "schema_field", "database_column_name",
        ("tên cột nhân sự", "schema nhân viên"), ("mã nhân viên được cấp", "nhân viên số"),
        lambda rng: f"employee_id_column_v{rng.randint(2, 9)}",
        lambda value: value.startswith("employee_id_column_v"),
    ),
    "NATIONAL_ID": _strategy(
        "national_id_as_schema_field", "NATIONAL_ID", "schema_field", "database_column_name",
        ("tên trường định danh", "schema công dân"), ("căn cước", "định danh do chính phủ cấp"),
        lambda rng: f"national_id_field_v{rng.randint(2, 9)}",
        lambda value: value.startswith("national_id_field_v"),
    ),
    "PASSPORT": _strategy(
        "passport_as_schema_field", "PASSPORT", "schema_field", "database_column_name",
        ("tên trường hộ chiếu", "schema giấy tờ"), ("hộ chiếu số", "xuất nhập cảnh"),
        lambda rng: f"passport_field_v{rng.randint(2, 9)}",
        lambda value: value.startswith("passport_field_v"),
    ),
    "LICENSE": _strategy(
        "license_as_catalog_code", "LICENSE", "catalog_code", "software_license_category",
        ("mã danh mục phần mềm", "loại giấy phép phần mềm"), ("giấy phép lái xe", "người điều khiển"),
        lambda rng: f"LICENSE-SOFTWARE-{rng.choice('ABC')}{rng.randint(1, 9)}",
        lambda value: value.startswith("LICENSE-SOFTWARE-"),
    ),
    "PLATE": _strategy(
        "plate_as_template_name", "PLATE", "template_identifier", "printing_template_name",
        ("mẫu in", "tên template"), ("biển số xe", "phương tiện đăng ký"),
        lambda rng: f"PLATE-TEMPLATE-V{rng.randint(2, 9)}",
        lambda value: value.startswith("PLATE-TEMPLATE-V"),
    ),
    "TIN": _strategy(
        "tin_as_schema_field", "TIN", "schema_field", "database_column_name",
        ("tên trường thuế", "schema báo cáo"), ("mã số thuế được cấp", "cơ quan thuế"),
        lambda rng: f"tax_id_field_v{rng.randint(2, 9)}",
        lambda value: value.startswith("tax_id_field_v"),
    ),
    "DATE": _strategy(
        "date_as_invalid_calendar_value", "DATE", "invalid_value", "date_validation_test",
        ("dữ liệu lỗi", "kiểm tra trường ngày"), ("ngày hẹn", "hạn chót", "diễn ra vào"),
        lambda rng: f"{rng.randint(32, 39):02d}/{rng.randint(13, 19):02d}/2026",
        lambda value: int(value.split("/")[0]) > 31 and int(value.split("/")[1]) > 12,
        negative_labels=("DATE",),
    ),
    "TIME": _strategy(
        "time_as_sla_code", "TIME", "business_identifier", "service_level_code",
        ("mã SLA", "hệ thống chăm sóc"), ("lúc", "giờ hẹn", "thời điểm"),
        lambda rng: f"SLA-TIME-{rng.randint(1, 99):02d}",
        lambda value: value.startswith("SLA-TIME-"),
    ),
    "MARITAL": _strategy(
        "marital_as_schema_code", "MARITAL", "schema_field", "database_column_name",
        ("mã schema", "tên trường dữ liệu"), ("tình trạng của", "đã kết hôn", "độc thân"),
        lambda rng: f"MARITAL-STATUS-CODE-V{rng.randint(2, 9)}",
        lambda value: value.startswith("MARITAL-STATUS-CODE-V"),
    ),
    "RELIGION": _strategy(
        "religion_as_option_group", "RELIGION", "configuration_code", "survey_option_group",
        ("nhóm lựa chọn", "form khảo sát"), ("tín ngưỡng của", "theo đạo"),
        lambda rng: f"RELIGION-OPTION-GROUP-{rng.choice('ABC')}",
        lambda value: value.startswith("RELIGION-OPTION-GROUP-"),
    ),
    "ETHNICITY": _strategy(
        "ethnicity_as_category_field", "ETHNICITY", "schema_field", "analytics_category_field",
        ("mã danh mục", "bảng phân tích"), ("dân tộc của", "nguồn gốc của"),
        lambda rng: f"ETHNICITY-CATEGORY-ID-{rng.randint(1, 9)}",
        lambda value: value.startswith("ETHNICITY-CATEGORY-ID-"),
    ),
    "TRADE_UNION": _strategy(
        "trade_union_as_policy_section", "TRADE_UNION", "document_section", "company_policy_section",
        ("mục chính sách", "quy định chung"), ("thành viên công đoàn", "tham gia công đoàn"),
        lambda rng: f"TRADE-UNION-POLICY-V{rng.randint(2, 9)}",
        lambda value: value.startswith("TRADE-UNION-POLICY-V"),
    ),
    "NATIONALITY": _strategy(
        "nationality_as_filter_name", "NATIONALITY", "configuration_code", "dashboard_filter_name",
        ("tên bộ lọc", "dashboard tổng hợp"), ("quốc tịch của", "công dân"),
        lambda rng: f"NATIONALITY-FILTER-V{rng.randint(2, 9)}",
        lambda value: value.startswith("NATIONALITY-FILTER-V"),
    ),
    "INSURANCE_ID": _strategy(
        "insurance_as_schema_field", "INSURANCE_ID", "schema_field", "database_column_name",
        ("tên trường bảo hiểm", "schema hợp đồng"), ("mã bảo hiểm được cấp", "hợp đồng bảo hiểm số"),
        lambda rng: f"insurance_id_field_v{rng.randint(2, 9)}",
        lambda value: value.startswith("insurance_id_field_v"),
    ),
    "MEDICAL_INFO": _strategy(
        "medical_as_report_type", "MEDICAL_INFO", "configuration_code", "aggregate_report_type",
        ("nhóm báo cáo tổng hợp", "tên loại dữ liệu"), ("chẩn đoán", "điều trị", "bệnh nhân"),
        lambda rng: f"HEALTH-STATUS-TYPE-{rng.choice('ABC')}",
        lambda value: value.startswith("HEALTH-STATUS-TYPE-"),
    ),
}

for _label, _strategies in ADDITIONAL_HARD_NEGATIVE_STRATEGIES.items():
    HARD_NEGATIVE_STRATEGIES[_label] += _strategies

HARD_NEGATIVE_SUPPORT = {label: label in HARD_NEGATIVE_STRATEGIES for label in _ALL_LABELS}


class PositiveSeedFactory:
    def __init__(self, provider: ValueBankEntityProvider, selector: ContextFrameSelector) -> None:
        self.provider = provider
        self.selector = selector

    def _entities(self, task: GenerationTask, rng: random.Random) -> List[PositiveEntitySeed]:
        entities: List[PositiveEntitySeed] = []
        seen: set[str] = set()
        for label in task.focus_labels:
            generated = self.provider.generate_with_variant(
                label,
                task.language,
                rng,
                excluded_values=seen,
            )
            value = generated.value
            seen.add(value)
            task.diversity_profile.entity_format_variants[label] = generated.format_variant
            entities.append(PositiveEntitySeed(
                label=label,
                value=value,
                semantic_role=SEMANTIC_ROLES.get(label, "record_field"),
                format_variant=generated.format_variant,
            ))
        return entities

    def build(self, task: GenerationTask, taxonomy: Sequence[TaxonomyLabel], rng: random.Random) -> SeedPack:
        return SeedPack(
            task_id=task.task_id,
            sample_type=SampleType.POSITIVE,
            positive_entities=self._entities(task, rng),
            context_frame=self.selector.select(
                task.focus_labels, rng, preferred_frame_id=task.diversity_profile.context_frame_id
            ),
        )


class PureNegativeContentFactory:
    """Builds safe vocabulary only; it does not need an entity-value provider."""

    def __init__(self, selector: ContextFrameSelector) -> None:
        self.selector = selector

    def build(self, task: GenerationTask, taxonomy: Sequence[TaxonomyLabel], rng: random.Random) -> SeedPack:
        frame = self.selector.select(
            task.focus_labels, rng, preferred_frame_id=task.diversity_profile.context_frame_id
        )
        return SeedPack(
            task_id=task.task_id,
            sample_type=SampleType.PURE_NEGATIVE,
            content_seeds=build_content_seeds(frame, rng),
            context_frame=frame,
        )


class HardNegativeSeedFactory:
    def __init__(
        self,
        provider: ValueBankEntityProvider,
        selector: ContextFrameSelector,
        config: HardNegativeConfig,
    ) -> None:
        self.positive_factory = PositiveSeedFactory(provider, selector)
        self.selector = selector
        self.config = config

    def build(self, task: GenerationTask, taxonomy: Sequence[TaxonomyLabel], rng: random.Random) -> SeedPack:
        taxonomy_codes = {label.code for label in taxonomy}
        unknown = set(task.focus_labels) - taxonomy_codes
        if unknown:
            raise UnsupportedDecoyError(f"hard-negative labels are absent from taxonomy: {sorted(unknown)}")
        if self.config.mode == "decoy_only" and len(task.focus_labels) != 1:
            raise UnsupportedDecoyError("decoy_only hard-negative tasks require exactly one taxonomy focus label")
        positives = (
            self.positive_factory._entities(task, rng)
            if self.config.mode == "mixed_contrastive" else []
        )
        supported = [label for label in task.focus_labels if HARD_NEGATIVE_SUPPORT.get(label, False)]
        if not supported:
            raise UnsupportedDecoyError("none of the taxonomy focus labels supports a hard-negative strategy")
        count = rng.randint(self.config.min_decoys, self.config.max_decoys)
        decoys: List[DecoySeed] = []
        seen: set[str] = set()
        for _ in range(count):
            target = rng.choice(supported)
            strategy = rng.choice(HARD_NEGATIVE_STRATEGIES[target])
            for _ in range(10):
                decoy = localize_decoy(
                    strategy.build(rng),
                    task.language,
                )
                key = decoy.value.casefold()
                if key not in seen:
                    seen.add(key)
                    decoys.append(decoy)
                    break
            else:
                raise UnsupportedDecoyError(f"could not generate unique decoy for taxonomy label {target}")
        frame = self.selector.select(
            task.focus_labels,
            rng,
            decoy_semantic_types=[decoy.semantic_type for decoy in decoys],
            preferred_frame_id=task.diversity_profile.context_frame_id,
        )
        return SeedPack(
            task_id=task.task_id,
            sample_type=SampleType.HARD_NEGATIVE,
            hard_negative_mode=self.config.mode,
            positive_entities=positives,
            decoys=decoys,
            context_frame=frame,
        )


class SampleTypeRouter:
    def __init__(
        self,
        positive_factory: SeedPackFactory,
        pure_negative_factory: SeedPackFactory,
        hard_negative_factory: SeedPackFactory,
    ) -> None:
        self.factories = {
            SampleType.POSITIVE: positive_factory,
            SampleType.PURE_NEGATIVE: pure_negative_factory,
            SampleType.HARD_NEGATIVE: hard_negative_factory,
        }

    def build_seed_pack(
        self, task: GenerationTask, taxonomy: Sequence[TaxonomyLabel], rng: random.Random
    ) -> SeedPack:
        return self.factories[SampleType(task.sample_type)].build(task, taxonomy, rng)


def build_sample_type_router(
    value_bank_config: ValueBankConfig,
    hard_negative: HardNegativeConfig,
) -> SampleTypeRouter:
    selector = ContextFrameSelector()
    provider = ValueBankEntityProvider(
        value_bank_config.path,
        value_bank_config.language_files,
    )
    return SampleTypeRouter(
        PositiveSeedFactory(provider, selector),
        PureNegativeContentFactory(selector),
        HardNegativeSeedFactory(provider, selector, hard_negative),
    )
