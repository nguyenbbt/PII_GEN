from __future__ import annotations

from .context_catalog import ALL_LABELS
from .hard_negative_base import DecoyStrategy, strategy


_SPECS = {
    "PERSON": (
        ("person_as_ui_component", "module_identifier", "ui_component_name", ("tên component giao diện", "thư viện UI"), ("họ tên", "người tên"), "UI-PERSON"),
        ("person_as_model_code", "catalog_code", "data_model_code", ("mã mô hình dữ liệu", "catalog schema"), ("khách hàng", "nhân viên"), "MODEL-PER"),
    ),
    "PHONE": (
        ("phone_as_config_field", "schema_field", "configuration_field_name", ("tên trường cấu hình", "schema liên lạc"), ("gọi", "nhắn tin"), "FIELD-TEL"),
        ("phone_as_test_case", "invalid_value", "phone_validation_case", ("mã test validate", "ca kiểm thử"), ("số điện thoại", "liên hệ"), "TEST-PHONE"),
    ),
    "EMAIL": (
        ("email_as_template_code", "catalog_code", "notification_template_code", ("mã mẫu thông báo", "catalog template"), ("gửi thư", "hộp thư"), "TMPL-MAIL"),
        ("email_as_report_column", "schema_field", "report_column_name", ("tên cột báo cáo", "schema xuất dữ liệu"), ("địa chỉ email", "liên hệ qua"), "COL-EMAIL"),
    ),
    "ADDRESS": (
        ("address_as_config_key", "configuration_code", "address_config_key", ("khóa cấu hình", "bảng tham số"), ("số nhà", "đường"), "CFG-ADDR"),
        ("address_as_form_field", "schema_field", "form_field_name", ("tên trường biểu mẫu", "schema giao hàng"), ("nơi ở", "giao tới"), "FIELD-ADDR"),
    ),
    "DATE": (
        ("date_as_release_code", "catalog_code", "release_calendar_code", ("mã lịch phát hành", "catalog release"), ("ngày", "lịch hẹn"), "REL-DATE"),
        ("date_as_schema_column", "schema_field", "database_column_name", ("tên cột ngày", "schema dữ liệu"), ("vào ngày", "ngày giao dịch"), "COL-DATE"),
    ),
    "TIME": (
        ("time_as_release_window_code", "catalog_code", "deployment_window_code", ("mã cửa sổ phát hành", "catalog triển khai"), ("lúc", "giờ hẹn"), "WIN-TIME"),
        ("time_as_report_column", "schema_field", "report_column_name", ("tên cột thời gian", "schema báo cáo"), ("thời điểm", "vào lúc"), "COL-TIME"),
    ),
    "MONEY": (
        ("money_as_metric_code", "analytics_code", "revenue_metric_name", ("mã chỉ số báo cáo", "dashboard tổng hợp"), ("thanh toán", "số tiền"), "METRIC-MONEY"),
        ("money_as_form_field", "schema_field", "financial_field_name", ("tên trường tài chính", "schema hóa đơn"), ("VND", "USD"), "FIELD-AMOUNT"),
    ),
    "URL": (
        ("url_as_config_key", "configuration_code", "routing_config_key", ("khóa cấu hình định tuyến", "bảng tham số"), ("truy cập", "liên kết"), "CFG-URL"),
        ("url_as_test_fixture", "test_identifier", "web_fixture_name", ("tên fixture kiểm thử", "bộ test web"), ("đường dẫn web", "mở trang"), "FIXTURE-URL"),
    ),
    "IP": (
        ("ip_as_inventory_code", "business_identifier", "network_inventory_tag", ("mã thiết bị kho", "danh mục phần cứng"), ("địa chỉ mạng", "kết nối"), "ASSET-IP"),
        ("ip_as_schema_field", "schema_field", "network_field_name", ("tên cột mạng", "schema log"), ("máy chủ", "địa chỉ IP"), "FIELD-IP"),
    ),
    "CARD_NUMBER": (
        ("card_number_as_form_field", "schema_field", "payment_field_name", ("tên trường thanh toán", "schema biểu mẫu"), ("số thẻ", "thẻ tín dụng"), "FIELD-PAN"),
        ("card_number_as_test_case", "test_identifier", "card_validation_case", ("mã ca kiểm thử", "bộ test thanh toán"), ("chủ thẻ", "thanh toán bằng thẻ"), "TEST-PAN"),
    ),
    "PLATE": (
        ("plate_as_report_column", "schema_field", "vehicle_column_name", ("tên cột phương tiện", "schema vận tải"), ("biển số", "xe mang"), "COL-PLATE"),
        ("plate_as_asset_class", "catalog_code", "vehicle_asset_class", ("mã lớp tài sản", "catalog phương tiện"), ("đăng ký xe", "biển kiểm soát"), "CLASS-PLATE"),
    ),
    "PASSPORT": (
        ("passport_as_form_field", "schema_field", "travel_field_name", ("tên trường hồ sơ", "schema du lịch"), ("hộ chiếu", "xuất nhập cảnh"), "FIELD-PASS"),
        ("passport_as_test_fixture", "test_identifier", "document_fixture_name", ("mã fixture tài liệu", "bộ test OCR"), ("số hộ chiếu", "người mang hộ chiếu"), "FIXTURE-PASS"),
    ),
    "MEDICAL_INFO": (
        ("medical_as_schema_section", "schema_field", "health_section_name", ("tên mục schema", "biểu mẫu y tế"), ("chẩn đoán", "bệnh nhân"), "SECTION-HEALTH"),
        ("medical_as_dashboard_metric", "analytics_code", "health_metric_code", ("mã chỉ số tổng hợp", "dashboard thống kê"), ("điều trị", "triệu chứng"), "METRIC-HEALTH"),
    ),
    "LOCATION": (
        ("location_as_report_group", "analytics_code", "regional_report_group", ("mã nhóm báo cáo", "dashboard vùng"), ("tỉnh", "thành phố"), "GROUP-LOC"),
        ("location_as_form_field", "schema_field", "region_field_name", ("tên trường khu vực", "schema địa lý"), ("quốc gia", "địa điểm"), "FIELD-LOC"),
    ),
}


def _build_strategy(label: str, spec: tuple[object, ...]) -> DecoyStrategy:
    strategy_id, family, semantic_type, required, forbidden, prefix = spec
    prefix_text = str(prefix)
    return strategy(
        str(strategy_id), label, str(family), str(semantic_type), tuple(required), tuple(forbidden),
        lambda rng, prefix_text=prefix_text: f"{prefix_text}-{rng.choice('ABC')}{rng.randint(10, 999):03d}",
        lambda value, prefix_text=prefix_text: value.startswith(f"{prefix_text}-"),
    )[0]


def _generic_specs(label: str) -> tuple[tuple[object, ...], ...]:
    slug = label.casefold()
    return (
        (
            f"{slug}_as_secondary_schema_field", "schema_field", "database_column_name",
            ("tên trường dữ liệu", "schema hệ thống"), ("thông tin của", "giá trị cá nhân"),
            f"FIELD-{label}",
        ),
        (
            f"{slug}_as_secondary_catalog_code", "catalog_code", "classification_code",
            ("mã danh mục", "bảng phân loại"), ("thông tin của", "giá trị cá nhân"),
            f"CAT-{label}",
        ),
    )


ADDITIONAL_HARD_NEGATIVE_STRATEGIES = {
    label: tuple(_build_strategy(label, spec) for spec in _SPECS.get(label, _generic_specs(label)))
    for label in ALL_LABELS
}
