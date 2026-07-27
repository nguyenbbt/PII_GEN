from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Callable, Dict, Sequence

from faker import Faker


VIETNAMESE_ADMINISTRATIVE_AREAS: Dict[str, Dict[str, Sequence[str]]] = {
    "TP. Hồ Chí Minh": {
        "Quận 3": ("phường 1", "phường 2", "phường Võ Thị Sáu"),
        "Quận 7": ("phường Tân Phú", "phường Tân Quy", "phường Tân Thuận Đông"),
    },
    "Đà Nẵng": {
        "quận Hải Châu": ("phường Hải Châu", "phường Thạch Thang"),
        "quận Sơn Trà": ("phường An Hải Bắc", "phường Phước Mỹ"),
    },
    "Hà Nội": {
        "quận Ba Đình": ("phường Điện Biên", "phường Kim Mã"),
        "quận Cầu Giấy": ("phường Dịch Vọng", "phường Yên Hòa"),
    },
}

STREETS = ("Hoa Phượng", "Nguyễn Du", "Lê Lợi", "Trần Hưng Đạo", "Hoàng Diệu")
VIETNAMESE_SURNAMES = (
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ",
    "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý",
)
VIETNAMESE_MIDDLE_NAMES = (
    "Văn", "Thị", "Đức", "Ngọc", "Minh", "Quốc", "Thanh", "Hải",
    "Gia", "Hữu", "Khánh", "Xuân",
)
VIETNAMESE_GIVEN_NAMES = (
    "An", "Bình", "Bảo", "Chi", "Dung", "Giang", "Hạnh", "Hiếu",
    "Huy", "Khang", "Lan", "Linh", "Mai", "Nam", "Phương", "Quân",
    "Trang", "Tú", "Uyên", "Yến",
)


@dataclass(frozen=True)
class GeneratedEntityValue:
    value: str
    format_variant: str


class VietnameseAddressProvider:
    def generate_pair_with_variants(
        self,
        rng: random.Random,
    ) -> tuple[GeneratedEntityValue, GeneratedEntityValue]:
        """Build one full address while preserving taxonomy span boundaries."""
        city = rng.choice(tuple(VIETNAMESE_ADMINISTRATIVE_AREAS))
        district = rng.choice(tuple(VIETNAMESE_ADMINISTRATIVE_AREAS[city]))
        ward = rng.choice(tuple(VIETNAMESE_ADMINISTRATIVE_AREAS[city][district]))
        street = f"{rng.randint(1, 999)} đường {rng.choice(STREETS)}"
        address_variant = rng.choice(("street", "apartment", "building", "room"))
        if address_variant == "apartment":
            street = f"Căn hộ {rng.choice('ABCDEFGH')}{rng.randint(1, 40):02d}, {street}"
        elif address_variant == "building":
            street = f"Tòa {rng.choice('ABCDEFGH')}, {street}"
        elif address_variant == "room":
            street = f"Phòng {rng.randint(101, 1608)}, {street}"

        location_variant = rng.choice(("city", "district_city", "ward_district_city"))
        locations = {
            "city": city,
            "district_city": f"{district}, {city}",
            "ward_district_city": f"{ward}, {district}, {city}",
        }
        return (
            GeneratedEntityValue(street, address_variant),
            GeneratedEntityValue(locations[location_variant], location_variant),
        )

    def generate_with_variant(self, rng: random.Random) -> GeneratedEntityValue:
        address, _ = self.generate_pair_with_variants(rng)
        return address

    def generate_location_with_variant(
        self,
        rng: random.Random,
    ) -> GeneratedEntityValue:
        _, location = self.generate_pair_with_variants(rng)
        return location

    def generate(self, rng: random.Random) -> str:
        return self.generate_with_variant(rng).value


class FakerEntityProvider:
    """Produces taxonomy-valid synthetic values and records their format variant."""

    def __init__(self, locale: str = "vi_VN") -> None:
        self.locale = locale
        self.addresses = VietnameseAddressProvider()

    def generate(self, label: str, rng: random.Random) -> str:
        return self.generate_with_variant(label, rng).value

    def generate_with_variant(self, label: str, rng: random.Random) -> GeneratedEntityValue:
        faker = Faker(self.locale)
        faker.seed_instance(rng.getrandbits(64))
        digits = lambda count: "".join(rng.choice(string.digits) for _ in range(count))
        focus_value = self._focus_value(label, rng, faker, digits)
        if focus_value is not None:
            return focus_value
        legacy: Dict[str, Callable[[], str]] = {
            "PREFIX": lambda: rng.choice(("Ông", "Bà", "Bác sĩ", "Tiến sĩ")),
            "GENDER": lambda: rng.choice(("nam", "nữ", "phi nhị nguyên")),
            "AGE": lambda: f"{rng.randint(18, 85)} tuổi",
            "BIRTHDATE": lambda: f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1940, 2006)}",
            "ZIP_CODE": lambda: digits(6),
            "COORDINATE": lambda: f"{rng.uniform(8, 23):.5f}, {rng.uniform(102, 110):.5f}",
            "USERNAME": lambda: f"nguoidung_{digits(6)}",
            "ACCOUNT_ID": lambda: f"USR-{digits(8)}",
            "TICKET_ID": lambda: f"INC-{digits(7)}",
            "PASSWORD": lambda: f"Test!{digits(6)}Aa",
            "PIN": lambda: digits(4),
            "API_KEY": lambda: f"sk_test_{digits(20)}",
            "BANK_ACCOUNT": lambda: digits(12),
            "CARD_ISSUER": lambda: rng.choice(("Visa", "Mastercard", "NAPAS")),
            "CVV": lambda: digits(3),
            "IBAN": lambda: f"DE89{digits(18)}",
            "SWIFT": lambda: rng.choice(("DEUTDEFF", "BOFAUS3N", "CHASUS33")),
            "WALLET": lambda: "0x" + "".join(rng.choice("0123456789abcdef") for _ in range(40)),
            "JOB_TITLE": lambda: rng.choice(("kỹ sư phần mềm", "kế toán viên", "giáo viên")),
            "ORGANIZATION": lambda: f"Công ty TNHH Thử Nghiệm {digits(4)}",
            "EMPLOYEE_ID": lambda: f"EMP-{digits(6)}",
            "NATIONAL_ID": lambda: digits(12),
            "LICENSE": lambda: f"VN{digits(9)}",
            "TIN": lambda: digits(10),
            "MARITAL": lambda: rng.choice(("độc thân", "đã kết hôn", "đã ly hôn")),
            "RELIGION": lambda: rng.choice(("Phật giáo", "Công giáo", "không theo tôn giáo")),
            "ETHNICITY": lambda: rng.choice(("Kinh", "Tày", "Thái", "Mường")),
            "TRADE_UNION": lambda: f"Công đoàn cơ sở Thử Nghiệm {digits(4)}",
            "NATIONALITY": lambda: rng.choice(("Việt Nam", "Singapore", "Nhật Bản")),
            "INSURANCE_ID": lambda: f"BH-{digits(10)}",
        }
        try:
            return GeneratedEntityValue(legacy[label]().replace("\n", ", "), "default")
        except KeyError as exc:
            raise ValueError(f"no Faker provider registered for label {label}") from exc

    def generate_address_location_pair(
        self,
        rng: random.Random,
    ) -> tuple[GeneratedEntityValue, GeneratedEntityValue]:
        return self.addresses.generate_pair_with_variants(rng)

    def _focus_value(
        self,
        label: str,
        rng: random.Random,
        faker: Faker,
        digits: Callable[[int], str],
    ) -> GeneratedEntityValue | None:
        variant = rng.choice(("a", "b", "c"))
        if label == "PERSON":
            surname = rng.choice(VIETNAMESE_SURNAMES)
            given_name = rng.choice(VIETNAMESE_GIVEN_NAMES)
            middle_names = rng.sample(VIETNAMESE_MIDDLE_NAMES, 2)
            values = {
                "a": f"{surname} {given_name}",
                "b": f"{surname} {middle_names[0]} {given_name}",
                "c": f"{surname} {middle_names[0]} {middle_names[1]} {given_name}",
            }
            names = {
                "a": "family_given",
                "b": "family_middle_given",
                "c": "family_double_middle_given",
            }
            return GeneratedEntityValue(values[variant], names[variant])
        if label == "PHONE":
            compact = rng.choice(("032", "037", "056", "076", "091")) + digits(7)
            values = {"a": compact, "b": f"{compact[:4]} {compact[4:7]} {compact[7:]}",
                      "c": f"+84 {compact[1:3]} {compact[3:6]} {compact[6:]}"}
            return GeneratedEntityValue(values[variant], {"a": "local_compact", "b": "local_spaced", "c": "international_spaced"}[variant])
        if label == "EMAIL":
            local = digits(8)
            values = {"a": f"nguoidung.{local}@example.test", "b": f"ho-so-{local}@mail.test",
                      "c": f"contact+{local}@demo.test"}
            return GeneratedEntityValue(values[variant], {"a": "dot_local", "b": "hyphen_local", "c": "plus_alias"}[variant])
        if label == "ADDRESS":
            return self.addresses.generate_with_variant(rng)
        if label == "LOCATION":
            return self.addresses.generate_location_with_variant(rng)
        if label == "DATE":
            day, month, year = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2020, 2035)
            return GeneratedEntityValue({"a": f"{day:02d}/{month:02d}/{year}", "b": f"{year}-{month:02d}-{day:02d}",
                                         "c": f"{day:02d}-{month:02d}-{year}"}[variant],
                                        {"a": "dmy_slash", "b": "iso_date", "c": "dmy_hyphen"}[variant])
        if label == "TIME":
            hour, minute, second = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
            hour12 = hour % 12 or 12
            return GeneratedEntityValue({"a": f"{hour:02d}:{minute:02d}", "b": f"{hour:02d}:{minute:02d}:{second:02d}",
                                         "c": f"{hour12:02d}:{minute:02d} {'AM' if hour < 12 else 'PM'}"}[variant],
                                        {"a": "24h", "b": "24h_seconds", "c": "12h_meridiem"}[variant])
        if label == "MONEY":
            amount = rng.randint(101, 9_999_999)
            values = {"a": f"{amount:,} VND".replace(",", "."), "b": f"VND {amount:,}", "c": f"₫{amount:,}"}
            return GeneratedEntityValue(values[variant], {"a": "amount_unit", "b": "code_prefix", "c": "symbol_prefix"}[variant])
        if label == "URL":
            suffix = digits(10)
            values = {"a": f"https://example.test/ho-so/{suffix}", "b": f"https://portal.example.test/ticket/{suffix}",
                      "c": f"http://api.example.test/v1/records/{suffix}"}
            return GeneratedEntityValue(values[variant], {"a": "web_path", "b": "subdomain_path", "c": "api_endpoint"}[variant])
        if label == "IP":
            if variant == "c":
                return GeneratedEntityValue(f"2001:db8:{rng.randint(0, 65535):x}:{rng.randint(0, 65535):x}::1", "ipv6_documentation")
            block = "192.0.2" if variant == "a" else "198.51.100"
            return GeneratedEntityValue(f"{block}.{rng.randint(1, 254)}", "ipv4_documentation")
        if label == "CARD_NUMBER":
            prefix = {"a": "424242", "b": "555555", "c": "400000"}[variant]
            payload = prefix + digits(15 - len(prefix))
            return GeneratedEntityValue(payload + self._luhn_check_digit(payload), {"a": "visa_test", "b": "mastercard_test", "c": "visa_alt_test"}[variant])
        if label == "PLATE":
            area = rng.randint(29, 99)
            values = {"a": f"{area}A-{digits(5)}", "b": f"{area}B1-{digits(5)}", "c": f"{area}C {digits(3)}.{digits(2)}"}
            return GeneratedEntityValue(values[variant], {"a": "car_standard", "b": "motorbike", "c": "spaced_dotted"}[variant])
        if label == "PASSPORT":
            values = {"a": f"B{digits(8)}", "b": f"C{digits(8)}", "c": f"P{digits(7)}"}
            return GeneratedEntityValue(values[variant], {"a": "series_b", "b": "series_c", "c": "series_p"}[variant])
        if label == "MEDICAL_INFO":
            condition = rng.choice(("hen suyễn", "tăng huyết áp", "đau nửa đầu", "dị ứng penicillin", "thiếu máu"))
            values = {"a": f"tiền sử {condition}", "b": f"đang điều trị {condition}", "c": f"kết quả theo dõi ghi nhận {condition}"}
            return GeneratedEntityValue(values[variant], {"a": "medical_history", "b": "current_treatment", "c": "test_result"}[variant])
        return None

    @staticmethod
    def _luhn_check_digit(payload: str) -> str:
        total = 0
        for index, character in enumerate(reversed(payload)):
            digit = int(character) * (2 if index % 2 == 0 else 1)
            total += digit - 9 if digit > 9 else digit
        return str((10 - total % 10) % 10)
