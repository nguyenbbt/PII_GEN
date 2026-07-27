import random
import unittest

from pii_factory.application.seed_generation import ContextFrameSelector
from pii_factory.application.validators import DeterministicOutputValidator, SeedPackValidator
from data_generator_worker.validation import validate_seeded_contract
from pii_factory.domain.models import (
    ContentSeeds,
    DecoySeed,
    GeneratedEntity,
    HardNegativeConfig,
    LengthTarget,
    PositiveEntitySeed,
    SeedPack,
    TaxonomyLabel,
    ValidationConfig,
)


def frame():
    return ContextFrameSelector().select(["DATE"], random.Random(1))


class DeterministicValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ValidationConfig()
        self.output = DeterministicOutputValidator(self.config)

    def test_seed_validator_rejects_mixed_locale_address(self) -> None:
        pack = SeedPack(
            task_id="bad-address", sample_type="positive", context_frame=frame(),
            positive_entities=[PositiveEntitySeed(
                label="ADDRESS", value="209 JaneHuyện, JohnQuận, TP. Hồ Chí Minh",
                semantic_role="service_location",
            )],
        )
        result = SeedPackValidator(HardNegativeConfig(), self.config).validate(
            pack, ["ADDRESS"], [TaxonomyLabel(code="ADDRESS", definition="address")]
        )
        self.assertFalse(result.valid)
        self.assertIn("mixed_locale", {issue.type for issue in result.issues})

    def test_output_rejects_modified_positive_seed_and_wrong_tag(self) -> None:
        pack = SeedPack(
            task_id="positive-1", sample_type="positive", context_frame=frame(),
            positive_entities=[PositiveEntitySeed(
                label="DATE", value="21/10/2026", semantic_role="appointment_date"
            )],
        )
        result = self.output.validate(
            tagged_text="Hẹn vào <TIME>21/10/2025</TIME>.",
            entities=[GeneratedEntity(label="TIME", value="21/10/2025")],
            seed_pack=pack, focus_labels=["DATE"], max_entities=2,
        )
        self.assertFalse(result.valid)
        self.assertIn("missing_positive_seed", {issue.type for issue in result.issues})

    def test_output_rejects_clean_text_outside_numeric_length_target(self) -> None:
        pack = SeedPack(
            task_id="short-positive",
            sample_type="positive",
            context_frame=frame(),
            positive_entities=[
                PositiveEntitySeed(
                    label="DATE",
                    value="21/10/2026",
                    semantic_role="appointment_date",
                )
            ],
        )

        result = self.output.validate(
            tagged_text="Hẹn <DATE>21/10/2026</DATE>.",
            entities=[GeneratedEntity(label="DATE", value="21/10/2026")],
            seed_pack=pack,
            focus_labels=["DATE"],
            max_entities=2,
            length_target=LengthTarget(
                bucket="short",
                min_words=80,
                max_words=120,
                unit="content_units",
                min_units=3,
                max_units=5,
            ),
        )

        self.assertFalse(result.valid)
        self.assertIn("length_out_of_range", {issue.type for issue in result.issues})

    def test_pure_negative_rejects_structured_candidates_and_accepts_generic_text(self) -> None:
        pack = SeedPack(
            task_id="negative-1", sample_type="pure_negative", context_frame=frame(),
            content_seeds=ContentSeeds(
                domain="support", document_type="internal_note", tone="neutral",
                generic_roles=["bộ phận kỹ thuật"], actions=["kiểm tra yêu cầu"], objects=["biểu mẫu"],
            ),
        )
        good = self.output.validate(
            tagged_text="Bộ phận kỹ thuật đã kiểm tra yêu cầu và chuyển biểu mẫu sang bước tiếp theo.",
            entities=[], seed_pack=pack, focus_labels=["EMAIL"], max_entities=2,
        )
        for candidate in ("Hãy gửi qua demo@example.test.", "Gọi số 0912345678.", "Hẹn ngày 21/10/2026."):
            with self.subTest(candidate=candidate):
                bad = self.output.validate(
                    tagged_text=candidate, entities=[], seed_pack=pack,
                    focus_labels=["EMAIL"], max_entities=2,
                )
                self.assertFalse(bad.valid)
        self.assertTrue(good.valid)

    def test_hard_negative_requires_empty_entities_and_nearby_cue(self) -> None:
        pack = SeedPack(
            task_id="hard-1", sample_type="hard_negative", hard_negative_mode="decoy_only",
            context_frame=frame(), positive_entities=[],
            decoys=[DecoySeed(
                strategy_id="date_as_invalid_calendar_value", target_label="DATE",
                value="32/13/2026", family="invalid_value", semantic_type="date_validation_test",
                negative_labels=["DATE"], required_context_cues=["dữ liệu lỗi"],
                forbidden_context_cues=["ngày hẹn"],
            )],
        )
        valid = self.output.validate(
            tagged_text="Bộ kiểm thử ghi nhận dữ liệu lỗi 32/13/2026 trong trường đầu vào.",
            entities=[], seed_pack=pack, focus_labels=["DATE"], max_entities=2,
        )
        unclear = self.output.validate(
            tagged_text="Bộ kiểm thử ghi nhận mã 32/13/2026 trong trường đầu vào.",
            entities=[], seed_pack=pack, focus_labels=["DATE"], max_entities=2,
        )
        tagged = self.output.validate(
            tagged_text="Dữ liệu lỗi <DATE>32/13/2026</DATE> được ghi nhận.",
            entities=[GeneratedEntity(label="DATE", value="32/13/2026")],
            seed_pack=pack, focus_labels=["DATE"], max_entities=2,
        )

        self.assertTrue(valid.valid)
        self.assertIn("decoy_context_unclear", {issue.type for issue in unclear.issues})
        self.assertIn("decoy_tagged", {issue.type for issue in tagged.issues})

    def test_decoy_only_accepts_one_natural_repetition_with_context_for_each_mention(self) -> None:
        pack = SeedPack(
            task_id="hard-repeat", sample_type="hard_negative", hard_negative_mode="decoy_only",
            context_frame=frame(),
            decoys=[DecoySeed(
                strategy_id="date_as_invalid_calendar_value", target_label="DATE",
                value="32/13/2026", family="invalid_value", semantic_type="date_validation_test",
                negative_labels=["DATE"], required_context_cues=["dữ liệu lỗi"],
                forbidden_context_cues=["ngày hẹn"],
            )],
        )
        text = (
            "Bộ kiểm thử ghi nhận dữ liệu lỗi 32/13/2026 trong trường đầu vào. "
            "Nhân viên xác nhận dữ liệu lỗi cần sửa vẫn là 32/13/2026."
        )

        result = self.output.validate(
            tagged_text=text, entities=[], seed_pack=pack, focus_labels=["DATE"], max_entities=2,
        )

        self.assertTrue(result.valid, [issue.dict() for issue in result.issues])
        validate_seeded_contract(
            tagged_text=text,
            entities=[],
            positive_entities=[],
            decoys=[pack.decoys[0].dict()],
            max_decoy_occurrences=2,
        )

    def test_decoy_only_rejects_excessive_or_context_free_repetition(self) -> None:
        pack = SeedPack(
            task_id="hard-repeat-invalid", sample_type="hard_negative", hard_negative_mode="decoy_only",
            context_frame=frame(),
            decoys=[DecoySeed(
                strategy_id="date_as_invalid_calendar_value", target_label="DATE",
                value="32/13/2026", family="invalid_value", semantic_type="date_validation_test",
                negative_labels=["DATE"], required_context_cues=["dữ liệu lỗi"],
                forbidden_context_cues=["ngày hẹn"],
            )],
        )
        context_free = self.output.validate(
            tagged_text=(
                "Bộ kiểm thử ghi nhận dữ liệu lỗi 32/13/2026 trong trường đầu vào. "
                "Nhân viên đọc lại chuỗi 32/13/2026."
            ),
            entities=[], seed_pack=pack, focus_labels=["DATE"], max_entities=2,
        )
        excessive = self.output.validate(
            tagged_text="Dữ liệu lỗi 32/13/2026 được đối chiếu với dữ liệu lỗi 32/13/2026 và dữ liệu lỗi 32/13/2026.",
            entities=[], seed_pack=pack, focus_labels=["DATE"], max_entities=2,
        )

        self.assertIn("decoy_context_unclear", {issue.type for issue in context_free.issues})
        self.assertIn("decoy_occurrence", {issue.type for issue in excessive.issues})

    def test_seed_validator_detects_collision_with_another_taxonomy_label(self) -> None:
        pack = SeedPack(
            task_id="collision-1", sample_type="hard_negative", hard_negative_mode="decoy_only",
            context_frame=frame(),
            decoys=[DecoySeed(
                strategy_id="date_as_invalid_calendar_value", target_label="DATE",
                value="qa@example.com", family="invalid_value", semantic_type="date_validation_test",
                negative_labels=["DATE"], required_context_cues=["dữ liệu lỗi"],
                forbidden_context_cues=["ngày hẹn"],
            )],
        )
        result = SeedPackValidator(HardNegativeConfig(), self.config).validate(
            pack, ["DATE"], [
                TaxonomyLabel(code="DATE", definition="date"),
                TaxonomyLabel(code="EMAIL", definition="email"),
            ],
        )

        self.assertFalse(result.valid)
        self.assertIn("taxonomy_collision", {issue.type for issue in result.issues})


if __name__ == "__main__":
    unittest.main()
