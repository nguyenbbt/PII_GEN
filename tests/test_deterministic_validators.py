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

    def test_seed_validator_accepts_standalone_street_level_address_components(self) -> None:
        for value in (
            "Căn hộ A12, Tòa nhà Bình Minh",
            "Phòng 804, Tòa B",
            "125 đường Lê Lợi",
        ):
            with self.subTest(value=value):
                pack = SeedPack(
                    task_id="valid-address",
                    sample_type="positive",
                    context_frame=frame(),
                    positive_entities=[
                        PositiveEntitySeed(
                            label="ADDRESS",
                            value=value,
                            semantic_role="street_address",
                        )
                    ],
                )
                result = SeedPackValidator(
                    HardNegativeConfig(),
                    self.config,
                ).validate(
                    pack,
                    ["ADDRESS"],
                    [TaxonomyLabel(code="ADDRESS", definition="address")],
                )

                self.assertTrue(result.valid, [issue.dict() for issue in result.issues])

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
        self.assertIn("length_below_minimum", {issue.type for issue in result.issues})

    def test_long_length_target_has_no_upper_word_limit(self) -> None:
        pack = SeedPack(
            task_id="long-positive",
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
        tagged_text = (
            "<DATE>21/10/2026</DATE> "
            + " ".join(f"nội_dung_{index}" for index in range(450))
        )

        result = self.output.validate(
            tagged_text=tagged_text,
            entities=[GeneratedEntity(label="DATE", value="21/10/2026")],
            seed_pack=pack,
            focus_labels=["DATE"],
            max_entities=2,
            length_target=LengthTarget(
                bucket="long",
                min_words=260,
                max_words=400,
                unit="content_units",
                min_units=10,
                max_units=14,
            ),
        )

        self.assertNotIn(
            "length_below_minimum",
            {issue.type for issue in result.issues},
        )

    def test_additional_annotated_context_entity_is_allowed(self) -> None:
        pack = SeedPack(
            task_id="context-entity",
            sample_type="positive",
            context_frame=frame(),
            positive_entities=[
                PositiveEntitySeed(
                    label="PERSON",
                    value="Mai Huyền",
                    semantic_role="patient",
                )
            ],
        )

        result = self.output.validate(
            tagged_text=(
                "<PERSON>Mai Huyền</PERSON> hẹn tái khám vào "
                "<DATE>15/05/2024</DATE>."
            ),
            entities=[
                GeneratedEntity(label="PERSON", value="Mai Huyền"),
                GeneratedEntity(label="DATE", value="15/05/2024"),
            ],
            seed_pack=pack,
            focus_labels=["PERSON"],
            allowed_labels=["PERSON", "DATE"],
            max_entities=2,
        )

        self.assertTrue(result.valid, [issue.dict() for issue in result.issues])

    def test_repeated_positive_seed_is_valid_when_every_occurrence_is_tagged(self) -> None:
        pack = SeedPack(
            task_id="repeated-person",
            sample_type="positive",
            context_frame=frame(),
            positive_entities=[
                PositiveEntitySeed(
                    label="PERSON",
                    value="Mai Huyền",
                    semantic_role="patient",
                )
            ],
        )
        result = self.output.validate(
            tagged_text=(
                "Người liên hệ <PERSON>Mai Huyền</PERSON>. "
                "Bệnh nhân <PERSON>Mai Huyền</PERSON> đã xác nhận. "
                "Hồ sơ của <PERSON>Mai Huyền</PERSON> được cập nhật."
            ),
            entities=[
                GeneratedEntity(label="PERSON", value="Mai Huyền"),
                GeneratedEntity(label="PERSON", value="Mai Huyền"),
                GeneratedEntity(label="PERSON", value="Mai Huyền"),
            ],
            seed_pack=pack,
            focus_labels=["PERSON"],
            max_entities=3,
        )

        self.assertTrue(result.valid, [issue.dict() for issue in result.issues])

    def test_composite_address_seed_accepts_taxonomy_safe_partition(self) -> None:
        value = "34 Nguyễn Chí Thanh, Ba Đình, Hà Nội"
        pack = SeedPack(
            task_id="partitioned-address",
            sample_type="positive",
            context_frame=frame(),
            positive_entities=[PositiveEntitySeed(
                label="ADDRESS",
                value=value,
                semantic_role="service_address",
            )],
        )
        tagged_text = (
            "Giao tại <ADDRESS>34 Nguyễn Chí Thanh</ADDRESS>, "
            "<LOCATION>Ba Đình, Hà Nội</LOCATION>."
        )
        entities = [
            GeneratedEntity(label="ADDRESS", value="34 Nguyễn Chí Thanh"),
            GeneratedEntity(label="LOCATION", value="Ba Đình, Hà Nội"),
        ]

        result = self.output.validate(
            tagged_text=tagged_text,
            entities=entities,
            seed_pack=pack,
            focus_labels=["ADDRESS"],
            allowed_labels=["ADDRESS", "LOCATION"],
            max_entities=2,
        )

        self.assertTrue(result.valid, [issue.dict() for issue in result.issues])
        validate_seeded_contract(
            tagged_text=tagged_text,
            entities=[entity.dict() for entity in entities],
            positive_entities=[pack.positive_entities[0].dict()],
            decoys=[],
        )

    def test_composite_seed_rejects_full_relabel_or_unannotated_surface(self) -> None:
        value = "34 Nguyễn Chí Thanh, Ba Đình, Hà Nội"
        pack = SeedPack(
            task_id="unsafe-address",
            sample_type="positive",
            context_frame=frame(),
            positive_entities=[PositiveEntitySeed(
                label="ADDRESS",
                value=value,
                semantic_role="service_address",
            )],
        )
        for tagged_text, entities in (
            (
                f"<LOCATION>{value}</LOCATION>",
                [GeneratedEntity(label="LOCATION", value=value)],
            ),
            (value, []),
        ):
            with self.subTest(tagged_text=tagged_text):
                result = self.output.validate(
                    tagged_text=tagged_text,
                    entities=entities,
                    seed_pack=pack,
                    focus_labels=["ADDRESS"],
                    allowed_labels=["ADDRESS", "LOCATION"],
                    max_entities=2,
                )
                self.assertIn(
                    "positive_seed_annotation_mismatch",
                    {issue.type for issue in result.issues},
                )

    def test_template_artifacts_and_clear_context_entities_are_fixable_findings(self) -> None:
        pack = SeedPack(
            task_id="context-findings",
            sample_type="positive",
            context_frame=frame(),
            positive_entities=[PositiveEntitySeed(
                label="PERSON",
                value="Mai Huyền",
                semantic_role="requester",
            )],
        )
        result = self.output.validate(
            tagged_text=(
                "<PERSON>Mai Huyền</PERSON> gửi [Tên Công ty] phiếu hỗ trợ "
                "mã sự cố SV-20231027-001 cho xe biển số 51C-123.45."
            ),
            entities=[GeneratedEntity(label="PERSON", value="Mai Huyền")],
            seed_pack=pack,
            focus_labels=["PERSON"],
            allowed_labels=["PERSON", "ORG", "TICKET_ID", "PLATE"],
            max_entities=4,
        )

        findings = {(issue.type, issue.label, issue.value) for issue in result.issues}
        self.assertIn(("template_artifact", None, "[Tên Công ty]"), findings)
        self.assertIn(
            ("missing_annotation_candidate", "TICKET_ID", "SV-20231027-001"),
            findings,
        )
        self.assertIn(
            ("missing_annotation_candidate", "PLATE", "51C-123.45"),
            findings,
        )

    def test_boundary_whitespace_produces_one_seed_specific_finding(self) -> None:
        pack = SeedPack(
            task_id="boundary-space",
            sample_type="positive",
            context_frame=frame(),
            positive_entities=[PositiveEntitySeed(
                label="PERSON",
                value="Mai Huyền",
                semantic_role="requester",
            )],
        )
        result = self.output.validate(
            tagged_text="Người gửi <PERSON> Mai Huyền </PERSON> đã xác nhận.",
            entities=[{"label": "PERSON", "value": " Mai Huyền "}],
            seed_pack=pack,
            focus_labels=["PERSON"],
            max_entities=2,
        )

        seed_findings = [
            issue for issue in result.issues
            if issue.type in {
                "missing_positive_seed",
                "missing_entity_metadata",
                "duplicate_positive_seed",
                "entity_boundary_whitespace",
                "positive_seed_annotation_mismatch",
            }
        ]
        self.assertEqual(
            [issue.type for issue in seed_findings],
            ["entity_boundary_whitespace"],
        )

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

    def test_hard_negative_rejects_meta_explanation_and_reports_exact_cues(self) -> None:
        pack = SeedPack(
            task_id="hard-meta",
            sample_type="hard_negative",
            hard_negative_mode="decoy_only",
            context_frame=frame(),
            decoys=[DecoySeed(
                strategy_id="date_as_invalid_calendar_value",
                target_label="DATE",
                value="32/13/2026",
                family="invalid_value",
                semantic_type="date_validation_test",
                negative_labels=["DATE"],
                required_context_cues=["dữ liệu lỗi"],
                forbidden_context_cues=["ngày hẹn"],
            )],
        )
        result = self.output.validate(
            tagged_text=(
                "Đây không phải PII; bộ kiểm thử chỉ ghi nhận mã 32/13/2026."
            ),
            entities=[],
            seed_pack=pack,
            focus_labels=["DATE"],
            max_entities=2,
        )

        by_type = {issue.type: issue for issue in result.issues}
        self.assertIn("hard_negative_meta_explanation", by_type)
        self.assertIn("decoy_context_unclear", by_type)
        self.assertIn("'dữ liệu lỗi'", by_type["decoy_context_unclear"].reason)

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

    def test_mixed_hard_negative_accepts_natural_decoy_repetition(self) -> None:
        pack = SeedPack(
            task_id="mixed-repeat",
            sample_type="hard_negative",
            hard_negative_mode="mixed_contrastive",
            context_frame=frame(),
            positive_entities=[PositiveEntitySeed(
                label="DATE",
                value="21/10/2026",
                semantic_role="appointment_date",
            )],
            decoys=[DecoySeed(
                strategy_id="date_as_invalid_calendar_value",
                target_label="DATE",
                value="32/13/2026",
                family="invalid_value",
                semantic_type="date_validation_test",
                negative_labels=["DATE"],
                required_context_cues=["test value"],
                forbidden_context_cues=["appointment date"],
            )],
        )
        text = (
            "The appointment date is <DATE>21/10/2026</DATE>. "
            "The test value 32/13/2026 was submitted and the same "
            "32/13/2026 was confirmed in the validation log."
        )
        entities = [{"label": "DATE", "value": "21/10/2026"}]

        result = self.output.validate(
            tagged_text=text,
            entities=entities,
            seed_pack=pack,
            focus_labels=["DATE"],
            max_entities=2,
        )

        self.assertTrue(result.valid, [issue.dict() for issue in result.issues])
        validate_seeded_contract(
            tagged_text=text,
            entities=entities,
            positive_entities=[pack.positive_entities[0].dict()],
            decoys=[pack.decoys[0].dict()],
            max_decoy_occurrences=3,
        )

    def test_schema_decoy_allows_field_coreference_in_later_paragraphs(self) -> None:
        pack = SeedPack(
            task_id="mixed-schema-repeat",
            sample_type="hard_negative",
            hard_negative_mode="mixed_contrastive",
            context_frame=frame(),
            positive_entities=[PositiveEntitySeed(
                label="MARITAL",
                value="Never married",
                semantic_role="employee_marital_status",
            )],
            decoys=[DecoySeed(
                strategy_id="marital_as_schema_field",
                target_label="MARITAL",
                value="MARITAL-STATUS-CODE-V9",
                family="schema_field",
                semantic_type="database_schema_field",
                negative_labels=["MARITAL"],
                required_context_cues=["schema field", "data field"],
                forbidden_context_cues=["marital status"],
            )],
        )
        text = (
            "The employee status is <MARITAL>Never married</MARITAL>. "
            "The schema field MARITAL-STATUS-CODE-V9 rejected the update."
            "\n\nThe MARITAL-STATUS-CODE-V9 field remains under review."
            "\n\nEngineers will patch the MARITAL-STATUS-CODE-V9 field."
        )
        entities = [{"label": "MARITAL", "value": "Never married"}]

        result = self.output.validate(
            tagged_text=text,
            entities=entities,
            seed_pack=pack,
            focus_labels=["MARITAL"],
            max_entities=2,
        )

        self.assertTrue(result.valid, [issue.dict() for issue in result.issues])

    def test_decoy_only_rejects_excessive_or_cross_paragraph_cueless_repetition(self) -> None:
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
                "\n\nNhân viên đọc lại chuỗi 32/13/2026."
            ),
            entities=[], seed_pack=pack, focus_labels=["DATE"], max_entities=2,
        )
        excessive = self.output.validate(
            tagged_text=(
                "Dữ liệu lỗi 32/13/2026 được đối chiếu với 32/13/2026, "
                "32/13/2026 và 32/13/2026."
            ),
            entities=[], seed_pack=pack, focus_labels=["DATE"], max_entities=2,
        )

        self.assertIn("decoy_context_unclear", {issue.type for issue in context_free.issues})
        self.assertIn("decoy_occurrence", {issue.type for issue in excessive.issues})

    def test_repeated_entity_annotation_is_case_sensitive_and_ignores_nested_substrings(self) -> None:
        missing = self.output._missing_repeated_annotations(
            "<PERSON>Nguyễn Thị Lan</PERSON> xác nhận Nguyễn Thị Lan."
        )
        nested = self.output._missing_repeated_annotations(
            "<PERSON>Lan</PERSON> gặp <PERSON>Nguyễn Thị Lan</PERSON>, "
            "còn LAN là mã nội bộ."
        )

        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].label, "PERSON")
        self.assertEqual(missing[0].value, "Nguyễn Thị Lan")
        self.assertIn("occurrence 2", missing[0].reason)
        self.assertEqual(nested, [])

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
