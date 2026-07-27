# Entry point → Quality Gate → Output Formatter

Tài liệu này giữ tên file cũ để tương thích liên kết. Pipeline hiện không còn kết
thúc tại Data Generator; tài liệu kiến trúc đầy đủ và cập nhật nằm trong
`README_V2.md`.

Phân hệ hiện bao gồm:

```text
FastAPI entry point
  → Taxonomy Service (versioned taxonomy snapshot)
  → Run Orchestrator
  → Coverage Controller tạo GenerationTask
  → DiversityPlanner phân bổ context/style theo seeded quota
  → SampleTypeRouter
      → positive: Faker PositiveSeedFactory
      → pure_negative: safe vocabulary, không gọi Faker
      → hard_negative: taxonomy-aware decoy, không gọi Faker ở mode decoy_only
  → SeedPackValidator + rule-based ContextFrameSelector
  → JSON Taxonomy Context Selector lấy guidance theo task
  → SampleStructure: contract | chat | custom
  → Data Generator gọi Azure OpenAI và bắt buộc dùng seed nguyên văn
  → deterministic validation (seed/decoy/tag/metadata/structured PII)
  → data.generated event (raw candidate)
  → optional NoveltyGuard + LLM Judge/Repair khi quality checks bật
  → Python Output Formatter
  → gen_data/*.json
```

Các payload xuyên suốt luồng là Pydantic models trong [models.py](</D:/Project/Pii llm gen/pii_factory/domain/models.py>). [Taxonomy Service](</D:/Project/Pii llm gen/pii_factory/application/taxonomy_service.py>) version hóa taxonomy và chỉ cung cấp context theo focus labels. Service nghiệp vụ chỉ phụ thuộc Protocol trong [ports.py](</D:/Project/Pii llm gen/pii_factory/ports.py>); `InMemoryRepository` và `InMemoryEventBus` là adapter local có thể thay bằng PostgreSQL/RabbitMQ.

`JsonTaxonomyParser` dùng Pydantic đọc `pii_taxonomy_rules.json`; test xác nhận đủ
44 label và ba nhóm example cho từng label.

## Chạy API offline

Chế độ này dùng client deterministic, không gọi Azure và không phát sinh hóa đơn;
`money_cost` của Generator là giá trị mô phỏng để kiểm thử accounting.

```powershell
& '.\.venv\bin\pii-factory.exe' --offline
```

API tự đăng ký `pii_taxonomy_rules.json` khi khởi động. Tạo run bằng body config
trực tiếp:

```json
POST /api/v1/runs
{
  "run_name": "vi_data_001",
  "num_samples": 20,
  "language": "vietnamese",
  "minimum_per_label": 1,
  "batch_size": 5,
  "focus_label": "PERSON",
  "robin_labels": ["PHONE", "EMAIL", "ADDRESS", "DATE", "TIME"],
  "robin_selection": {"min_per_sample": 1, "max_per_sample": 2},
  "difficulty_distribution": {"easy": 0.2, "medium": 0.6, "hard": 0.2},
  "sample_type_distribution": {"positive": 0.75, "pure_negative": 0.0, "hard_negative": 0.25},
  "sample_structure": {"type": "contract", "custom_instruction": null},
  "optional_constraint_distribution": {"teen_code": 0.05, "light_typo": 0.05, "abbreviation": 0.1},
  "max_entities": {"easy": 2, "medium": 4, "hard": 6},
  "max_regenerate_attempts": 2,
  "faker": {"locale": "vi_VN", "max_seed_pack_attempts": 5, "allow_additional_unseeded_pii": false},
  "hard_negative": {"mode": "mixed_contrastive", "min_decoys": 1, "max_decoys": 1, "max_focus_labels": 3, "unsupported_label_policy": "rebuild_task"},
  "complexity_limits": {"positive": 4, "pure_negative": 2, "hard_negative": 4},
  "validation": {"quality_checks_enabled": false},
  "verifier": {"max_repairs_per_candidate": 1},
  "random_seed": 174
}
```

Sau đó gọi `POST /api/v1/runs/{run_id}/generate?limit=2`. Kết quả có
`sample_structure`, `seed_pack_id`, `context_frame_id`, `diversity_profile`,
`tagged_text`, `entities`, technical validation result, taxonomy context, token usage và
`money_cost` (USD). Novelty và Verifier chỉ chạy khi
`validation.quality_checks_enabled=true`.

`RegenerationRouter` định nghĩa rõ semantics: `TEXT` giữ seed và frame, `SEEDS` tạo
seed pack ID mới, `CONTEXT` giữ seed pack/entity nhưng chọn frame mới. Pipeline hiện
kết thúc tại Python Output Formatter và file JSON; LLM Verifier là quality gate tùy
chọn.

## Chạy Azure OpenAI

Không thêm `--offline`; ứng dụng đọc `.env` đã có `OPENAI_API_KEY`, `BASE_URL`,
`API_VERSION`, `DEPLOYMENT_NAME`, `GENERATOR_MODEL`, `VERIFIER_MODEL`,
`TEMPERATURE`, `MAX_TOKENS`, và rates chi phí. `MODEL` là fallback tương thích.
Chỉ endpoint `/generate` mới gọi model.

## Chạy taxonomy JSON đến sample đầu tiên

```powershell
& '.\.venv\bin\pii-factory.exe' --taxonomy-json pii_taxonomy_rules.json --samples 2 --focus-labels TIME --hard-negative
```

Chạy bằng file config distribution:

```powershell
& '.\.venv\bin\pii-factory.exe' `
  --config configs\run_config.example.json
```

Ở anchor mode, `focus_label` là label bắt buộc và luôn là positive entity trung tâm của mọi sample. `robin_labels` là pool label phụ trợ; `robin_selection.min_per_sample` và `max_per_sample` điều khiển số robin được chọn ngẫu nhiên, không lặp, cho từng task. `minimum_per_label` chỉ đặt ngưỡng tối thiểu cho anchor; vì anchor xuất hiện ở mọi task nên `num_samples: 20` và `minimum_per_label: 3` là hợp lệ. `pure_negative` phải bằng `0`; hard-negative phải dùng `mixed_contrastive` để anchor vẫn là positive entity. `random_seed` làm lựa chọn robin tái lập được.

`focus_labels` dạng mảng vẫn được hỗ trợ như chế độ legacy: đây là pool để chọn primary label ngẫu nhiên và không được dùng đồng thời với `focus_label`/`robin_labels`. Nếu bỏ cả hai chế độ, hệ thống dùng toàn bộ taxonomy. Mọi mã label phải khớp chính xác với `pii_taxonomy_rules.json`.

Ở mode legacy `hard_negative.mode = "decoy_only"`, `focus_labels` mô tả nhãn taxonomy mà decoy dễ bị nhận nhầm. Output phải chứa decoy một lần, hoặc tối đa hai lần khi lần lặp phục vụ xác nhận, đính chính hay đối chiếu tự nhiên; mọi lần xuất hiện đều có cue giải thích trong cùng câu, không có XML tag và output luôn trả `entities: []`. Ở mode `mixed_contrastive`, positive seed được gắn đúng label theo nghĩa trong ngữ cảnh, kể cả khi bề mặt dễ nhầm với label hoặc nghĩa khác; decoy vẫn không được gắn tag. Registry có tối thiểu ba strategy thuộc ít nhất hai family cho đủ 44 mã taxonomy; validator loại seed va chạm với định dạng PII có cấu trúc trước khi gọi LLM.

## Audit độ đa dạng offline

```powershell
& '.\.venv\bin\python.exe' scripts\audit_diversity.py `
  --config configs\run_config.example.json `
  --taxonomy-json pii_taxonomy_rules.json `
  --samples 100
```

Report gồm entity unique ratio, exact/near sentence duplicate rate, phân phối context frame, hard-negative strategy, constraint và entity format variant. Audit luôn ép `novelty_mode=audit` và không gọi Azure.
