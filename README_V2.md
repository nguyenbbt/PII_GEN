# PII Data Factory v2

## 1. Tổng quan

PII Data Factory tạo dữ liệu PII tổng hợp phục vụ huấn luyện và kiểm thử mô hình
Named Entity Recognition. Luồng hiện tại là một modular monolith chạy từ CLI hoặc
FastAPI: đọc taxonomy, lập kế hoạch sample, sinh dữ liệu bằng Azure OpenAI, kiểm tra
deterministic và xuất JSON có offset. NoveltyGuard cùng LLM Judge/Repair là quality
checks độc lập; config mẫu bật Gemini 2.5 Pro Judge cho mọi candidate online.
Offline chỉ là smoke test và không được dùng làm dataset.

Hệ thống không sử dụng Faker để sinh positive entity. Value được chọn bằng Python
từ Value Bank ba ngôn ngữ; Generator chỉ viết nội dung và placeholder skeleton.

## 2. Phạm vi đã triển khai

```mermaid
flowchart TD
    ENTRY[CLI / FastAPI]
    TAX[Taxonomy Service]
    RUN[Run Orchestrator]
    COV[Coverage Controller]
    SEED[SampleType Router + Seed Factories]
    SEEDVAL[SeedPack Validator]
    TAXCTX[JSON Taxonomy Context Selector]
    GEN[Data Generator]
    DET[Technical Deterministic Validators]
    NOVELTY{NoveltyGuard enabled?}
    QUALITY{verifier.enabled?}
    JUDGE[LLM Judge]
    REPAIR[LLM Repair]
    RECHECK[Deterministic Re-check]
    FINALJUDGE[LLM Final Judge]
    FMT[Python Output Formatter]
    FILE[(gen_data/*.json)]

    ENTRY --> TAX --> RUN --> COV --> SEED --> SEEDVAL
    SEEDVAL --> TAXCTX --> GEN --> DET
    DET -->|critical risk| COV
    DET --> NOVELTY
    NOVELTY -->|duplicate| GEN
    NOVELTY -->|pass / disabled| QUALITY
    QUALITY -->|false| FMT
    QUALITY -->|true| JUDGE
    JUDGE -->|PASS| FMT
    JUDGE -->|FIXABLE| REPAIR --> RECHECK
    RECHECK -->|valid| FINALJUDGE
    RECHECK -->|invalid| GEN
    FINALJUDGE -->|PASS| FMT
    FINALJUDGE -->|not PASS| GEN
    JUDGE -->|REGENERATE| GEN
    JUDGE -->|REJECTED| COV
    FMT --> FILE
```

Luồng quality-first của config mẫu:

```text
Generator
  → Technical Deterministic Validators
  → Gemini 2.5 Pro Judge
  → optional Repair + re-check + final Judge
  → Python Output Formatter
  → gen_data
```

`verifier.enabled` điều khiển Judge/Repair. Cờ legacy
`validation.quality_checks_enabled` chỉ còn điều khiển NoveltyGuard và được migrate
sang `verifier.enabled` khi config cũ không khai báo `verifier`. Dù Judge tắt,
Formatter không bao giờ được bỏ qua technical validation.

## 3. Các module

### 3.1 Entry point và API

Các file chính:

- `pii_factory/main.py`: CLI, import taxonomy JSON và chạy đến khi run hoàn tất.
- `pii_factory/api.py`: FastAPI.
- `pii_factory/bootstrap.py`: nối các service và infrastructure adapter.

Các endpoint hiện có:

```text
GET  /
GET  /health
GET  /api/v1/taxonomies
GET  /api/v1/taxonomies/{version_id}
POST /api/v1/runs
POST /api/v1/taxonomies/{version_id}/runs
GET  /api/v1/runs/{run_id}
GET  /api/v1/runs/{run_id}/tasks
POST /api/v1/runs/{run_id}/generate?limit=5
GET  /api/v1/runs/{run_id}/events
```

Khi API khởi động, taxonomy canonical được đọc từ `PII_TAXONOMY_PATH` hoặc
`pii_taxonomy_rules.json`. `POST /api/v1/runs` nhận trực tiếp `RunConfig` và dùng
taxonomy version mặc định này.

`limit` là số sample đã accept cần trả về, không phải số request LLM tối đa. Một
sample có thể cần nhiều Generator attempt và, khi Verifier bật, nhiều Verifier
call.

Lỗi hạ tầng Verifier trả HTTP `503` mà không tiêu hao Generator attempt khi quality
gate được bật.

### 3.2 Taxonomy Service

`pii_factory/application/taxonomy_service.py`:

- đăng ký snapshot taxonomy có version;
- kiểm tra label của run;
- trả definition, rule và examples đúng cho task;
- phát event liên quan đến taxonomy.

`pii_factory/infrastructure/json_taxonomy.py` dùng Pydantic đọc trực tiếp
`pii_taxonomy_rules.json`. File này là nguồn taxonomy runtime duy nhất và hiện có
44 label; mỗi label phải có ít nhất ba example cho từng nhóm `POSITIVE`,
`PURE_NEGATIVE` và `HARD_NEGATIVE`.

Label từ tài liệu hard-negative bên ngoài không được thêm vào hệ thống. Chỉ label có
trong taxonomy snapshot của run được sử dụng.

### 3.3 Coverage Controller

`CoverageController` tạo `GenerationTask` theo:

- difficulty distribution;
- sample type distribution;
- optional constraints;
- exact seeded quota cho `short/medium/long`;
- sample structure được random có seed từ pool cấu hình;
- focus label và robin labels;
- entity/complexity limits;
- deterministic `random_seed`.

Preset độ dài:

| Structure | short | medium | long |
|---|---:|---:|---:|
| Contract | 80–120 từ, 3–5 content units | 150–230 từ, 6–9 units | 260–400 từ, 10–14 units |
| Chat | 80–120 từ, 6–8 lượt | 150–230 từ, 10–14 lượt | 260–400 từ, 16–22 lượt |
| Custom | 80–120 từ | 150–230 từ | 260–400 từ |

Các khoảng trên là mục tiêu để Generator hướng tới. Validator chỉ enforce cận dưới
`80/150/260`; sample dài hơn cận trên vẫn hợp lệ và không bị regenerate.

Planner dùng largest-remainder quota rồi shuffle bằng `random_seed`; với 10 mẫu và
distribution `0.2/0.5/0.3`, quota luôn là `2/5/3`.

Mỗi task ban đầu có `slot_no`. Nếu task hết Generator attempt, task thay thế:

- giữ nguyên `slot_no`;
- giữ focus labels, difficulty, sample type và constraints;
- nhận task ID, seed và context mới;
- tăng `replacement_no`.

Run chỉ `COMPLETED` khi mỗi slot có một sample `ACCEPTED`.

### 3.4 Value Bank và seed generation

`SampleTypeRouter` chọn factory:

- `positive`: lấy positive seed từ Value Bank theo `task.language` và label;
- `pure_negative`: dùng vocabulary an toàn, không có positive PII;
- `hard_negative/decoy_only`: chỉ có decoy không gắn tag;
- `hard_negative/mixed_contrastive`: có positive PII và decoy dễ nhầm.

`ValueBankEntityProvider` đọc lazy và cache file
`{language}_pii_value_pools.json` trong `value_bank.path`. Ba file hiện tại là
`vi_pii_value_pools.json`, `en_pii_value_pools.json` và
`de_pii_value_pools.json`; mỗi file phải có `version: 1`, object
`entity_values`, class hợp lệ, danh sách không rỗng, item `value` không rỗng và
`locale` khớp tên file.

Provider dùng đúng instance `random.Random` đã seed bằng `task.random_seed`.
Mọi entry trong file đều được giữ nguyên trong bộ nhớ, kể cả duplicate và các biến
thể chỉ khác hoa/thường. Khi một sample có nhiều entity, factory chỉ loại chuỗi đã
chọn nếu chuỗi mới giống hoàn toàn; `Visa`, `VISA` và `visa` vẫn là các value khác
nhau. Thiếu directory/language/class, class rỗng, JSON lỗi, version sai hoặc locale
sai đều tạo `value_bank_error` scope `SEEDS` và dừng retry vô ích.

`SeedPackValidator` kiểm tra:

- label thuộc taxonomy;
- positive value không trùng chính xác trong cùng completion; khác hoa/thường được
  xem là value khác;
- seed ngoài Value Bank vẫn qua format/mixed-locale checks cũ;
- seed từ Value Bank phải mang `format_variant=value_bank`, sau khi file đã được
  provider validate;
- decoy strategy tồn tại;
- decoy không va chạm positive seed hoặc label cấu trúc khác;
- required/forbidden context cues.

PERSON được lấy nguyên văn từ đúng class và language trong Value Bank; không dùng
Faker name. Quy tắc taxonomy phân tách địa chỉ như sau:

- `ADDRESS`: số nhà, đường, tòa, căn hộ hoặc phòng;
- `LOCATION`: phường, quận/huyện, tỉnh/thành phố hoặc quốc gia;
- `ZIP_CODE`: span bưu chính độc lập.

### 3.5 JSON Taxonomy Context Selector

`pii_factory/application/taxonomy_context.py` tạo context có cấu trúc cho từng task:

- label neo nhận `definition`, `rule` và đúng ba example tương ứng với
  `task.sample_type` đã được random;
- robin labels chỉ nhận `definition` và `rule`;
- nếu một nhóm có hơn ba example, selector dùng `task.random_seed` để chọn ba mẫu
  có thể tái lập;
- context thực tế được lưu trong `taxonomy_context_used`.

Đây là lookup deterministic theo label, không phải semantic retrieval và không dùng
embedding model.

### 3.6 Data Generator

Generator nhận:

- `GenerationTask`;
- validated `SeedPack`;
- `ContextFrame`;
- structured taxonomy guidance;
- reflection feedback của attempt trước.
- một `sample_structure` được code chọn bằng seeded random, chỉ từ pool
  `sample_structures` trong config;
- `length_target` có khoảng từ và content-unit/turn mục tiêu; chỉ cận dưới của số
  từ là bắt buộc.
- số entity bắt buộc bằng đúng số positive seed của task.

Ba structure được hỗ trợ:

- `contract`: fragment tài liệu doanh nghiệp/hành chính hoàn chỉnh;
- `chat`: hội thoại tự nhiên đúng hai người;
- `custom`: làm theo `custom_instruction` về bối cảnh và hình thức trình bày.

`custom_instruction` là untrusted data và không được ghi đè taxonomy, label,
sample type, seed/decoy, annotation hoặc output contract.
Không còn tầng random biến thể ẩn như `agreement_clause`, `company_notice`,
`friend_chat` hoặc `customer_support_chat`. Model nhận đúng type/instruction đã
được chọn từ file config.

Generator chỉ:

1. đổi positive value thành placeholder có đánh số theo class;
2. xây system/user prompt;
3. gọi Azure OpenAI để viết nội dung/skeleton;
4. parse `tagged_text` và `entities`;
5. chèn positive value bằng Python;
6. tạo `GenerationCandidate`;
7. tính token/cost của Generator;
8. phát `data.generated`.

Generator không quyết định candidate có được accept hay không.

Few-shot chỉ dạy ngữ nghĩa label và ranh giới annotation. Prompt version
`data-generator.v11.3.0` cấm sao chép hoặc paraphrase gần scenario, actor, action,
opening phrase, clause order và sentence structure của example.
Prompt cấm ghép seed thành danh sách dấu phẩy; mỗi entity phải có vai trò nghiệp vụ
và được phân bố qua nhiều câu/lượt trong cùng một sự kiện.

Ví dụ LLM nhìn thấy `<PERSON>[PERSON_1]</PERSON>` và
`entities[].value="[PERSON_1]"`. LLM không nhìn thấy value thật. Python thay đồng
bộ placeholder trong tagged text và metadata trước technical gate. Placeholder lạ
hoặc chưa resolve bị từ chối. Validator seed/tag hiện hữu vẫn quyết định placeholder
thiếu, lặp hoặc sai tag có hợp lệ hay không.

Output thô:

```json
{
  "tagged_text": "Chị <PERSON>Lò Thị Cẩy</PERSON> đã gửi hồ sơ.",
  "entities": [
    {
      "label": "PERSON",
      "value": "Lò Thị Cẩy"
    }
  ]
}
```

### 3.7 Deterministic Validators

Technical gate luôn chạy trước Formatter:

- schema/tag syntax;
- entity metadata khớp tag;
- allowed/focus labels;
- entity count;
- clean-text word count đạt tối thiểu của `length_target`; vượt cận trên được bỏ qua;
- positive seed xuất hiện nguyên văn; nếu cùng value lặp lại trong câu thì mọi
  occurrence đều phải có tag và một metadata entry tương ứng;
- decoy luôn untagged và có context cue;
- pure-negative/hard-negative structured PII scan;
- structured PII scan cho negative sample.
- `FewShotImitationGuard` che tagged entity rồi so sánh sequence/token n-gram với
  ba focus examples; candidate quá giống phải regenerate ngay cả khi Judge
  đang tắt.

Khi `quality_checks_enabled=true`, NoveltyGuard kiểm tra entity value giữa các sample
đã accept và sentence/entity novelty trong cùng run. Các occurrence lặp lại hợp lệ
trong cùng một sample không bị xem là duplicate annotation.

`DeterministicIssueRouter` ánh xạ kết quả theo route rõ ràng:

- `FIXABLE`: lỗi metadata, tag/span boundary, duplicate span hoặc missing annotation
  có thể sửa cục bộ; chuyển cho Judge/Repair nếu Verifier bật;
- `REGENERATE`: chỉ các lỗi nội dung/ngữ nghĩa không thể sửa cục bộ, seed/decoy
  semantics, novelty/imitation hoặc structured PII làm sai sample type;
- `REJECTED`: credential/real-PII risk severity `critical`.

Nếu seed value đã có trong text nhưng thiếu tag/metadata, route là `FIXABLE`. Nếu
seed value hoàn toàn vắng mặt, route là `REGENERATE` trực tiếp và không tốn một lượt
Judge để xác nhận lại lỗi contract mà code đã biết chắc.

Candidate cần regenerate tạo reflection feedback và quay lại Generator. Với lỗi scope
`SEEDS` hoặc `CONTEXT`, `RegenerationRouter` tạo seed/context mới theo đúng scope.

### 3.8 LLM Verifier

Verifier dùng cùng endpoint nhưng có model, prompt và sampling config riêng. Module
này được gọi khi `verifier.enabled=true`; config mẫu dùng `gemini-2.5-pro` cho Judge
và Repair trong khi Generator dùng `gemini-2.5-flash`.

Judge kiểm tra cận dưới độ dài, độ tự nhiên, duplicate skeleton, seed/decoy contract,
few-shot imitation và boundary `ADDRESS/LOCATION/ZIP_CODE`; không được reject vì vượt
cận trên. Với positive sample, Judge quét toàn văn theo `annotation_labels`. PII phát
sinh trong ngữ cảnh nhưng chưa có tag/metadata phải trả `FIXABLE/MISSING_ANNOTATION`;
Repair thêm tag và entity, deterministic gate kiểm tra lại, rồi Formatter tính lại
start/end offset. Deterministic issue là bằng chứng bắt buộc: Judge không được trả
`PASS` khi danh sách này còn issue.

Sau Repair, code bảo toàn mọi occurrence của positive seed và dựng lại metadata từ
tag trước khi re-check. Vì vậy nếu LLM vô tình bỏ tag ở lần nhắc lại thứ hai hoặc thứ
ba, code tự khôi phục thay vì regenerate cả nội dung.

Với Value Bank entry ghép nhiều taxonomy boundary, Repair được phép tách tag nhưng
không được đổi clean surface. Ví dụ
`34 Nguyễn Chí Thanh, Ba Đình, Hà Nội` có thể trở thành ADDRESS
`34 Nguyễn Chí Thanh` + LOCATION `Ba Đình, Hà Nội`. Preservation validator yêu cầu
chuỗi gốc vẫn xuất hiện nguyên vẹn, mọi phần chữ/số được gắn nhãn, các occurrence
được tách nhất quán và ít nhất một segment giữ label seed gốc.

Các trường điền cho người đọc như `[Tên Công ty]`, `[Ngày]`, `[Chức danh]`,
`[Tên Tài Xế]` được deterministic validator đánh dấu `template_artifact`. Repair
chỉ được thay vùng đó bằng mô tả chung không phải PII; mọi phần clean text khác phải
giữ nguyên. Candidate còn trường điền sau Repair bị từ chối.

Placeholder chuẩn luôn có ngoặc vuông và label viết hoa, ví dụ `[PERSON_1]`. Biến thể
thiếu ngoặc như `PERSON_1`, placeholder sai định dạng hoặc không thuộc seed contract
đều bị từ chối trước Verifier; code chỉ chèn Value Bank khi contract chính xác.

Response JSON của Generator/Judge/Repair chấp nhận JSON object thuần, JSON trong code
fence, hoặc object có phần giải thích bao quanh. JSON thực sự sai hoặc bị cắt vẫn được
retry và log ghi rõ dòng, cột cùng preview để chẩn đoán.

Judge chỉ lập quyết định và edit plan, không trực tiếp viết lại candidate:

```json
{
  "status": "FIXABLE",
  "score": 84,
  "issues": [
    {
      "type": "BOUNDARY",
      "severity": "low",
      "field": "tagged_text",
      "reason": "Boundary chứa dấu câu.",
      "suggested_fix": "Đưa dấu câu ra ngoài tag."
    }
  ],
  "edits": [
    {
      "action": "split_tag",
      "source_label": "ADDRESS",
      "source_value": "34 Nguyễn Chí Thanh, Ba Đình, Hà Nội",
      "segments": [
        {"label": "ADDRESS", "value": "34 Nguyễn Chí Thanh"},
        {"label": "LOCATION", "value": "Ba Đình, Hà Nội"}
      ],
      "reason": "Số nhà và tên đường là ADDRESS; quận và thành phố là LOCATION."
    }
  ]
}
```

Mỗi edit bắt buộc có `reason` giải thích bằng ngữ cảnh/taxonomy. Repair nhận cả
`issues` và `edits`, thực thi thay đổi cục bộ rồi dựng lại `entities`. Prompt Judge
có regression examples cho các lỗi đã gặp: trường điền `[Tên Công ty]`, `[Ngày]`,
`[Chức danh]`; PLATE/TICKET_ID/JOB_TITLE rõ ngữ cảnh nhưng chưa gán; khoảng trắng
trong tag; ADDRESS chứa LOCATION; và hard-negative tự giải thích “đây không phải PII”.

Trạng thái:

- `PASS`: `issues` bắt buộc rỗng;
- `FIXABLE`: chỉ được chứa issue severity `low`, gồm boundary/tag/metadata và missing
  annotation có thể xác định chắc chắn;
- `REGENERATE`: lỗi nội dung như thiếu/đối nghịch context, sai ngữ nghĩa không thể sửa
  cục bộ, hard-negative mơ hồ, thiếu tự nhiên hoặc rập khuôn;
- `REJECTED`: phải có issue `critical`, ví dụ credential/real-PII risk.

Luồng lỗi nhẹ:

```text
Judge FIXABLE
  → Repair sửa candidate trực tiếp
  → kiểm tra seed/decoy preservation
  → Deterministic re-check
  → Judge lần cuối
  → chỉ PASS mới được formatter
```

Repair không được thay:

- positive seed;
- decoy value hoặc số lần xuất hiện;
- focus labels;
- sample type;
- ý nghĩa task.

Response LLM là untrusted input. Pydantic dùng `extra = forbid`; response có field dư,
status mâu thuẫn hoặc schema sai được xem là lỗi hạ tầng. Candidate và taxonomy
guidance được đặt trong JSON envelope và không có quyền ghi đè system instructions.

### 3.9 Output Formatter

`pii_factory/application/formatting.py` hoàn toàn deterministic:

- parse XML-like tags;
- loại tag;
- kiểm tra unknown/nested/malformed tag;
- kiểm tra metadata khớp chính xác tagged spans;
- tính offset bằng Python Unicode code-point;
- `end` là exclusive;
- chặn empty/overlapping spans;
- xác minh `sample.text[start:end] == entity.text`.

Output cuối:

```json
[
  {
    "entities": [
      {
        "label": "PERSON",
        "start": 4,
        "end": 14,
        "text": "Lò Thị Cẩy"
      },
      {
        "label": "ETHNICITY",
        "start": 24,
        "end": 28,
        "text": "Cống"
      },
      {
        "label": "CARD_ISSUER",
        "start": 42,
        "end": 50,
        "text": "Agribank"
      }
    ],
    "text": "Chị Lò Thị Cẩy, dân tộc Cống, vay vốn tại Agribank.",
    "token_usage": {
      "input_tokens": 1240,
      "output_tokens": 380
    }
  }
]
```

Pure-negative:

```json
[
  {
    "entities": [],
    "text": "Bộ phận kỹ thuật đã chuyển biểu mẫu sang bước tiếp theo.",
    "token_usage": {
      "input_tokens": 980,
      "output_tokens": 215
    }
  }
]
```

### 3.10 Dataset writer

`JsonDatasetWriter`:

- ghi UTF-8;
- `ensure_ascii=false`;
- ghi file atomically trong cùng thư mục;
- sanitize `run_name` để không path traversal;
- ghi tiến độ vào
  `gen_data/{safe_run_name}-{run_id}.partial.json`;
- chỉ đổi thành
  `gen_data/{safe_run_name}-{run_id}.json`
  khi đủ `num_samples`.
- mỗi sample final có `token_usage.input_tokens` và
  `token_usage.output_tokens`, tính trên toàn bộ Generator/Judge/Repair call
  thuộc logical slot đó, kể cả retry đã phát sinh token.

Nếu replacement budget cạn:

- run chuyển `FAILED`;
- không công bố file `.json` hoàn chỉnh;
- file `.partial.json` đã có vẫn là artifact chẩn đoán, không phải dataset final.

## 4. Pydantic contracts

Các schema chính trong `pii_factory/domain/models.py`:

```text
RunConfig
TaxonomySnapshot / TaxonomyLabel
GenerationTask
SeedPack / PositiveEntitySeed / DecoySeed
GenerationCandidate
DeterministicValidationResult
SampleStructureConfig
VerificationIssue / VerifierDecision
RepairResult / VerificationTrace
FormattedEntity / FormattedSample
TokenUsage / PipelineTokenUsage
DataGenerationResult
```

`DataGenerationResult` giữ các field Generator cũ và bổ sung:

```text
verification_trace
formatted_sample
pipeline_token_usage
```

`verification_trace` là `null` khi `verifier.enabled=false`.

`token_usage` cũ là usage của Generator attempt đã accept.
`pipeline_token_usage.total` gồm:

- toàn bộ Generator attempt của logical slot;
- Judge call nếu Verifier bật;
- Repair call nếu phát sinh;
- final Judge call nếu phát sinh;
- call đã trả response có schema lỗi nhưng vẫn phát sinh token.

File dataset cuối chỉ bổ sung số lượng input/output token theo sample; không chứa
diagnostic, prompt, money cost hoặc taxonomy context. Summary in trên terminal vẫn
có tổng token của toàn run.

## 5. Run config

Ví dụ đầy đủ nằm tại `configs/run_config.example.json`.

Các field chính:

| Field | Ý nghĩa |
|---|---|
| `run_name` | Tên run và thành phần của output filename |
| `num_samples` | Số sample `ACCEPTED` bắt buộc |
| `language` | Value Bank language: `vi`, `en`, `de`; alias phổ biến được normalize |
| `minimum_per_label` | Coverage tối thiểu |
| `batch_size` | Số sample accept tối đa mỗi `generate_pending` |
| `focus_label` | Anchor label bắt buộc ở mọi sample |
| `robin_labels` | Pool label phụ trợ |
| `robin_selection` | Số robin label được random cho mỗi task |
| `focus_labels` | Chế độ legacy: pool focus labels |
| `difficulty_distribution` | Xác suất easy/medium/hard |
| `sample_type_distribution` | Xác suất positive/pure-negative/hard-negative |
| `sample_length_distribution` | Quota short/medium/long; đủ đúng ba key và tổng bằng 1 |
| `sample_structure` | Field đơn cũ, chỉ còn parse để tương thích; không dùng trong config mới |
| `sample_structures` | Nguồn structure duy nhất của config mới; mỗi sample chọn một phần tử bằng `random_seed` |
| `optional_constraint_distribution` | Xác suất `teen_code`, `light_typo`, `abbreviation` |
| `max_entities` | Mục tiêu entity khi generate; verifier có thể bổ sung annotation bị bỏ sót |
| `max_regenerate_attempts` | Số lần sinh lại trong cùng task |
| `max_task_replacements` | Số task thay thế tối đa cho mỗi slot |
| `value_bank` | `path`, seed-pack retry và unseeded-PII policy; bật `allow_additional_unseeded_pii` để verifier gắn nhãn PII phát sinh trong context |
| `hard_negative` | Mode, decoy count và focus limits |
| `complexity_limits` | Complexity budget theo sample type |
| `validation.quality_checks_enabled` | Cờ legacy cho NoveltyGuard; migrate sang Verifier nếu config không có `verifier` |
| `verifier.enabled` | Bật/tắt LLM Judge/Repair; config mẫu bật |
| `verifier.max_repairs_per_candidate` | Hiện chỉ cho phép `0` hoặc `1` |
| `parallel_generation` | Số worker, kích thước shard và số lần retry mỗi shard |
| `random_seed` | Tái lập task/seed selection |

Ví dụ:

```json
{
  "language": "vi",
  "value_bank": {
    "path": "PII_Value_Bank",
    "max_seed_pack_attempts": 5,
    "allow_additional_unseeded_pii": false
  }
}
```

Path tuyệt đối được dùng nguyên trạng; path tương đối được resolve từ working
directory của process. Config `faker`/`seed_generation` cũ được parse như alias
migration sang `value_bank`; `locale` cũ bị bỏ qua và không còn runtime Faker.

Trong anchor mode:

- `focus_label` luôn là positive entity trung tâm;
- `pure_negative` phải bằng `0`;
- hard-negative phải dùng `mixed_contrastive`;
- robin label được chọn ngẫu nhiên, không lặp trong cùng task.
- `1 + robin_selection.max_per_sample` không được vượt capacity; config sai bị từ
  chối thay vì âm thầm cắt số label.

Ví dụ structure:

```json
{
  "sample_structures": [
    {"type": "contract"},
    {"type": "chat"},
    {
      "type": "custom",
      "custom_instruction": "Viết dưới dạng email nghiệp vụ"
    },
    {
      "type": "custom",
      "custom_instruction": "Viết dưới dạng báo cáo sự việc"
    }
  ]
}
```

Mỗi task bốc ngẫu nhiên một phần tử trong `sample_structures`; đây không phải quota
cố định. Cùng `random_seed` tạo cùng chuỗi lựa chọn. Pool khai báo rỗng bị từ chối.
Config cũ không có pool vẫn được migrate từ field đơn `sample_structure`; config mới
không nên dùng field đơn này. Có thể thêm nhiều phần tử `custom` để mở rộng format;
`custom_instruction` bắt buộc với `custom` và bị từ chối với `contract` hoặc `chat`.
Các field validation cũ vẫn được parse để tương thích. Seed/tag/decoy/metadata/offset
và cận dưới độ dài không thể tắt. Cận trên của cả `short`, `medium`, `long` chỉ là
guidance và không tạo lỗi validation.

## 6. Azure/OpenAI settings

Không ghi API key vào source hoặc README. Online mode đọc:

```dotenv
OPENAI_API_KEY=<secret>
BASE_URL=https://<resource>.openai.azure.com
OPENAI_API_STYLE=auto
API_VERSION=2024-08-01-preview
MODEL=gemini-2.5-flash
GENERATOR_MODEL=gemini-2.5-flash
VERIFIER_MODEL=gemini-2.5-pro
DEPLOYMENT_NAME=gpt-4o
TEMPERATURE=0.2
MAX_TOKENS=6000

VERIFIER_TEMPERATURE=0.0
VERIFIER_JUDGE_MAX_TOKENS=4000
VERIFIER_REPAIR_MAX_TOKENS=2500

LLM_TIMEOUT_SECONDS=120
LLM_INFRA_MAX_RETRIES=3
INPUT_TOKEN_PRICE_PER_MILLION_USD=2.50
OUTPUT_TOKEN_PRICE_PER_MILLION_USD=10.00
GEN_DATA_DIR=gen_data
```

`GENERATOR_MODEL` được dùng cho Data Generator. `VERIFIER_MODEL` được dùng cho
Judge và Repair. `MODEL` là fallback tương thích khi một trong hai biến theo vai
trò bị thiếu.

`OPENAI_API_STYLE=auto` dùng Azure deployment route và header `api-key` cho
hostname Azure OpenAI native; với gateway tùy chỉnh, client dùng
`/chat/completions`, header Bearer và gửi model đúng theo vai trò trong request.
Có thể ép `azure` hoặc `openai` nếu gateway không thể nhận diện đúng bằng
hostname.

Khi Verifier bật, Verifier transport/contract failure:

- retry theo infrastructure policy;
- không tăng Generator `attempt_no`;
- reuse candidate đã lưu;
- không đưa raw critical candidate vào reflection.

Offline mode dùng `OfflineCompletionClient` và `OfflineVerifierClient`, không gọi
Azure nên không phát sinh hóa đơn. Artifact được ghi dưới
`gen_data/offline-smoke/`; CLI cảnh báo đây không phải dataset. Verifier cost bằng
`0`; Generator vẫn trả token count và `money_cost` mô phỏng để kiểm thử accounting.

## 7. Chạy hệ thống

### 7.1 Tạo virtual environment và cài project

```powershell
py -3.11 -m venv .venv
& '.\.venv\Scripts\Activate.ps1'
python -m pip install -e .
```

Trong workspace hiện tại executable nằm trong `.venv\bin`.

### 7.2 Chạy API offline

```powershell
& '.\.venv\bin\pii-factory.exe' --offline
```

Mở:

```text
http://127.0.0.1:8000/docs
```

### 7.3 Chạy taxonomy JSON offline để smoke test

```powershell
& '.\.venv\bin\pii-factory.exe' `
  --offline `
  --config configs\run_config.example.json
```

Chọn taxonomy hoặc output directory khác:

```powershell
& '.\.venv\bin\pii-factory.exe' `
  --offline `
  --taxonomy-json pii_taxonomy_rules.json `
  --samples 2 `
  --focus-labels TIME `
  --output-dir gen_data
```

CLI in:

- run status;
- output path;
- accepted sample count;
- input/output/total tokens;
- `money_cost`;
- diagnostics: candidate bị loại, deterministic/Verifier rejection, task replacement,
  verification outcome và issue-type counts;
- các formatted sample.

### 7.4 Chạy online

Bỏ `--offline`:

```powershell
& '.\.venv\bin\pii-factory.exe' `
  --config configs\run_config.example.json
```

CLI ghi progress log theo thời gian thực ra `stderr`, gồm số sample hiện tại/tổng
số, sample type, length bucket, label, generator attempt, LLM retry/latency,
deterministic validation route, verifier judge/repair và tiến độ accepted. Log không
ghi API key, header hoặc prompt. Diagnostic log có ghi tagged text, Value Bank value
đã bind, toàn bộ feedback và nội dung trước/sau verifier repair để phục vụ điều tra
local. Mặc định file UTF-8 có timestamp được tạo trong `--output-dir`; dùng
`--log-file <path>` nếu muốn chỉ định tên khác. Báo cáo JSON hoàn chỉnh vẫn được ghi
ra `stdout` sau khi run kết thúc và có thêm `diagnostic_log_path`.

Online mode gọi Generator và Verifier nên phát sinh chi phí.

Để chạy config theo nhiều shard song song và chỉ publish khi đủ toàn bộ sample:

```powershell
& '.\.venv\bin\pii-factory-parallel.exe' `
  --config configs\run_config.example.json
```

Mỗi shard dùng một `random_seed` độc lập. Runner kiểm tra số lượng, schema, offset
và duplicate text trước khi ghi một file JSON hợp nhất vào `gen_data`.

## 8. State và events

Run status:

```text
CREATED → RUNNING → COMPLETED
                  └→ FAILED
```

Task status:

```text
CREATED
→ GENERATING
→ GENERATED
→ VALIDATING
→ VERIFYING
→ optional REPAIRING
→ FORMATTING
→ ACCEPTED

hoặc REJECTED → replacement task
```

Các event chính:

```text
run.created
generation.task.created
seed.validated / seed.rejected
data.generated
data.deterministic.validated
data.generation.rejected
data.verification.judged / rejected  # chỉ khi verifier.enabled=true
sample.formatted
sample.accepted
generation.task.replaced
generation.task.rejected
run.completed / run.failed
```

Repository và EventBus hiện là thread-safe in-memory adapter. Restart process không
khôi phục run đang chạy.

## 9. Kiểm thử

Project dùng `unittest`, không yêu cầu pytest:

```powershell
& '.\.venv\bin\python.exe' -m unittest discover -s tests -q
```

Test suite bao phủ:

- taxonomy JSON và structured context selector;
- distributions/focus/robin config;
- contract/chat/custom sample structures;
- Value Bank provider/seed/hard-negative strategies;
- reproducibility theo seed và chống trùng chính xác trong sample, không collapse
  biến thể hoa/thường;
- placeholder đánh số, replacement và unknown-placeholder rejection;
- deterministic validators;
- novelty/diversity;
- Verifier contract và bốn verdict;
- Judge → Repair → re-check → final Judge;
- repair thay seed/decoy bị chặn;
- verifier infrastructure retry không tăng Generator attempt;
- replacement task và replacement budget;
- pipeline cost;
- Unicode/emoji offsets;
- offset sau khi chèn Value Bank value;
- partial/final JSON writer;
- offline end-to-end.
- Verifier disabled vẫn giữ technical gate.
- exact length quota, minimum word target, bỏ qua upper-bound và entity density;
- verifier bổ sung missing annotation rồi formatter tính lại offset;
- PERSON thuần Việt và ADDRESS/LOCATION/ZIP_CODE boundary.

## 10. Giới hạn hiện tại và hướng production

Chưa triển khai:

- PostgreSQL persistence;
- RabbitMQ/Redis Streams;
- transactional outbox và DLQ;
- distributed worker deployment;
- vector database/embedding retrieval;
- human review UI;
- authentication/RBAC;
- semantic duplicate detection bằng embedding;
- resume run sau khi process restart.

Các thành phần trên là hướng production tương lai, không phải dependency runtime của
repository hiện tại.

Rủi ro còn lại:

- chất lượng phụ thuộc taxonomy, model và prompt;
- Generator và Verifier vẫn có thể có correlated model-family bias;
- LLM Verifier tùy chọn không thay thế human review cho dataset quan trọng;
- bật Verifier làm tăng latency/cost ở sample cần Repair;
- real-PII detection rule-based không thể đảm bảo tuyệt đối;
- JSON offset dùng Python code point; consumer dùng UTF-16 phải tự chuyển index.

## 11. Tiêu chí hoàn thành một run

Run chỉ được xem là thành công khi:

- có đúng `num_samples` sample `ACCEPTED`;
- mọi sample deterministic-valid;
- nếu Verifier bật, mọi sample có Judge cuối `PASS`;
- mọi formatted span thỏa `text[start:end] == entity.text`;
- không có task slot bị cạn replacement budget;
- file `.json` final đã được publish atomically;
- token và money cost được tổng hợp cho toàn bộ logical slot.
