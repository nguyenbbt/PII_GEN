# Value Bank Migration – Commit Notes

## 1. Mục tiêu thay đổi

Thay cơ chế sinh positive entity value bằng Faker sang lấy value từ Value Bank ba
ngôn ngữ trong `PII_Value_Bank`, đồng thời giữ nguyên pipeline tạo sample, schema
đầu ra, deterministic validators và Output Formatter hiện có.

Các nguyên tắc của thay đổi:

- Python chọn entity value, không để LLM tự sinh value.
- Value được chọn theo đúng `task.language` và taxonomy class.
- LLM chỉ viết nội dung và placeholder skeleton.
- Python chèn value thật trước khi chạy validator và formatter.
- Random tiếp tục dùng `random.Random` đã được seed bởi pipeline.
- Không chỉnh sửa nội dung các file JSON trong `PII_Value_Bank`.

## 2. Luồng cũ

```text
GenerationTask
  → FakerEntityProvider
  → sinh positive entity value
  → tạo SeedPack chứa value thật
  → gửi value thật cho LLM
  → LLM viết nội dung và chèn value vào tagged_text/entities
  → deterministic validators
  → OutputFormatter
```

Hạn chế chính của luồng cũ:

- Runtime phụ thuộc Faker.
- Locale của provider được cấu hình riêng bằng `faker.locale`.
- LLM nhìn thấy và trực tiếp chèn entity value.
- Logic sinh format/value nằm rải trong `entity_variants.py`.
- Khó sử dụng một nguồn value chung đã được kiểm soát cho cả ba ngôn ngữ.

## 3. Luồng mới

```text
GenerationTask
  → ValueBankEntityProvider
  → đọc {language}_pii_value_pools.json
  → chọn value theo language + taxonomy class bằng seeded random
  → tạo SeedPack chứa value thật
  → chuyển value thành placeholder [LABEL_N] trong prompt
  → LLM viết nội dung và placeholder skeleton
  → Python thay placeholder bằng value trong SeedPack
  → deterministic validators hiện có
  → OutputFormatter hiện có
```

Ví dụ prompt gửi tới LLM:

```text
<PERSON>[PERSON_1]</PERSON> gặp <PERSON>[PERSON_2]</PERSON>.
```

Sau bước Python binding:

```text
<PERSON>Nguyễn An</PERSON> gặp <PERSON>Lê Bình</PERSON>.
```

Value thật không được đưa vào prompt Generator. Placeholder được đánh số riêng theo
từng class:

```text
[PERSON_1]
[PERSON_2]
[EMAIL_1]
[EMAIL_2]
```

## 4. Value Bank provider

Provider mới nằm tại:

```text
pii_factory/application/value_bank.py
```

Class chính:

```text
ValueBankEntityProvider
```

Provider thực hiện:

- Resolve đường dẫn Value Bank từ config.
- Lazy-load và cache từng file ngôn ngữ.
- Chọn file theo mẫu `{language}_pii_value_pools.json`.
- Chọn đúng array theo taxonomy class.
- Dùng chính `random.Random` được pipeline truyền vào.
- Giữ nguyên mọi entry trong bộ nhớ để duplicate và biến thể hoa/thường trong source
  vẫn tham gia đúng xác suất chọn đã cấu hình.
- Không ghi hoặc sửa lại file Value Bank.
- Hỗ trợ `excluded_values` để hạn chế lấy trùng trong cùng sample.
- Gắn `format_variant="value_bank"` cho positive seed.

Ba file hiện tại:

```text
PII_Value_Bank/vi_pii_value_pools.json
PII_Value_Bank/en_pii_value_pools.json
PII_Value_Bank/de_pii_value_pools.json
```

Mỗi file hiện có đủ 44 taxonomy class.

## 5. Validation của Value Bank

Provider phát lỗi rõ ràng cho các trường hợp:

- Thư mục Value Bank không tồn tại hoặc không phải directory.
- Không có file cho ngôn ngữ được yêu cầu.
- Language code không hợp lệ.
- File không phải JSON hợp lệ.
- Root JSON không phải object.
- `version` không phải version được hỗ trợ, hiện là `1`.
- Thiếu object `entity_values`.
- Class name không hợp lệ.
- Class không phải array.
- Class không có item.
- Item không phải object.
- `value` không phải string hoặc chỉ chứa whitespace.
- `locale` không khớp ngôn ngữ của file.
- Class không tồn tại.
- Không còn value chưa sử dụng sau khi áp dụng `excluded_values`.

Lỗi Value Bank được pipeline chuyển thành:

```text
ValidationIssue.type = "value_bank_error"
ValidationIssue.scope = "SEEDS"
```

Lỗi cấu hình hoặc lỗi file Value Bank là lỗi terminal của seed planning; pipeline
không lặp lại cùng một lỗi file qua toàn bộ seed retry budget.

## 6. Placeholder binding

Logic placeholder mới nằm tại:

```text
data_generator_worker/placeholders.py
```

Các hàm chính:

```text
build_placeholder_bindings()
placeholder_entities()
replace_entity_placeholders()
```

`placeholder_entities()` tạo bản sao metadata cho prompt và chỉ thay trường `value`
bằng placeholder. `SeedPack` gốc vẫn giữ value thật.

`replace_entity_placeholders()` thay đồng bộ placeholder trong:

- `tagged_text`;
- `entities[].value`.

Replacement chỉ chạy một lần. Value được lấy từ Value Bank không thể vô tình kích
hoạt một vòng replacement thứ hai.

Placeholder lạ hoặc chưa resolve bị từ chối. Placeholder cùng class được phép lặp;
sau replacement, mọi occurrence phải có tag và một metadata entry tương ứng. Các lỗi
như thiếu positive entity, occurrence lặp nhưng chưa tag, sai tag hoặc metadata không
khớp được chuyển cho deterministic validators xử lý.

## 7. Reproducibility và xử lý value lặp

Pipeline tiếp tục khởi tạo:

```python
rng = random.Random(task.random_seed)
```

Provider sử dụng trực tiếp instance `rng` này. Không tạo random generator riêng và
không dùng global random state.

Với cùng:

- Value Bank content;
- Value Bank path;
- language;
- taxonomy class;
- task seed;
- thứ tự gọi provider;

kết quả chọn value có thể tái lập.

Trong một sample, `PositiveSeedFactory` truyền toàn bộ value đã dùng qua
`excluded_values`. Việc so sánh chỉ dùng chuỗi chính xác:

```text
value
```

Do đó các value chỉ khác hoa/thường được xem là các lựa chọn khác nhau. Ví dụ
`Visa`, `VISA` và `visa` đều được giữ lại. Chỉ chuỗi hoàn toàn giống một value đã
chọn mới bị loại khỏi lần chọn tiếp theo trong cùng sample.

Value vẫn có thể xuất hiện lại ở sample khác. Đây là hành vi dự kiến vì mỗi sample
có random state riêng và Value Bank là một tập hữu hạn.

## 8. Config mới

Config mới:

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

Quy tắc resolve path:

- Path tuyệt đối được dùng nguyên trạng.
- Path tương đối được resolve từ working directory của process.

Language được hỗ trợ bởi Value Bank hiện tại:

```text
vi
en
de
```

Các alias phổ biến được normalize:

```text
vietnamese, vi-vn, vi_vn → vi
english, en-us, en_us    → en
german, de-de, de_de     → de
```

Config key `faker` và `seed_generation` cũ chỉ còn được parse như alias migration
sang `value_bank`. Field `locale` cũ bị bỏ qua. Config mới không nên tiếp tục dùng
key `faker`.

## 9. Phần Faker đã được loại bỏ

Đã xóa:

```text
pii_factory/application/entity_variants.py
tests/test_entity_variants.py
```

Đã loại bỏ khỏi `pyproject.toml`:

```text
Faker==37.3.0
```

Các thành phần cũ đã bị loại bỏ cùng file `entity_variants.py`:

- `FakerEntityProvider`;
- `VietnameseAddressProvider`;
- các controlled factory sinh phone/email/date/time/card/address;
- dữ liệu hành chính và street hard-code phục vụ Faker;
- khởi tạo và seed Faker instance.

Runtime entity generation mới không import hoặc cần cài Faker.

Tên `faker` chỉ còn xuất hiện trong parser config để migrate config cũ và trong tài
liệu giải thích deprecation.

## 10. Validator và output được giữ nguyên

Các bước sau vẫn được giữ trong pipeline:

- `SeedPackValidator`;
- `DeterministicOutputValidator`;
- validation tag/metadata;
- positive seed preservation;
- occurrence-level entity validation; cùng value được phép lặp trong một sample nếu
  mọi occurrence đều có tag và metadata;
- decoy validation;
- hard-negative context cue validation;
- pure-negative structured PII scan;
- `FewShotImitationGuard`;
- optional NoveltyGuard;
- optional LLM Judge/Repair;
- `OutputFormatter`;
- Unicode code-point offsets;
- end-exclusive offset;
- kiểm tra `sample.text[start:end] == entity.text`.

Positive seed có `format_variant="value_bank"` được xem là đã qua format/locale
validation của provider. Positive seed không đến từ Value Bank vẫn dùng các
format/mixed-locale checks cũ.

Value replacement diễn ra trước deterministic validation và Output Formatter, nên
offset luôn được tính trên value thật, không tính trên placeholder.

Schema dataset cuối không thay đổi:

```json
{
  "entities": [
    {
      "label": "PERSON",
      "start": 4,
      "end": 13,
      "text": "Nguyễn An"
    }
  ],
  "text": "Chị Nguyễn An đã gửi hồ sơ."
}
```

## 11. Prompt thay đổi

Prompt version:

```text
data-generator.v9.0.0
→ data-generator.v11.3.0
```

Prompt mới yêu cầu:

- Dùng đúng mọi positive placeholder.
- Không sửa, dịch hoặc normalize placeholder.
- Không tự tạo final entity value.
- Trả cùng placeholder trong tag và `entities[].value`.
- Không tạo thêm PII.

LLM vẫn chịu trách nhiệm:

- nội dung tự nhiên;
- bối cảnh;
- cấu trúc contract/chat/custom;
- XML-like tag placement;
- hard-negative semantic realization;
- decoy integration.

Python chịu trách nhiệm:

- chọn value;
- chống trùng trong sample;
- placeholder numbering;
- chèn value;
- validation;
- offset.

## 12. File đã thêm

```text
COMMIT_VALUE_BANK.md
data_generator_worker/placeholders.py
pii_factory/application/value_bank.py
tests/test_placeholder_replacement.py
tests/test_value_bank.py
```

## 13. File đã sửa

### Runtime và domain

```text
data_generator_worker/prompt.py
data_generator_worker/worker.py
pii_factory/application/seed_generation.py
pii_factory/application/services.py
pii_factory/application/validators.py
pii_factory/domain/models.py
pyproject.toml
```

### Config

```text
configs/run_config.example.json
```

### Test hiện có

```text
tests/test_hard_negative_diversity.py
tests/test_pipeline_seed_strategies.py
tests/test_prompt.py
tests/test_prompt_diversity.py
tests/test_retry_router.py
tests/test_run_config.py
tests/test_seed_generation.py
tests/test_worker.py
```

### Tài liệu

```text
README.md
README_V2.md
ENTRY_TO_DATA_GENERATOR.md
DATA_GENERATOR_WORKER.md
reports/diversity-offline.md
```

## 14. File đã xóa

```text
pii_factory/application/entity_variants.py
tests/test_entity_variants.py
```

## 15. Test đã bổ sung

`tests/test_value_bank.py` kiểm tra:

- Ba ngôn ngữ đều cover đủ taxonomy class.
- Chọn đúng language và class.
- Reproducibility với cùng seed.
- `excluded_values` chỉ chống trùng chính xác trong sample.
- Duplicate source và biến thể hoa/thường được giữ nguyên; file không bị sửa.
- Thiếu directory.
- Thiếu language file.
- Thiếu class.
- Class rỗng.
- JSON không hợp lệ.
- Locale mismatch.

`tests/test_placeholder_replacement.py` kiểm tra:

- Nhiều placeholder cùng class.
- Numbering theo class.
- Replacement đồng bộ tagged text và entity metadata.
- Existing validators chấp nhận output sau replacement.
- Unknown placeholder bị từ chối.
- Existing seed validator từ chối placeholder thiếu hoặc occurrence chưa được tag;
  occurrence lặp đã tag đầy đủ được chấp nhận.
- Offset được tính đúng sau khi chèn Unicode và emoji value.

Các test cũ về seed generation, hard-negative, prompt, worker, retry và config đã
được chuyển khỏi Faker sang Value Bank.

## 16. Kết quả test

Lệnh đã chạy:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Kết quả hiện tại sau khi merge và bổ sung regression tests:

```text
179 passed
```

Offline diversity audit:

```text
Requested samples: 100
Generated samples: 100
Prompt version: data-generator.v11.3.0
Random seed: 174
```

Online Azure/OpenAI đã được chạy thủ công bằng config local. Log lần chạy này giúp
phát hiện response JSON không ổn định ở Repair và lỗi bỏ tag khi cùng seed xuất hiện
nhiều lần; cả hai trường hợp đã có parser/log chẩn đoán và regression test tương ứng.

## 17. Structure và Verifier hardening sau online test

Config mới dùng `sample_structures` làm nguồn structure duy nhất. Code chọn một
phần tử trong chính pool này bằng seeded random và đưa nguyên `type` cùng
`custom_instruction` sang prompt. Các biến thể ẩn từng được tự chọn cho contract/chat
đã bị bỏ khỏi task planner. Field đơn `sample_structure` chỉ còn để migrate config cũ;
pool khai báo rỗng bị từ chối.

Generator prompt cấm để lại trường điền cho người đọc như `[Tên Công ty]`, `[Ngày]`,
`[Chức danh]` hoặc `[Tên Tài Xế]`. Deterministic validator phát hiện các artifact này
và route sang `FIXABLE`.

Khi `allow_additional_unseeded_pii=true`, `annotation_labels` dùng toàn bộ taxonomy
của run. Generator và Verifier vì vậy có thể gắn các class rõ ngữ cảnh nhưng không
thuộc positive seed hiện tại, ví dụ PLATE, TICKET_ID và JOB_TITLE. Code có detector
deterministic cho DATE, PLATE, TICKET_ID và các structured class sẵn có; Verifier
thực hiện semantic pass cho các class còn lại.

Judge response có thêm `edits`. Mỗi edit bắt buộc có `reason` và dùng một action cục
bộ như `add_tag`, `split_tag`, `adjust_tag_boundary`,
`replace_template_artifact` hoặc `sync_entities`. Cả `issues` và `edits` được truyền
sang Repair và được log để chẩn đoán.

Repair được phép tách một composite Value Bank seed theo taxonomy boundary, ví dụ
ADDRESS street detail + LOCATION administrative geography, với các điều kiện:

- clean seed surface không đổi;
- mọi phần chữ/số được span hợp lệ bao phủ;
- ít nhất một segment giữ label seed gốc;
- mọi occurrence của seed dùng cùng cách phân tách;
- metadata khớp chính xác tagged spans.

Hard-negative feedback hiện ghi cả cue bắt buộc và context quan sát được. Dạng text
tự giải thích “đây không phải PII” bị từ chối để buộc model thể hiện contrast bằng
ngữ cảnh nghiệp vụ tự nhiên.

Các file bổ sung trong đợt hardening:

```text
pii_factory/application/diversity.py
pii_factory/application/verification.py
pii_factory/infrastructure/clients.py
pii_factory/bootstrap.py
tests/test_deterministic_validators.py
tests/test_quality_contracts.py
tests/test_verifier_service.py
```

## 18. Vấn đề hiện có trong Value Bank nhưng chưa sửa

Theo yêu cầu, không file Value Bank nào được tự động chỉnh sửa.

### Duplicate trong cùng class

Số entry duplicate sau khi normalize bằng `strip().casefold()`:

| Language | Duplicate entries |
|---|---:|
| `vi` | 190 |
| `en` | 28 |
| `de` | 12 |

Provider giữ nguyên các entry này trong bộ nhớ khi chọn. Các biến thể hoa/thường
được coi là value khác nhau; duplicate hoàn toàn giống nhau vẫn giữ nguyên trọng số
lấy mẫu của dữ liệu nguồn.

Các class có duplicate đáng chú ý:

- `vi`: `GENDER`, `USERNAME`, `PIN`, `CARD_ISSUER`, `WALLET`, `DATE`, `TIME`,
  `MARITAL`, `RELIGION`, `ETHNICITY`, `TRADE_UNION`, `NATIONALITY`,
  `MEDICAL_INFO`.
- `en`: `PASSPORT`, `PIN`, `USERNAME`, `ZIP_CODE`.
- `de`: `DATE`, `PASSPORT`, `PREFIX`, `USERNAME`.

### Value trùng giữa nhiều class

Số normalized value xuất hiện ở nhiều class:

| Language | Cross-class values |
|---|---:|
| `vi` | 51 |
| `en` | 48 |
| `de` | 35 |

Ví dụ:

- `vi`: một số value xuất hiện đồng thời trong `EMPLOYEE_ID`, `PIN`,
  `TICKET_ID`; một số value xuất hiện trong cả `CVV` và `PIN`.
- `en`: nationality/ethnicity có nhiều surface form giống nhau; một số số định
  danh xuất hiện ở nhiều class.
- `de`: nationality/ethnicity có nhiều surface form giống nhau; một số date xuất
  hiện trong cả `DATE` và `BIRTHDATE`.

Provider chống chọn cùng exact value trong một sample, nhưng overlap semantic giữa
các class vẫn là vấn đề dữ liệu cần review riêng.

### Whitespace đầu/cuối

Phát hiện một value trong English `ZIP_CODE` có leading whitespace:

```text
" Docklands VIC 3008"
```

File không được sửa trong commit này.

### Tập value hữu hạn

Value Bank chỉ hạn chế chọn lại cùng chuỗi chính xác trong một sample và không đảm
bảo uniqueness trên toàn run. Audit 100 sample với `PERSON` làm anchor có unique
ratio thấp nhất là `58%`.
Nếu cần uniqueness toàn dataset, phải bổ sung run-level allocation policy trong một
thay đổi riêng.

### Trạng thái structural validation

Ba file hiện tại:

- Parse JSON thành công.
- Có `version: 1`.
- Có đủ 44 class.
- Không có class rỗng.
- Không có locale mismatch.
- Không có item value rỗng.

## 19. Lưu ý Git trước khi push

`PII_Value_Bank/` được `.gitignore` loại khỏi commit vì dữ liệu mẫu có thể kích hoạt
GitHub secret scanning. Không chạy `git add -f PII_Value_Bank`.

Người clone repository phải tự đặt Value Bank hợp lệ vào đường dẫn được cấu hình,
mặc định là `PII_Value_Bank/`, hoặc đổi:

```json
{
  "value_bank": {
    "path": "D:/data/PII_Value_Bank"
  }
}
```

Nếu directory/file ngôn ngữ/class bị thiếu hoặc JSON sai, pipeline dừng với
`value_bank_error`; hệ thống không âm thầm quay lại Faker.

Không commit các artifact môi trường local:

```text
.tmp/
.venv/
.venv310/
```

Checklist đề xuất:

```powershell
git status --short
git diff --check
.\.venv\Scripts\python.exe -m pytest tests -q
git add COMMIT_VALUE_BANK.md
```

Sau đó review danh sách file staged trước khi commit.

## 20. Làm cứng hợp đồng JSON của verifier

Verifier đôi khi trả `occurrence: 0` với ý nghĩa “lần xuất hiện đầu tiên”, trong khi
hợp đồng nội bộ dùng chỉ số bắt đầu từ `1`. Sai khác quy ước này trước đây làm
Pydantic ném `ValidationError` và dừng toàn bộ lượt chạy.

Thay đổi hiện tại:

- Prompt judge quy định rõ `occurrence` là số nguyên bắt đầu từ `1`, không dùng `0`.
- Biên nhận response và schema cùng chuẩn hóa riêng `0`/`"0"` thành `1`.
- Số âm vẫn không hợp lệ; hệ thống không âm thầm chấp nhận chỉ số nguy hiểm.
- Các biến thể hình thức vô hại được chuẩn hóa: hoa/thường của
  `status`/`severity`/`action`, collection `null`, và field tùy chọn rỗng.
- Khi hợp đồng vẫn sai, log ghi chính xác đường dẫn field lỗi, ví dụ
  `edits.0.occurrence`, thay vì chỉ báo chung chung.

Các test hồi quy nằm trong `tests/test_quality_contracts.py`,
`tests/test_verifier_client.py` và `tests/test_verifier_service.py`.

## 21. Verifier null fallback và token usage trong dataset

- `suggested_fix: null`, field bị thiếu hoặc chuỗi rỗng không còn làm mất toàn bộ
  issue. Code điền một hướng xử lý an toàn theo severity/status rồi giữ nguyên
  `REGENERATE`, `FIXABLE` hoặc `REJECTED`.
- Root validator chỉ chạy khi nested issue đã hợp lệ, nên log không còn báo phụ
  sai lệch rằng `REGENERATE` không có issue.
- Prompt judge yêu cầu `suggested_fix` luôn là chuỗi không rỗng và mô tả rõ cách
  tránh lỗi ở lần generate tiếp theo.
- Mỗi sample trong file `.json` cuối có thêm đúng một field:

```json
{
  "token_usage": {
    "input_tokens": 1240,
    "output_tokens": 380
  }
}
```

Đây là usage của toàn logical slot: Generator, Judge, Repair, re-Judge và các retry
đã thực sự phát sinh token. Mảng dataset, tagged/clean text, entity, privacy mask và
offset không thay đổi. Terminal summary vẫn báo tổng token toàn run.
