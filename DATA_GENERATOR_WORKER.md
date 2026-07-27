# Data Generator Worker

Worker này nhận một payload `data.generation.requested` đã có validated `SeedPack`
và rule-based `ContextFrame`, rồi yêu cầu Azure OpenAI tạo nội dung với placeholder
skeleton. Prompt chỉ chứa `[LABEL_N]`, không chứa positive value thật. Sau response,
worker dùng Python chèn value từ `SeedPack`, rồi mới kiểm tra positive seed phải giữ
nguyên, decoy phải không tag và có cue ngữ nghĩa gần nó. `Output Formatter` là thành
phần khác và tính offsets sau bước chèn value.

## Local run

Môi trường ảo đã ở `.venv`. Với Python MSYS hiện có, dùng:

```powershell
& '.\.venv\bin\python.exe' -m data_generator_worker.main --input examples\data-generation-request.json
```

Trên Python Windows thông thường, đường dẫn tương đương là `.venv\Scripts\python.exe`.

`.env` là local-only và đã được ignore. Không commit file này. Các biến kết nối
gồm `OPENAI_API_KEY`, `BASE_URL`, `API_VERSION`, `GENERATOR_MODEL`,
`DEPLOYMENT_NAME`, `TEMPERATURE`, và `MAX_TOKENS`. `MODEL` vẫn là fallback khi
`GENERATOR_MODEL` không được cấu hình.

Nếu Python không nhận system certificate store, đặt `SSL_CERT_FILE` trỏ tới CA bundle tin cậy. Adapter vẫn thực hiện TLS certificate verification.

## Token and cost output

Mỗi `data.generated.payload.token_usage` luôn chứa:

```json
{
  "input_tokens": 812,
  "output_tokens": 131,
  "total_tokens": 943,
  "money_cost": "0.00334000"
}
```

`input_tokens`, `output_tokens`, `total_tokens` lấy từ trường `usage` của Azure response. Chi phí được tính chính xác theo công thức:

```text
input_tokens / 1,000,000 * INPUT_TOKEN_PRICE_PER_MILLION_USD
+ output_tokens / 1,000,000 * OUTPUT_TOKEN_PRICE_PER_MILLION_USD
```

Giá mặc định trong `.env` là 2.50 USD input và 10.00 USD output mỗi một triệu token; đây là biến cấu hình, nên phải điều chỉnh theo Azure region/contract trước khi dùng cho billing chính thức.

## Taxonomy context output

Mỗi kết quả thành công có `data.generated.payload.taxonomy_context_used`. Trường
này chứa đúng structured guidance đã được đưa vào prompt của LLM trong attempt đó:
focus label có definition/rule/ba example đúng sample type, còn robin labels chỉ có
definition/rule.

```json
{
  "taxonomy_context_used": {
    "taxonomy_version_id": "taxonomy-v1",
    "sample_type": "positive",
    "focus_label": {
      "label": "TIME",
      "definition": "Mốc thời gian cụ thể.",
      "rule": "Gắn đúng giá trị chỉ thời gian.",
      "examples": ["ba example có id, expected_tagged_text và rationale"]
    },
    "robin_labels": []
  }
}
```

Nguồn duy nhất của context này là `pii_taxonomy_rules.json`; không có embedding,
vector retrieval hoặc knowledge chunk.

## Diversity profile output

Prompt version `data-generator.v10.0.0` chuyển `difficulty`, optional constraints và
`diversity_profile` thành các realization rule cụ thể, đồng thời đánh số placeholder
theo class (`[PERSON_1]`, `[PERSON_2]`, `[EMAIL_1]`). Few-shot chỉ được dùng để học
ngữ nghĩa label và ranh giới annotation; prompt cấm sao chép scenario, actor,
action, wording hoặc sentence structure, còn Python `FewShotImitationGuard` chặn
candidate quá giống và yêu cầu regenerate. Khi task có `focus_label`, prompt buộc
label này làm entity trung tâm và chỉ dùng các `robin_labels` đã chọn để hỗ trợ cùng
một sự kiện, không nối bằng mệnh đề rời rạc. Với hard-negative `decoy_only`, decoy
mặc định xuất hiện một lần và chỉ lặp lại tối đa một lần khi cùng sự kiện thật sự
cần xác nhận, đính chính hoặc đối chiếu; mỗi câu chứa decoy phải sao chép nguyên văn
ít nhất một cue ngữ nghĩa. Với `mixed_contrastive`, model phải phân giải các bề mặt
dễ nhầm theo ngữ nghĩa của ngữ cảnh (ví dụ visa du lịch khác mạng thẻ VISA), đồng
thời dùng decoy như trường, khóa, chỉ số hoặc nguyên nhân trực tiếp xử lý positive
record. Prompt áp dụng kiểm tra phản thực: nếu bỏ mệnh đề decoy mà sự kiện positive
không đổi thì phải viết lại, và mệnh đề phải kết thúc bằng hậu quả vận hành thay vì
disclaimer giải thích annotation. Event output có thêm `diversity_profile`, cho
biết context frame, vai người nói, mục đích, cấu trúc tài liệu, văn phong, độ dài và
nguồn format `value_bank`.

## Production integration

`DataGeneratorWorker.process()` nhận `DataGenerationRequest`, dùng idempotency key `{task_id}:attempt:{attempt_no}:data-generation`, và trả event. Adapter production cần thay `InMemoryAttemptStore` bằng PostgreSQL transaction + transactional outbox, sau đó publish event này vào queue `data.generated` chỉ sau khi transaction commit.
