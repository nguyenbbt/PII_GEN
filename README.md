# PII Data Factory

Repository này triển khai pipeline của PII Data Factory: từ REST/CLI entry point,
JSON Taxonomy/Seed/Data Generator, qua technical deterministic validators và
few-shot imitation guard, đến Gemini Judge, Python Output Formatter và file JSON
trong `gen_data`. Config mẫu bật Judge cho mọi candidate online.

```text
POST /api/v1/runs
  → Taxonomy Service (snapshot/version)
  → Run Orchestrator
  → Coverage Controller
  → DiversityPlanner (seeded quota + balanced context)
  → SampleTypeRouter + multilingual Value Bank
  → SeedPackValidator + ContextFrameSelector
  → JSON Taxonomy Context Selector
  → Data Generator (Azure OpenAI placeholder skeleton)
  → Python placeholder binding (Value Bank values)
  → technical DeterministicOutputValidator + FewShotImitationGuard
  → optional NoveltyGuard + configured LLM Judge/Repair
  → OutputFormatter
  → gen_data/{run_name}-{run_id}.json
```

Hard-negative mặc định chạy theo mode `decoy_only`: hệ thống chọn ngẫu nhiên đúng một mã trong `focus_labels`, tạo decoy theo registry phủ đủ 44 mã của taxonomy, không tạo positive entity và yêu cầu output `entities: []`. Chỉ mã nhãn chính xác từ taxonomy snapshot được chấp nhận; tài liệu PDF hard-negative chỉ cung cấp nguyên tắc thiết kế ví dụ, không cung cấp label cho code.

Positive entity value được lấy từ file ánh xạ theo `RunConfig.language` và taxonomy
class. Mặc định `en` dùng `PII_Value_Bank/en_pii_value_pools.json`. LLM chỉ viết nội
dung cùng placeholder như `[PERSON_1]`; Python chèn value trước validator và
formatter. Cấu hình thư mục qua `value_bank.path` và tên file từng ngôn ngữ qua
`value_bank.language_files` trong run config. Xem chi tiết tại
[README_V2.md](README_V2.md#34-value-bank-và-seed-generation).

- Tài liệu kiến trúc tổng thể: [README_V2.md](README_V2.md)
- Hướng dẫn chi tiết pipeline đã triển khai: [README_V2.md](README_V2.md)

Chạy API offline, không gọi LLM và không phát sinh hóa đơn (cost trong output là
giá trị mô phỏng để kiểm thử accounting):

```powershell
& '.\.venv\bin\pii-factory.exe' --offline
```

Chạy với taxonomy JSON mặc định và distribution config:

```powershell
& '.\.venv\bin\pii-factory.exe' --config configs\run_config.example.json --offline
```

Output offline nằm trong `gen_data/offline-smoke` và chỉ dùng để smoke test, không
dùng làm dataset. Chạy online bằng cách bỏ `--offline`.

Config test 10 sample có thể chạy thật 10 tiến trình bằng runner song song:

```powershell
python -m pii_factory.parallel `
  --config configs\run_config.online-test.local.json `
  --output-dir gen_data\online-vi-10
```

Config này dùng `workers=10`, `shard_size=1`: mỗi sample là một shard/process độc
lập. Runner lưu log từng shard, file dataset hợp nhất và file `*-summary.json` chứa
tổng input/output token của toàn phiên, tách riêng Generator và Verifier. Nếu có
shard hết retry, runner chờ các shard đang chạy kết thúc, ghi `*-failed-summary.json`
rồi dừng mà không xuất dataset thiếu mẫu.

Config online có thể bật
`validation.accept_last_candidate_on_exhaustion=true`. Khi toàn bộ Generator attempt
và task replacement đã hết, candidate cuối vẫn được xuất nếu Formatter xác nhận
schema, tag và offset an toàn; summary tăng `fallback_accepts`. Lỗi hạ tầng, Value
Bank hoặc credential/real-PII critical không bị ép thành sample.

Đo độ đa dạng 100 sample mà không gọi Azure:

```powershell
& '.\.venv\bin\python.exe' scripts\audit_diversity.py --samples 100
```
