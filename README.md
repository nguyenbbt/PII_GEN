# PII Data Factory

Repository này triển khai pipeline của PII Data Factory: từ REST/CLI entry point,
JSON Taxonomy/Seed/Data Generator, qua technical deterministic validators và
few-shot imitation guard, đến Python Output Formatter và file JSON trong
`gen_data`. NoveltyGuard cùng LLM Verifier là quality checks tùy chọn.

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
  → optional NoveltyGuard + LLM Judge/Repair
  → OutputFormatter
  → gen_data/{run_name}-{run_id}.json
```

Hard-negative mặc định chạy theo mode `decoy_only`: hệ thống chọn ngẫu nhiên đúng một mã trong `focus_labels`, tạo decoy theo registry phủ đủ 44 mã của taxonomy, không tạo positive entity và yêu cầu output `entities: []`. Chỉ mã nhãn chính xác từ taxonomy snapshot được chấp nhận; tài liệu PDF hard-negative chỉ cung cấp nguyên tắc thiết kế ví dụ, không cung cấp label cho code.

Positive entity value được lấy từ `PII_Value_Bank/{vi,en,de}_pii_value_pools.json`
theo `RunConfig.language` và taxonomy class. LLM chỉ viết nội dung cùng placeholder
như `[PERSON_1]`; Python chèn value trước validator và formatter. Cấu hình thư mục
qua `value_bank.path` trong run config. Xem chi tiết tại
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

Đo độ đa dạng 100 sample mà không gọi Azure:

```powershell
& '.\.venv\bin\python.exe' scripts\audit_diversity.py --samples 100
```
