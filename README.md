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
  → SampleTypeRouter + Faker/controlled seed factories
  → SeedPackValidator + ContextFrameSelector
  → JSON Taxonomy Context Selector
  → Data Generator (Azure OpenAI)
  → technical DeterministicOutputValidator + FewShotImitationGuard
  → optional NoveltyGuard + configured LLM Judge/Repair
  → OutputFormatter
  → gen_data/{run_name}-{run_id}.json
```

Hard-negative mặc định chạy theo mode `decoy_only`: hệ thống chọn ngẫu nhiên đúng một mã trong `focus_labels`, tạo decoy theo registry phủ đủ 44 mã của taxonomy, không tạo positive entity và yêu cầu output `entities: []`. Chỉ mã nhãn chính xác từ taxonomy snapshot được chấp nhận; tài liệu PDF hard-negative chỉ cung cấp nguyên tắc thiết kế ví dụ, không cung cấp label cho code.

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

Đo độ đa dạng 100 sample mà không gọi Azure:

```powershell
& '.\.venv\bin\python.exe' scripts\audit_diversity.py --samples 100
```
