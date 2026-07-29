# Taxonomy few-shot + Decoy Planner: báo cáo xác minh

Ngày chạy: 2026-07-29.

## Phạm vi

Báo cáo này ghi lại baseline online đã có và ba probe online gồm năm slot
`mixed_contrastive`. Raw JSON và log vẫn nằm cục bộ trong `gen_data/` và không
được commit. Probe dùng Generator `gemini-2.5-flash`, Judge
`gemini-2.5-pro`, tối đa hai Generator attempts cho mỗi slot và không tạo task
thay thế.

Đây chưa phải A/B 50 mẫu. Sau probe cuối, tỷ lệ pass lần đầu vẫn dưới tiêu chí
85%; vì vậy run 50 mẫu không được tiếp tục chỉ để tiêu thêm chi phí và có nguy
cơ tạo một báo cáo đạt giả tạo.

## Kết quả đo được

| Run | Slot | Accepted | Pass lần đầu | Token tổng | Cost (USD) | Cost/accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline `quality-ab-v3` | 10 logical slots | 7 | Không có số liệu tương đương | 345,100 | 0.74115500 | 0.10587929 |
| Probe `r2` | 5 | 1 | 1/5 (20%) | 98,527 | 0.12571050 | 0.12571050 |
| Probe `r3` | 5 | 4 | 2/5 (40%) | 105,433 | 0.20570540 | 0.05142635 |
| Probe `r4` | 5 | 4 | 3/5 (60%) | 103,229 | 0.22548985 | 0.05637246 |

Baseline sinh 46 candidates, accept 7, pass rate trên candidate 15.22% và lỗi
chủ yếu là `length_out_of_range` (33 lần). Cấu hình Judge sau thay đổi không còn
reject vì vượt cận trên. Do baseline và probe khác kích thước/budget, chênh lệch
cost/accepted chỉ là tín hiệu định hướng, không phải kết luận thống kê.

Ở `r4`:

- ba slot pass Generator + deterministic gate + Judge ngay lần đầu;
- một slot bị `decoy_integration_weak`, regenerate rồi PASS;
- một slot bị Judge loại cả hai lần; lần cuối là `UNNATURAL_TEXT` và
  `MISSING_ANNOTATION`;
- không slot nào còn lỗi `missing_positive_seed`;
- run kết thúc `FAILED`, chỉ giữ partial artifacts và không công bố file dataset
  hoàn chỉnh khi thiếu một slot.

Review thủ công còn tìm được một mẫu dùng lời giải thích trực tiếp “không phải
địa chỉ giao hàng”. Mẫu này từng được Judge PASS. Sau probe, deterministic gate
đã được mở rộng để route mọi semantic disclaimer dạng này sang `REGENERATE` và
có regression test tương ứng.

## Bảo đảm bằng code và test

- Taxonomy có 44 label, mỗi label có ba blueprint và ít nhất hai blueprint phi
  kỹ thuật.
- Toàn bộ 132 `source_example_ids` được parser đối chiếu với
  `EXAMPLES.HARD_NEGATIVE` của chính label.
- Registry hiện có 7/132 blueprint kỹ thuật (5.30%), thấp hơn hard cap 15%.
- Chỉ decoy target đã chọn nhận đúng ba hard-negative examples; robin label
  khác chỉ nhận definition/rule.
- Planner chọn family/context/anchor/relation có thể tái lập theo random seed.
- Deterministic gate kiểm tra anchor cùng hoặc liền kề, stock tail, context
  mismatch, few-shot imitation, semantic disclaimer và ADDRESS/LOCATION
  boundary.
- Value Bank không chọn ADDRESS đã gộp LOCATION; source file không bị sửa.
- 260 unittest pass.

## Đối chiếu tiêu chí chấp nhận

| Tiêu chí | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Technical/schema/category ≤15% | Đạt ở registry; chưa đo trên 50 output | 5.30% blueprint là technical; planner có hard cap |
| 0 stock trailing sentence | Được gate bằng test; chưa chứng minh trên 50 output | `decoy_stock_scaffold` → `REGENERATE` |
| Anchor cùng hoặc liền kề | Được gate bằng test | `DecoyIntegrationValidator` |
| Final Judge PASS 50/50 | Chưa đạt/chưa chạy | Probe cuối 4/5 |
| First-attempt PASS ≥85% | Chưa đạt | Probe cuối 3/5 = 60% |
| Duplicate skeleton ≤10% | Chưa đủ mẫu để kết luận | Không báo cáo số liệu suy đoán |
| Naturalness thủ công ≥4/5 cho 90% | Chưa đạt độ tin cậy | Probe nhỏ vẫn có semantic disclaimer và một slot bị Judge loại |
| Cost trung bình tăng ≤15% | Chưa thể kết luận | Cấu hình baseline/probe không tương đương |

## Kết luận

Kiến trúc taxonomy-backed planner, contract dữ liệu, prompt abstraction và các
quality gate đã được triển khai và kiểm thử. Probe cho thấy xu hướng pass lần đầu
tăng từ 20% lên 60% và lỗi missing seed đã được loại bỏ, nhưng tiêu chí online
50/50 và 85% first-pass chưa đạt. Một A/B 50 mẫu mới chỉ nên chạy sau khi probe
5–10 slot kế tiếp đạt ít nhất 85% first-pass với chính cấu hình production.
