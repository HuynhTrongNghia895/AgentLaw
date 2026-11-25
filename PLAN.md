# Plan: Backend correctness fixes — clarification, evidence gating, synthesis

## Context
Các luồng thật cho thấy backend vẫn trả evidence không liên quan cho câu hỏi mơ hồ, không hỏi lại khi thiếu số hiệu văn bản/scope thuế, và để câu đa intent thất bại toàn bộ khi Planner lỗi. Mục tiêu là chặn retrieval trước khi đủ điều kiện, giữ đúng primary/context provenance, và trả output có cấu trúc tự nhiên hơn. Chỉ chạy unit/mock/local; không khởi động web, không gọi LLM/WebSearch/API thật.

## Recommended implementation
1. **Eligibility và clarification gate**
   - Rà `LegalRetrieval.answer` và planner records để các intent `needs_clarification` không đi vào retrieval.
   - Bổ sung rule xác định query chỉ có Điều/Khoản nhưng thiếu số hiệu văn bản; trả clarification yêu cầu tên/số hiệu văn bản.
   - Bổ sung gate cho query quá mơ hồ/không có chủ đề pháp lý đủ cụ thể; trả `CLARIFICATION_REQUIRED` hoặc `INSUFFICIENT_EVIDENCE`, với `primary_evidence=[]`, không synthesis.
   - Bảo đảm câu hỏi thuế/từ thiện thiếu tax type, taxpayer hoặc donation type luôn dừng để hỏi scope.

2. **Primary/context và claim support**
   - Giữ `primary_evidence` chỉ gồm căn cứ trực tiếp; `legal_context` chỉ dùng cho quan hệ/bối cảnh.
   - Với claim có document/article/clause/point anchor, chỉ cho kết luận nếu primary evidence match đúng anchor; không nâng article/context evidence thành support cho clause claim.
   - Gắn metadata verification/abstain rõ ràng để `api_server.py` không phục hồi generic evidence thành primary.

3. **Planner fallback và multi-intent**
   - Khi Planner LLM lỗi/parse lỗi, dùng fallback rule-based hiện có hoặc bổ sung fallback tối thiểu để tách các ý rõ.
   - Cho phép intent đủ dữ kiện tiếp tục xử lý; chỉ intent thiếu dữ kiện mới ở trạng thái clarification.
   - Không biến toàn bộ câu hỏi thành lỗi chỉ vì một intent pending.

4. **Output synthesis tự nhiên và an toàn**
   - Tách kết luận khỏi evidence dump/marker nội bộ; loại `[[INTENT:*]]` và raw diagnostics khỏi answer production.
   - Chuẩn hóa metadata `verification_status` cho confirmed/provisional/unverified, đặc biệt time-travel/Web unavailable.
   - Không trả raw provider exception trong API metadata.

5. **Tests và verification**
   - Bổ sung/điều chỉnh mock tests cho: query mơ hồ, Điều thiếu văn bản, từ thiện thiếu scope, clause mismatch, context không thành primary, planner fallback, và time-travel unverified.
   - Chạy các test backend liên quan bằng stub/mock, `py_compile`, `git diff --check`, và `npm --prefix frontend run build` chỉ để kiểm tra compile frontend; không chạy server/browser.

## Critical files
- `src/agents/LegalRetrieval.py`
- `src/agents/QueryDecomposer.py`
- `src/agents/Orchestrator.py`
- `api_server.py`
- `tests/test_phase_b.py`
- `tests/test_phase_c.py`
- `tests/test_orchestrator_web.py`

## Stop condition
Báo cáo chính xác file đã sửa, test mock/local đã chạy, build frontend đã chạy, warning/failure còn lại (nếu có), và xác nhận không khởi động web/không gọi dịch vụ thật.

## Progress
- [ ] Backend eligibility/clarification gates
- [ ] Primary/context and claim validation
- [ ] Planner fallback and synthesis output
- [ ] Mock regression tests
- [ ] Local verification