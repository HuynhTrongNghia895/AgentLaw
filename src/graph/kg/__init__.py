"""KG 4 tầng cho pháp luật VN (Document -> Dieu -> Kham -> Diem).

Seed deterministic (0 LLM) từ data/chunked + 2 CSV verified.
Graphiti dùng làm store + retrieval ở GĐ4; phần LLM-extract chỉ trigger
khi add văn bản luật MỚI (con đường chưa dùng hiện tại).
"""
