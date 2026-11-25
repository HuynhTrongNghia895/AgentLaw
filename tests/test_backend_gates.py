from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "src" / "graph"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agents.QueryDecomposer import QueryDecomposer


def test_provision_without_document_requires_clarification():
    plan = QueryDecomposer(llm=None).plan("Điều 5 quy định thủ tục gì?")
    intent = plan["intents"][0]
    assert intent["needs_clarification"] is True
    assert "document" in intent["missing_scope"]
    assert "văn bản" in intent["clarification"]


def test_vague_legal_request_requires_scope():
    plan = QueryDecomposer(llm=None).plan("Cho tôi kết luận pháp lý cụ thể nhưng không nêu văn bản, điều khoản hay dữ kiện vụ việc.")
    intent = plan["intents"][0]
    assert intent["needs_clarification"] is True
    assert intent["ambiguity_type"] == "scope"
    assert intent["missing_scope"] == ["scope"]


def test_donation_tax_request_requires_tax_scope():
    plan = QueryDecomposer(llm=None).plan("Khoản đóng góp từ thiện có được giảm thuế không?")
    intent = plan["intents"][0]
    assert intent["needs_clarification"] is True
    assert {"tax_type", "taxpayer", "donation_type"}.issubset(intent["missing_scope"])
