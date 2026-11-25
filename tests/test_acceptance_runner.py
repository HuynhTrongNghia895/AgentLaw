"""Unit tests for AX parsing, input adaptation, conservative grading and reports."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from acceptance_runner import (  # noqa: E402
    adapt_ax_input, evaluate_metrics, execute_case, grade_case, is_web_enabled, is_web_required, load_cases,
    run_dialogue, write_reports,
)


def test_loads_all_28_ax_cases_and_preserves_ids_and_status_dimensions():
    cases = load_cases()
    assert len(cases) == 28
    assert len({case["id"] for case in cases}) == 28
    assert all(set(("legal_gt_status", "behavioral_oracle_status", "runtime_status")) <= case.keys()
               for case in cases)
    assert "AX_G5_02" not in {case["id"] for case in cases}


def test_g9_web_off_case_reuses_g9_question_and_does_not_call_web():
    cases = {case["id"]: case for case in load_cases()}
    case = cases["AX_G9_02"]
    assert case["runtime_question"] == cases["AX_G9_01"]["question"]
    assert adapt_ax_input(case)["initial_query"] == cases["AX_G9_01"]["question"]
    assert is_web_required(case) is False


def test_focused_six_case_suite_loads_without_changing_default_28_case_baseline():
    cases = load_cases(ROOT / "evaluation" / "acceptance_multi_intent_scenarios.jsonl")
    assert len(cases) == 6
    assert len({case["id"] for case in cases}) == 6
    assert all(case["id"].startswith("AX_G12_") for case in cases)
    assert len(load_cases()) == 28


def test_web_availability_is_separate_from_required_web_call_oracle():
    case = {"web_enabled": True, "web_required": False}
    assert is_web_enabled(case) is True
    assert is_web_required(case) is False


def test_g11_provenance_case_reuses_the_multi_intent_source_question():
    cases = {case["id"]: case for case in load_cases()}
    assert cases["AX_G11_01"]["runtime_question"] == cases["AX_G2_01"]["question"]


def test_g3_refiner_on_oracle_checks_clarification_before_retrieval():
    case = next(case for case in load_cases() if case["id"] == "AX_G3_03")
    result = grade_case(case, [{"route": "refine", "follow_up": "Điều 4 của văn bản nào?",
                                "evidence": []}], {})
    assert result["status"] == "BLOCKED"  # missing Planner trace prevents layer-complete acceptance
    assert next(c for c in result["checks"] if c["name"] == "clarification")["status"] == "PASS"
    assert next(c for c in result["checks"] if c["name"] == "preclarification_retrieval")["status"] == "PASS"


def test_ax_question_adapts_to_conversation_runner_input_and_dialogue():
    case = next(c for c in load_cases() if c["id"] == "AX_G4_01")
    adapted = adapt_ax_input(case)
    assert adapted["initial_query"] == case["question"]
    assert adapted["dialogue"] == case["dialogue"]


def test_dialogue_keeps_pending_planner_state_until_clarification_then_resets_on_new_query():
    class Stub:
        def __init__(self):
            self.calls = []
            self.state = {"original_question": "A và B", "pending_intent_id": "i2",
                          "intents": [{"intent_id": "i1", "needs_clarification": False},
                                      {"intent_id": "i2", "needs_clarification": True}]}

        def run(self, question, **kwargs):
            self.calls.append((question, kwargs))
            if len(self.calls) == 1:
                return {"follow_up": "Làm rõ B", "planner_state": self.state,
                        "planner_audit": {"original_question": "A và B"}}
            if len(self.calls) == 2:
                return {"answer": "Đã làm rõ", "evidence": [], "planner_state": None,
                        "planner_rewrites": [{"original_sub_query": "B",
                                              "rewritten_query": "B đã đủ ngữ cảnh"}]}
            return {"answer": "Truy vấn mới", "evidence": [], "planner_state": None}

    case = {"id": "AX_G4_01", "question": "A và B", "dialogue": [
        {"type": "initial", "text": "A và B"},
        {"type": "clarification_response", "text": "Trả lời ý B"},
        {"type": "query", "text": "Câu mới"},
    ]}
    stub = Stub()
    transcript = run_dialogue(stub, case)
    assert len(transcript) == 3
    assert stub.calls[1][0] == "A và B"
    assert stub.calls[1][1]["planner_state"]["pending_intent_id"] == "i2"
    assert stub.calls[1][1]["user_answer"] == "Trả lời ý B"
    assert stub.calls[2][0] == "Câu mới"
    assert "planner_state" not in stub.calls[2][1]
    assert transcript[0]["planner_audit"]["original_question"] == "A và B"
    assert transcript[1]["planner_rewrites"][0]["rewritten_query"] == "B đã đủ ngữ cảnh"
    assert transcript[1]["user_answer"] == "Trả lời ý B"
    assert "Trả lời ý B" in transcript[2]["conversation_context"]


def test_multi_intent_planner_response_requires_numbered_answers():
    from runner_conversation import _valid_planner_response

    state = {
        "original_question": "ý A và ý B",
        "pending_intent_ids": ["i1", "i2"],
        "intents": [
            {"intent_id": "i1", "needs_clarification": True},
            {"intent_id": "i2", "needs_clarification": True},
        ],
    }
    assert not _valid_planner_response(state, "câu trả lời không đánh số")
    assert _valid_planner_response(state, "1) câu A\n2) câu B")


def test_grader_blocks_provisional_oracles_and_passes_only_structured_verified_checks():
    case = {"id": "AX_G1_01", "expected_route": "exact", "expected_node_ids": [],
            "expected_evidence": {}, "expected_relations": [],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"route_decision": [{"state": "OK", "data": {"five_route": "exact"}}]}}
    result = grade_case(case, [{"evidence": []}], trace)
    assert result["status"] == "PASS"

    case["behavioral_oracle_status"] = {"status": "provisional"}
    assert grade_case(case, [{"evidence": []}], trace)["status"] == "BLOCKED"


def test_grader_distinguishes_execution_error_from_unresolved_oracle():
    case = {"id": "AX_G1_01"}
    error = grade_case(case, [], {}, runtime_error="Ollama unavailable")
    assert error["status"] == "ERROR"
    blocked = grade_case({**case, "expected_route": "exact",
                          "legal_gt_status": {"status": "verified"},
                          "behavioral_oracle_status": {"status": "verified"}}, [], {})
    assert blocked["status"] == "BLOCKED"


def test_grader_marks_planner_rewrite_execution_error_as_error():
    result = grade_case(
        {"id": "AX_G4_01"},
        [{"execution_status": "ERROR", "answer": "rewrite model unavailable"}],
        {},
    )
    assert result["status"] == "ERROR"


def test_grader_treats_anchor_rejection_as_behavioral_result_not_execution_error():
    case = {"id": "AX_G7_01", "expected_route": "exact",
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"anchor_verify": [{"state": "FAILED",
              "reason": "anchor not verified", "data": {"missing": ["date"]}}],
              "route_decision": [{"state": "OK", "data": {"five_route": "anchor_unresolved"}}]}}
    assert grade_case(case, [], trace)["status"] == "FAIL"


def test_full_route_without_structured_planner_trace_is_blocked_not_failed():
    case = {"id": "AX_G2_01", "expected_route": "full",
            "expected_intents": [{"intent": "a"}, {"intent": "b"}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"route_decision": [{"state": "OK",
              "data": {"five_route": "exact"}}]}}
    result = grade_case(case, [], trace)
    assert result["status"] == "BLOCKED"
    assert not any(check["status"] == "FAIL" for check in result["checks"])


def test_evidence_parent_node_can_materialize_expected_clause_node():
    case = {"id": "AX_G3_02", "expected_route": "exact",
            "expected_evidence": {"expected_node_ids": ["296_2026_ND_CP_D22_K1"]},
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"route_decision": [{"state": "OK",
              "data": {"five_route": "exact"}}]}}
    result = grade_case(case, [{"evidence": [{"node_id": "296_2026_ND_CP_D22"}]}], trace)
    evidence_check = next(c for c in result["checks"] if c["name"] == "evidence_ids")
    assert evidence_check["status"] == "PASS"


def test_required_web_case_fails_if_pipeline_skips_the_call():
    case = {"id": "AX_G9_01", "expected_route": "semantic",
            "expected_steps": ["tra web nguồn chính thức"],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"route_decision": [{"state": "OK",
              "data": {"five_route": "semantic"}}],
              "web_decision": [{"state": "SKIPPED", "reason": "internal enough"}]}}
    result = grade_case(case, [], trace, web_calls=0)
    assert result["status"] == "FAIL"


def test_g4_context_oracle_checks_answer_forwarding_and_history_retention():
    case = {"id": "AX_G4_01", "expected_route": "full",
            "dialogue": [{"type": "initial", "text": "Q"},
                         {"type": "answer", "text": "108/2025"},
                         {"type": "query", "text": "Next"}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "provisional"}}
    transcript = [
        {"event": "initial", "user_answer": "", "conversation_context": ""},
        {"event": "answer", "user_answer": "108/2025", "conversation_context": "Q"},
        {"event": "query", "user_answer": "", "conversation_context": "Q\nAgent: clarification\n108/2025"},
    ]
    result = grade_case(case, transcript, {"spans_by_layer": {}})
    assert next(c for c in result["checks"] if c["name"] == "dialogue_context")["status"] == "PASS"
    assert result["status"] == "BLOCKED"  # provisional oracle is not promoted


def test_multi_intent_provenance_accepts_all_intent_indices():
    case = {"id": "AX_G2_01", "expected_route": "full",
            "expected_intents": [{"intent": "a"}, {"intent": "b"}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"planner": [{"state": "OK", "data": {
        "intents": [{"intent_id": "i1", "sub_query": "a"},
                    {"intent_id": "i2", "sub_query": "b"}]}}]}}
    evidence = [{"node_id": "shared", "metadata": {"intent_indices": [0, 1]}}]
    result = grade_case(case, [{"evidence": evidence}], trace)
    assert "evidence lacks intent_index" not in result["reason"]


def test_intent_oracle_checks_evidence_belongs_to_expected_intent():
    case = {"id": "AX_G12_03", "expected_route": "full",
            "expected_intents": [{"intent": "a"}, {"intent": "b"}],
            "intent_oracles": [
                {"index": 0, "expected_verdict": "COVERED", "expected_node_ids": ["a_node"]},
                {"index": 1, "expected_verdict": "COVERED", "expected_node_ids": ["b_node"]}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"planner": [{"layer": "planner", "state": "OK", "data": {"intents": [
        {"intent_id": "i1"}, {"intent_id": "i2"}]}}]}}
    transcript = [{"per_intent": [
        {"intent_index": 0, "intent_verdict": "COVERED", "node_ids": ["a_node"]},
        {"intent_index": 1, "intent_verdict": "COVERED", "node_ids": ["b_node"]}],
        "evidence": [
            {"node_id": "a_node", "metadata": {"intent_indices": [0]}},
            {"node_id": "b_node", "metadata": {"intent_indices": [1]}}]}]
    result = grade_case(case, transcript, trace)
    assert result["status"] == "PASS"
    evidence_checks = [c for c in result["checks"] if c["name"] == "intent_evidence"]
    assert [c["status"] for c in evidence_checks] == ["PASS", "PASS"]
    transcript[0]["per_intent"][0]["node_ids"] = ["b_node"]
    transcript[0]["evidence"][0]["metadata"]["intent_indices"] = [1]
    wrong = grade_case(case, transcript, trace)
    assert wrong["status"] == "FAIL"


def test_intent_results_are_collected_across_turns_not_only_from_last_turn():
    case = {"id": "AX_G12_03", "expected_route": "full",
            "expected_intents": [{"intent": "a"}, {"intent": "b"}],
            "intent_oracles": [
                {"index": 0, "label": "A", "expected_verdict": "COVERED", "expected_node_ids": ["a_node"]},
                {"index": 1, "label": "B", "expected_verdict": "COVERED", "expected_node_ids": ["b_node"]}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"planner": [{"layer": "planner", "state": "OK", "data": {"intents": [
        {"intent_id": "i1"}, {"intent_id": "i2"}]}}]}}
    transcript = [
        {"per_intent": [
            {"intent_index": 0, "intent_verdict": "COVERED"},
            {"intent_index": 1, "intent_verdict": "COVERED"}],
         "evidence": [{"node_id": "a_node", "metadata": {"intent_indices": [0]}},
                      {"node_id": "b_node", "metadata": {"intent_indices": [1]}}]},
        {"per_intent": [], "evidence": []},
    ]
    result = grade_case(case, transcript, trace)
    assert result["status"] == "PASS"
    assert result["intent_results"]["A"]["status"] == "PASS"
    assert result["intent_results"]["B"]["status"] == "PASS"
    assert set(result["layers"]) >= {"planner", "clarification_route", "retrieval_relations", "answer", "runtime"}


def test_missing_intent_provenance_is_blocked_but_wrong_intent_attribution_fails():
    case = {"id": "AX_G12_03", "expected_route": "full",
            "expected_intents": [{"intent": "a"}, {"intent": "b"}],
            "intent_oracles": [{"index": 1, "label": "B", "expected_node_ids": ["b_node"]}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"planner": [{"layer": "planner", "state": "OK", "data": {"intents": [
        {"intent_id": "i1"}, {"intent_id": "i2"}]}}]}}
    untagged = grade_case(case, [{"evidence": [{"node_id": "b_node"}]}], trace)
    assert next(c for c in untagged["checks"] if c["name"] == "intent_evidence")["status"] == "BLOCKED"
    wrong = grade_case(case, [{"evidence": [{"node_id": "b_node", "metadata": {"intent_indices": [0]}}]}], trace)
    assert next(c for c in wrong["checks"] if c["name"] == "intent_evidence")["status"] == "FAIL"


def test_intent_oracle_without_per_intent_record_is_blocked_not_false_fail():
    case = {"id": "AX_G12_03", "expected_route": "full",
            "expected_intents": [{"intent": "a"}],
            "intent_oracles": [{"index": 0, "label": "A", "expected_verdict": "COVERED"}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"planner": [{"layer": "planner", "state": "OK", "data": {"intents": [
        {"intent_id": "i1"}]}}]}}
    result = grade_case(case, [{"per_intent": []}], trace)
    assert next(c for c in result["checks"] if c["name"] == "intent_verdict")["status"] == "BLOCKED"
    assert result["status"] == "BLOCKED"


def test_intent_verdict_falls_back_to_matching_rerank_span():
    case = {"id": "AX_G12_03", "expected_route": "semantic",
            "expected_intents": [{"intent": "foo"}],
            "intent_oracles": [{"index": 0, "label": "A", "expected_verdict": "COVERED"}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {
        "planner": [{"layer": "planner", "state": "OK", "data": {"intents": [
            {"intent_id": "i1", "sub_query": "foo"}]}}],
        "rerank_per_intent": [{"layer": "rerank_per_intent", "intent": "foo",
                               "data": {"intent_id": "i1", "verdict": "COVERED"}}],
    }}
    result = grade_case(case, [{"per_intent": [{"intent_index": 0, "status": "covered"}]}], trace)
    check = next(c for c in result["checks"] if c["name"] == "intent_verdict")
    assert check["status"] == "PASS"
    assert check["actual"] == "COVERED"


def test_intent_execution_error_propagates_to_case_error():
    case = {"id": "AX_G12_03", "expected_route": "full",
            "expected_intents": [{"intent": "a"}],
            "intent_oracles": [{"index": 0, "label": "A", "expected_verdict": "COVERED"}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"planner": [{"layer": "planner", "state": "OK", "data": {"intents": [
        {"intent_id": "i1"}]}}]}}
    result = grade_case(case, [{"per_intent": [{"intent_index": 0, "execution": "ERROR"}]}], trace)
    assert result["status"] == "ERROR"
    assert result["layers"]["runtime"]["status"] == "ERROR"


def test_claim_citation_link_requires_and_accepts_explicit_human_review_ledger():
    case = {"id": "AX_G1_01", "expected_route": "exact",
            "expected_citations": [{"document_number": "1/2025/QH15", "article": "1"}],
            "human_answer_review_required": True,
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"route_decision": [{"state": "OK", "data": {"five_route": "exact"}}]}}
    transcript = [{"answer": "Theo [Luật 1/2025/QH15 Điều 1], quy định này áp dụng.",
                   "evidence": [{"node_id": "n1"}],
                   "claim_evidence_review": {"status": "human_verified", "claims": [
                       {"review_status": "supported", "evidence_ids": ["n1"],
                        "citation": "Luật 1/2025/QH15 Điều 1"}]}}]
    result = grade_case(case, transcript, trace)
    assert next(c for c in result["checks"] if c["name"] == "citation_scope")["status"] == "PASS"
    assert next(c for c in result["checks"] if c["name"] == "citation_evidence_link")["status"] == "PASS"
    assert result["status"] == "PASS"


def test_metrics_intent_f1_uses_one_to_one_anchor_matching_and_separate_case_status():
    case = {"id": "AX_G12_01", "expected_intents": [{}, {}],
            "metrics_oracles": {"intent_match_terms": [["thoi tiet"], ["108/2025/QH15", "article 2"]]}}
    result = {"id": case["id"], "status": "FAIL", "trace": {"spans_by_layer": {"planner": [
        {"data": {"intents": [{"sub_query": "th\u1eddi ti\u1ebft tomorrow"},
                                {"sub_query": "108/2025/QH15 article 2"},
                                {"sub_query": "extra"}]}}]}}}
    metric = evaluate_metrics([result], [case])
    assert metric["planner"]["intent_precision"]["value"] == 2 / 3
    assert metric["planner"]["intent_recall"]["value"] == 1
    assert metric["planner"]["intent_f1_micro"]["value"] == 0.8
    assert result["status"] == "FAIL"  # metric evaluation never changes Acceptance


def test_metrics_retrieval_is_not_evaluable_without_ranked_trace_and_has_n_zero():
    case = {"id": "AX_G12_01", "intent_oracles": [
        {"index": 0, "expected_node_ids": ["n1"]}],
        "metrics_oracles": {"intent_match_terms": [["x"]]}}
    result = {"id": case["id"], "status": "PASS", "trace": {"spans_by_layer": {
        "planner": [{"data": {"intents": [{"intent_id": "i0"}]}}]}}}
    metrics = evaluate_metrics([result], [case])
    assert metrics["retrieval"]["mrr"]["status"] == "NOT_EVALUABLE"
    assert metrics["retrieval"]["mrr"]["n"] == 0
    assert metrics["retrieval"]["mrr"]["skipped"] == 1


def test_metrics_retrieval_hit_at_k_and_mrr_use_ranked_pre_top_k_nodes():
    case = {"id": "AX_G12_01", "intent_oracles": [
        {"index": 0, "expected_node_ids": ["target"]}]}
    result = {"id": case["id"], "status": "PASS", "trace": {"spans_by_layer": {
        "planner": [{"data": {"intents": [{"intent_id": "i0"}]}}],
        "retrieval_rank": [{"intent": "i0", "data": {
            "ranking_stage": "post_rerank_pre_topk",
            "node_ids_ranked": ["wrong", "target", "other"]}}]}}}
    metrics = evaluate_metrics([result], [case])["retrieval"]
    assert metrics["hit_at_1"]["value"] == 0
    assert metrics["hit_at_3"]["value"] == 1
    assert metrics["hit_at_5"]["value"] == 1
    assert metrics["mrr"]["value"] == 0.5


def test_metrics_relation_precision_recall_f1_count_extra_edges_and_keep_n():
    expected = {"from": "A", "relation_type": "REPEALS", "to": "B", "to_article": "3"}
    extra = {"from": "A", "relation_type": "AMENDS", "to": "B", "to_article": "4"}
    case = {"id": "AX_G12_04", "metrics_oracles": {"relation_gt_complete": True},
            "expected_intents": [{}, {}], "intent_oracles": [
                {"index": 0, "expected_relations": [expected, {**expected, "to_article": "5"}]}]}
    result = {"id": case["id"], "status": "PASS", "trace": {"spans_by_layer": {
        "planner": [{"data": {"intents": [{"intent_id": "i0"}, {"intent_id": "i1"}]}}],
        "kg_relations": [{"intent": "i0", "data": {"actual_relations": [expected, extra]}}]}}}
    metrics = evaluate_metrics([result], [case])["knowledge_graph"]
    assert metrics["relation_precision"]["value"] == 0.5
    assert metrics["relation_recall"]["value"] == 0.5
    assert metrics["relation_f1"]["value"] == 0.5
    assert metrics["relation_f1"]["n"] == 2


def test_g7_report_has_separate_acceptance_and_metrics_sections():
    result = {"id": "AX_G12_01", "group": "G8", "status": "BLOCKED",
              "elapsed_ms": 1, "llm_calls": 0, "web_calls": 0, "trace_ref": None}
    case = {"id": "AX_G12_01"}
    with tempfile.TemporaryDirectory(dir=ROOT / "evaluation") as temp_dir:
        _, json_path, md_path = write_reports([result], Path(temp_dir) / "g7", [case])
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["summary"]["counts"] == {"BLOCKED": 1}
        assert payload["metrics_evaluation"]["planner"]["intent_f1_micro"]["status"] == "NOT_EVALUABLE"
        markdown = md_path.read_text(encoding="utf-8")
        assert "## Acceptance Evaluation" in markdown
        assert "## Metrics Evaluation" in markdown


def test_web_metric_is_separate_and_does_not_count_runtime_errors_as_no_call():
    cases = [
        {"id": "AX_G12_01", "expected_web_calls": {"min": 0, "max": 0}},
        {"id": "AX_G12_02", "expected_web_calls": {"min": 1}},
    ]
    results = [
        {"id": "AX_G12_01", "status": "FAIL", "web_calls": 0},
        {"id": "AX_G12_02", "status": "ERROR", "web_calls": 0},
    ]
    metric = evaluate_metrics(results, cases)["routing"]["web_decision_accuracy"]
    assert metric["value"] == 1.0
    assert metric["n"] == 1
    assert metric["skipped"] == 1


def test_runtime_latency_reports_mean_median_p95_and_keeps_errors():
    cases = [{"id": f"AX_G12_0{i}"} for i in range(1, 4)]
    results = [
        {"id": cases[0]["id"], "status": "PASS", "elapsed_ms": 120,
         "timings_ms": {"orchestrator_ms": 100}},
        {"id": cases[1]["id"], "status": "FAIL", "elapsed_ms": 220,
         "timings_ms": {"orchestrator_ms": 200}},
        {"id": cases[2]["id"], "status": "ERROR", "reason": "timeout", "elapsed_ms": 320,
         "timings_ms": {"orchestrator_ms": 300}},
    ]
    runtime = evaluate_metrics(results, cases)["runtime"]
    assert runtime["average_latency_ms"]["value"] == 200
    assert runtime["median_latency_ms"]["value"] == 200
    assert runtime["p95_latency_ms"]["value"] == 300
    assert runtime["runtime_errors"] == 1
    assert runtime["runtime_timeouts"] == 1


def test_execution_result_records_stage_timings_and_layer_statuses():
    class Recorder:
        def summary(self):
            return {"trace_id": "t1", "n_llm_calls": 0,
                    "spans_by_layer": {"route_decision": [{"state": "OK", "data": {"five_route": "exact"}}]}}
    class Stub:
        web = None
        def run(self, query, **kwargs):
            return {"answer": "answer", "evidence": [], "execution_status": "SUCCESS"}
    case = {"id": "AX_G1_01", "group": "G1", "question": "Q", "expected_route": "exact",
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    result = execute_case(case, lambda use_web: (Stub(), Recorder()))
    assert result["status"] == "PASS"
    assert set(result["timings_ms"]) == {"initialization_ms", "orchestrator_ms", "trace_summary_ms", "grading_ms"}
    assert result["layers"]["runtime"]["status"] == "PASS"


def test_relation_oracle_can_check_article_and_clause_targets():
    case = {"id": "AX_G12_04", "expected_route": "exact",
            "expected_relations": [{"from": "296/2026/NĐ-CP", "relation_type": "REPEALS",
                                    "to": "168/2025/NĐ-CP", "to_article": "52",
                                    "to_clause": "2"}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {
        "route_decision": [{"state": "OK", "data": {"five_route": "exact"}}],
        "kg_relations": [{"layer": "kg_relations", "state": "OK", "data": {"actual_relations": [
            {"from": "296/2026/NĐ-CP", "relation_type": "REPEALS",
             "to": "168/2025/NĐ-CP", "to_article": "52", "to_clauses": "khoản 2"}]}}]}}
    result = grade_case(case, [], trace)
    assert next(c for c in result["checks"] if c["name"] == "relations")["status"] == "PASS"
    trace["spans_by_layer"]["kg_relations"][0]["data"]["actual_relations"][0]["to_clauses"] = "khoản 3"
    result = grade_case(case, [], trace)
    assert result["status"] == "FAIL"


def test_web_probe_is_checked_for_expected_intent_and_human_review_blocks_final_pass():
    case = {"id": "AX_G12_02", "expected_route": "full",
            "expected_intents": [{"intent": "internal"}, {"intent": "current status"}],
            "expected_web_calls": {"min": 1}, "expected_web_intent_indices": [1],
            "web_required": True, "human_answer_review_required": True,
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {
        "planner": [{"layer": "planner", "state": "OK", "data": {"intents": [
            {"intent_id": "i1"}, {"intent_id": "i2"}]}}],
        "web": [{"layer": "web", "state": "OK", "data": {"intent_ids": ["i2"]}}]}}
    result = grade_case(case, [], trace, web_calls=1)
    assert next(c for c in result["checks"] if c["name"] == "web_call_count")["status"] == "PASS"
    assert next(c for c in result["checks"] if c["name"] == "web_intent_scope")["status"] == "PASS"
    assert result["status"] == "BLOCKED"
    assert "human answer review" in result["reason"]


def test_intent_clarification_and_temporal_dates_are_checked_separately():
    base = {"id": "AX_G12_07", "expected_route": "full",
            "expected_intents": [{"intent": "answerable"}, {"intent": "clarify"}],
            "intent_oracles": [
                {"index": 0, "expected_verdict": "COVERED", "expected_node_ids": ["n0"]},
                {"index": 1, "clarification_required": True,
                 "clarification_contains_all": ["sắc thuế", "loại khoản đóng góp"],
                 "expect_no_evidence": True}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {"planner": [{"layer": "planner", "state": "OK", "data": {
        "intents": [{"intent_id": "i0"}, {"intent_id": "i1", "needs_clarification": True}]}}]}}
    transcript = [{"follow_up": "Bạn cho biết sắc thuế và loại khoản đóng góp nhé.",
                   "per_intent": [{"intent_index": 0, "intent_verdict": "COVERED", "node_ids": ["n0"]}],
                   "evidence": [{"node_id": "n0", "metadata": {"intent_indices": [0]}}]}]
    result = grade_case(base, transcript, trace)
    assert result["status"] == "PASS"
    clarify_check = next(c for c in result["checks"] if c["name"] == "intent_clarification")
    assert clarify_check["status"] == "PASS"
    no_evidence_check = next(c for c in result["checks"] if c["name"] == "intent_evidence" and c["index"] == 1)
    assert no_evidence_check["status"] == "PASS"

    timed = {"id": "AX_G12_05", "expected_route": "full",
             "expected_intents": [{"intent": "before"}, {"intent": "from"}],
             "intent_oracles": [
                 {"index": 0, "expected_as_of_date": "2026-07-22"},
                 {"index": 1, "expected_as_of_date": "2026-07-23"}],
             "legal_gt_status": {"status": "verified"},
             "behavioral_oracle_status": {"status": "verified"}}
    timed_trace = {"spans_by_layer": {
        "planner": [{"layer": "planner", "state": "OK", "data": {"intents": [
            {"intent_id": "before"}, {"intent_id": "from"}]}}],
        "legal_version": [
            {"layer": "legal_version", "intent": "before", "state": "OK",
             "data": {"actual_version": {"as_of_date": "2026-07-22"}}},
            {"layer": "legal_version", "intent": "from", "state": "OK",
             "data": {"actual_version": {"as_of_date": "2026-07-23"}}}]}}
    result = grade_case(timed, [{"evidence": [
        {"node_id": "n0", "metadata": {"intent_indices": [0, 1]}}]}], timed_trace)
    date_checks = [c for c in result["checks"] if c["name"] == "intent_version_time"]
    assert [c["status"] for c in date_checks] == ["PASS", "PASS"]


def test_relation_grader_compares_structured_graph_observations():
    case = {"id": "AX_G6_01", "expected_route": "exact",
            "expected_relations": [{"from": "296/2026/NĐ-CP", "relation_type": "REPEALS",
                                    "to": "168/2025/NĐ-CP"}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    trace = {"spans_by_layer": {
        "route_decision": [{"state": "OK", "data": {"five_route": "exact"}}],
        "kg_relations": [{"layer": "kg_relations", "state": "OK", "data": {"actual_relations": [
            {"from": "296/2026/NĐ-CP", "relation_type": "REPEALS",
             "to": "168/2025/NĐ-CP"}]}}]}}
    result = grade_case(case, [], trace)
    assert next(check for check in result["checks"] if check["name"] == "relations")["status"] == "PASS"


def test_g2_citation_grader_requires_the_specific_clause_not_only_article():
    case = {"id": "AX_G2_01", "expected_route": "full",
            "expected_intents": [{"intent": "a"}, {"intent": "b"}],
            "expected_citations": [
                {"document_number": "108/2025/QH15", "article": "2", "clause": "1"},
                {"document_number": "168/2025/NĐ-CP", "article": "2", "clause": "1"}],
            "legal_gt_status": {"status": "verified"},
            "behavioral_oracle_status": {"status": "verified"}}
    broad = grade_case(case, [{"answer": "[108/2025/QH15 Điều 2] [168/2025/NĐ-CP Điều 2]"}], {})
    precise = grade_case(case, [{"answer": "[108/2025/QH15 Điều 2 Khoản 1] "
                                 "[168/2025/NĐ-CP Điều 2 Khoản 1]"}], {})
    assert next(c for c in broad["checks"] if c["name"] == "citation_scope")["status"] == "FAIL"
    assert next(c for c in precise["checks"] if c["name"] == "citation_scope")["status"] == "PASS"


def test_report_writer_emits_jsonl_json_markdown_and_status_counts():
    results = [{"id": "AX_G1_01", "group": "G1", "status": "BLOCKED",
                "elapsed_ms": 2, "llm_calls": 0, "web_calls": 0,
                "reason": "oracle incomplete", "trace_ref": "trace-1"}]
    with tempfile.TemporaryDirectory(dir=ROOT / "evaluation") as temp_dir:
        jsonl_path, json_path, md_path = write_reports(results, Path(temp_dir) / "report")
        assert json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])["id"] == "AX_G1_01"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["summary"]["counts"] == {"BLOCKED": 1}
        assert "AX_G1_01" in md_path.read_text(encoding="utf-8")
