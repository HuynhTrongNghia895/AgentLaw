# Plan: Phase B — clarification, anchor, duplicate-submit, abstain evidence

## Context
Phase A is approved. Phase B must fix clarification generation, anchor preservation, temporal comparison scope, duplicate request submission, and the missing abstain evidence contract. Tests must be mock/unit only: zero LLM tokens, no API calls, and no web server startup. Existing stderr logging and unrelated working-tree changes remain untouched.

## Recommended implementation

1. **Abstain evidence contract**
   - Update `src/agents/LegalRetrieval.py` so every insufficient/abstain return has `primary_evidence=[]` and does not fall through to rerank fallback evidence.
   - Mark abstain/insufficient metadata explicitly where needed so downstream mapping can distinguish it from a normal empty result.
   - Update `api_server.py` mapping to preserve `primary_evidence=[]` for abstain/insufficient outputs and never promote generic `evidence` into primary evidence in that state.

2. **Planner-driven clarification**
   - Extend `src/agents/QueryDecomposer.py` planner records with the actual `ambiguity_type` and `missing_scope` fields when supplied by the planner, and derive clarification text/options only from those fields and the original query.
   - Remove generic “văn bản hiện hành / dự thảo” options from `api_server.py`; generate targeted options for missing document, provision, taxpayer/scope, or temporal fields. Keep `Khác` as the only generic fallback.
   - Ensure temporal comparison plans with two complete dates produce no clarification; if one date/scope is missing, ask only for that missing field.

3. **Anchor preservation**
   - In `QueryDecomposer.rewrite_clarified_intent`, detect whether `original_sub_query` contains a legal anchor (document number, Điều, Khoản, Điểm). Apply missing-anchor rejection only when at least one such anchor exists; preserve protection against unsupported anchors introduced by the rewrite.
   - Add focused tests for anchored rejection and unanchored rewrite acceptance.

4. **Frontend duplicate-submit guard**
   - Add a stable request fingerprint helper in `frontend/src/services/chatService.ts` based on normalized question, clarification answer, and request kind; exclude `queryTime` and other continuously changing fields.
   - In `frontend/src/App.tsx`, track in-flight fingerprints per session/request flow. Reject only duplicate submissions while the same request is active; clear the fingerprint in `finally` so identical text can be submitted in a later independent turn. Cover both initial submit and clarification confirmation.

5. **Mock tests and verification**
   - Add `tests/test_phase_b.py` with fake/no-op LLM and stub retrieval/API mapping tests for abstain evidence, planner clarification scope/options, anchor behavior, and temporal comparison handling.
   - Add frontend unit-level coverage only if existing test infrastructure supports it; otherwise rely on TypeScript/build verification and keep all backend tests token-free.
   - Run Phase A + Phase B pytest files, Python compilation, `npm --prefix frontend run build`, and `git diff --check`.
   - Update the Phase B checklist in this plan only after all checks pass.

## Critical files
- `src/agents/LegalRetrieval.py`
- `src/agents/QueryDecomposer.py`
- `src/retrieval/QueryRefiner.py` if planner/refiner metadata needs propagation
- `api_server.py`
- `frontend/src/App.tsx`
- `frontend/src/services/chatService.ts`
- `tests/test_phase_b.py`
- `PLAN.md`

## Verification and stop condition
Report exact files changed, mock test/build results, remaining warnings, and two manual commands: one for `api_server.py`, one for `frontend`. Do not start Phase C or launch a server.

## Progress
- [ ] Phase B implementation
- [ ] Phase B mock tests
- [ ] Phase B verification
- [ ] Mark Phase B complete after verification
- [ ] Stop before Phase C
