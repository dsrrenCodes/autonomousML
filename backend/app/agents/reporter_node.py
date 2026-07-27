"""
Reporter Node — deterministic, templated. No LLM call.

Per spec: explicitly thin. The Judge's `justification` field is already
the narration — this node does NOT add a second LLM call, it just
formats what's already in state into readable text for the UI.

Input (from AgentState):  judge_decision, critic_findings
Output (to AgentState):   report — formatted verdict text

Owner: Person B (stub written to unblock graph wiring — Darren owns
this per the ownership table, swap in his version if/when he builds one).
"""


def reporter_node(state: dict) -> dict:
    decision = state.get("judge_decision", {})
    findings = state.get("critic_findings", [])

    verdict = decision.get("verdict", "unknown")
    selected_model = decision.get("selected_model")
    justification = decision.get("justification", "")
    overrode_rules = decision.get("overrode_rules", False)
    rule_based_verdict = decision.get("rule_based_verdict")
    flagged = decision.get("flagged_for_manual_review")

    failing = [f for f in findings if not f.get("passed", True)]
    passing = [f for f in findings if f.get("passed", True)]

    lines = []
    lines.append(f"VERDICT: {verdict.upper()}")
    if selected_model:
        lines.append(f"Selected model: {selected_model}")

    lines.append("")
    lines.append("Critic findings:")
    for f in findings:
        mark = "PASS" if f.get("passed") else "FAIL"
        lines.append(f"  [{mark}] {f.get('test')}: {f.get('detail', '')}")

    if overrode_rules:
        lines.append("")
        lines.append(
            f"NOTE: Judge overrode the rule-based verdict "
            f"({rule_based_verdict} -> {verdict})."
        )
        if flagged:
            lines.append("This override is flagged for manual review.")

    lines.append("")
    lines.append(f"Justification: {justification}")

    report_text = "\n".join(lines)

    return {"report": report_text}