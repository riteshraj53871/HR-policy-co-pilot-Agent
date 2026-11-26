from app.graph.builder import ask

CASES = [
    ("How many annual leave days do employees receive?", "kb, no retries"),
    ("How many sick days do I get?", "kb"),
    ("Can I work from home?", "kb (weak score, so the LLM grader decides)"),
    ("I did not receive my salary, what should I do?", "kb: Payroll, NOT the HR-only runbook"),
    ("What is the process for a new employee's first day?", "none: company-specific, so NO web, rewrites then give up"),
    ("What are the latest public holiday rules in Bangladesh?", "web: source web, URLs, disclaimer"),
    ("Hello!", "direct"),
    ("How many leave days do I have left?", "direct: out_of_scope message"),
    ("What is the capital of France?", "direct: out_of_scope message"),
]

for question, expected in CASES:
    print("\n" + "=" * 70)
    print(f"Q: {question}\n   expect: {expected}")
    state = ask(question)
    path = " -> ".join(t["node"] for t in state["trace"])
    print(f"   path:   {path}")
    print(f"   source: {state['source_type']}   retries used: {state['retry_count']}")
    for t in state["trace"]:
        if t["node"] in ("route_question", "grade_private", "rewrite_query", "web_search", "grade_web"):
            detail = {k: v for k, v in t.items() if k != "node"}
            print(f"     {t['node']}: {detail}")
    print(f"   answer: {state['answer']}")
    for c in state["citations"]:
        print(f"   cite:   {c['title']} > {c.get('section', 'N/A')} ({c.get('score', 'N/A')})")