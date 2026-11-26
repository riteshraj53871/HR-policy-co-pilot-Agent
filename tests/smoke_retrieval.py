from app.services import embeddings, vector_store

EMPLOYEE_FILTER = {"visibility": {"$eq": "all_employees"}}

# (question, expectation)
QUESTIONS = [
    ("How many annual leave days do employees receive?", "Annual Leave, strong"),
    ("How many sick days do I get?", "Sick Leave, strong"),
    ("Can I work from home?", "Remote Work Policy, strong"),
    ("What time does the work day start?", "Working Hours and Attendance, strong"),
    ("When is salary paid?", "Payroll, strong"),
    ("What insurance does the company provide?", "Benefits, strong"),
    ("I did not receive my salary, what should I do?", "Payroll, and NO hr_only row"),
    ("What is the process for a new employee's first day?", "weak (runbook is HR-only)"),
    ("What are the latest public holiday rules in Bangladesh?", "weak, should go to web"),
    ("What is the capital of France?", "weakest"),
]


def show(label: str, results: list[dict]) -> None:
    print(f"  {label}")
    for r in results:
        m = r["metadata"]
        print(f"    {r['score']:.3f}  [{m['visibility']}] {m['title']} > {m['heading_path']}")


def main() -> None:
    stats = vector_store.stats()
    ns = stats.get("namespaces", {})
    print(f"Index vectors: {stats.get('total_vector_count')}, namespaces: {list(ns)}")
    if not stats.get("total_vector_count"):
        print("Index is empty. Run ingestion first (new vectors can take 10-30s to appear).")
        return

    leaks = 0
    for question, expected in QUESTIONS:
        print(f"\nQ: {question}\n  expect: {expected}")
        vec = embeddings.embed_query(question)

        employee_view = vector_store.query(vec, top_k=3, filter=EMPLOYEE_FILTER)
        show("employee view (filtered)", employee_view)
        leaks += sum(1 for r in employee_view if r["metadata"]["visibility"] != "all_employees")

        if "salary" in question and "not receive" in question:
            show("for comparison, UNFILTERED", vector_store.query(vec, top_k=3))

    print("\n" + "=" * 60)
    print("ACCESS CHECK:", "PASS, no hr_only chunk reached the employee view"
          if leaks == 0 else f"FAIL, {leaks} hr_only chunks leaked")


if __name__ == "__main__":
    main()