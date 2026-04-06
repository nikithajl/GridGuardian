from __future__ import annotations

from baseline import run_all_tasks


def main() -> None:
    results = run_all_tasks()
    if len(results) < 6:
        raise SystemExit("Expected at least 6 tasks in this submission.")

    for result in results:
        score = result.final_score
        if not 0.0 <= score <= 1.0:
            raise SystemExit(f"Score for {result.task_id} is out of range: {score}")
        print(
            f"[verify] {result.task_id}: score={result.final_score:.4f}, "
            f"reward={result.total_reward:.4f}, steps={result.steps}"
        )

    print("[verify] All task graders returned scores within [0.0, 1.0].")


if __name__ == "__main__":
    main()
