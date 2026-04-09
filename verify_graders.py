from __future__ import annotations

from baseline import detect_base_url, run_task_locally, run_task_remote
from tasks import list_tasks


def main() -> None:
    base_url = detect_base_url()
    task_specs = list_tasks()
    if len(task_specs) < 6:
        raise SystemExit("Expected at least 6 tasks in this submission.")

    if base_url:
        print(f"[verify] using remote base url: {base_url}")
    else:
        print("[verify] using local in-process environment")

    results = []
    for index, task in enumerate(task_specs, start=1):
        print(f"[verify] running task {index}/{len(task_specs)}: {task.task_id}", flush=True)
        if base_url:
            result = run_task_remote(task.task_id, base_url=base_url)
        else:
            result = run_task_locally(task.task_id)
        results.append(result)

    for result in results:
        score = result.final_score
        if not 0.0 < score < 1.0:
            raise SystemExit(f"Score for {result.task_id} must be strictly between 0 and 1: {score}")
        print(
            f"[verify] {result.task_id}: score={result.final_score:.4f}, "
            f"reward={result.total_reward:.4f}, steps={result.steps}"
        )

    print("[verify] All task graders returned scores strictly within (0.0, 1.0).")


if __name__ == "__main__":
    main()
