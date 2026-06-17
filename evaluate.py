import json
import os
from config import VALID_LABELS, DATA_PATH, TEST_FILE
from classifier import classify_episode, load_labeled_examples


def run_evaluation() -> dict:
    """
    Run the classifier against the held-out test set and return full results.

    This function:
      1. Loads the labeled training examples (from `my_labels.json`)
      2. Loads the test episodes (with ground-truth labels)
      3. Runs classify_episode() on each test description
      4. Returns a results dict with predictions, ground truth, and per-episode detail
    """
    labeled_examples = load_labeled_examples()

    test_path = os.path.join(DATA_PATH, TEST_FILE)
    with open(test_path, encoding="utf-8") as f:
        test_episodes = json.load(f)

    results = []
    for episode in test_episodes:
        print(f"  Classifying: {episode['title'][:60]}...")
        prediction = classify_episode(episode["description"], labeled_examples)

        if prediction["label"] == "unknown":
            print(prediction)

        results.append(
            {
                "id": episode["id"],
                "title": episode["title"],
                "description": episode["description"],
                "ground_truth": episode["label"],
                "predicted": prediction["label"],
                "reasoning": prediction["reasoning"],
                "correct": prediction["label"] == episode["label"],
            }
        )

    predictions = [r["predicted"] for r in results]
    ground_truth = [r["ground_truth"] for r in results]

    return {
        "results": results,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "total": len(results),
    }


def compute_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    """
    Computes overall classification accuracy.
    """
    if not ground_truth:
        return 0.0

    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(ground_truth)


def compute_per_class_accuracy(
    predictions: list[str], ground_truth: list[str]
) -> dict[str, dict]:
    """
    Computes accuracy broken down by each label class.

    Returns a dict keyed by label. Example:
      {
        "interview": {"correct": 4, "total": 5, "accuracy": 0.8},
        "solo":      {"correct": 5, "total": 5, "accuracy": 1.0},
        ...
      }
    """
    stats = {
        label: {"correct": 0, "total": 0, "accuracy": 0.0} for label in VALID_LABELS
    }

    for p, g in zip(predictions, ground_truth):
        if g in stats:
            stats[g]["total"] += 1
            if p == g:
                stats[g]["correct"] += 1

    for label in stats:
        if stats[label]["total"] > 0:
            stats[label]["accuracy"] = stats[label]["correct"] / stats[label]["total"]

    return stats


def format_evaluation_report(eval_results: dict) -> str:
    """
    Formats evaluation results into a readable report string.
    """
    predictions = eval_results["predictions"]
    ground_truth = eval_results["ground_truth"]
    results = eval_results["results"]

    accuracy = compute_accuracy(predictions, ground_truth)
    per_class = compute_per_class_accuracy(predictions, ground_truth)

    lines = [
        f"## Evaluation Results\n",
        f"**Overall accuracy:** {accuracy:.1%} ({sum(r['correct'] for r in results)}/{eval_results['total']})\n",
        "\n**Per-class accuracy:**",
    ]
    for label, stats in per_class.items():
        bar = "█" * int(stats["accuracy"] * 10) + "░" * (
            10 - int(stats["accuracy"] * 10)
        )
        lines.append(
            f"  {label:<12} {bar}  {stats['accuracy']:.0%}  ({stats['correct']}/{stats['total']})"
        )

    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        lines.append(f"\n**Misclassified ({len(misclassified)}):**")
        for r in misclassified:
            lines.append(f"  [{r['ground_truth']} → {r['predicted']}] {r['title']}\n")
    else:
        lines.append("\n**No misclassifications — perfect score!**")

    return "\n".join(lines)
