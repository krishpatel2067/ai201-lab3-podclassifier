# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:

- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does

Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter      | Type        | Description                                                |
| -------------- | ----------- | ---------------------------------------------------------- |
| `predictions`  | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`.    |

### Output

| Return value | Type    | Description                  |
| ------------ | ------- | ---------------------------- |
| accuracy     | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
accuracy = # correct / # total samples

A sample is correct if its prediction matches the ground truth.
```

---

**Step-by-step logic:**

```
1. Initialize `correct = 0`.
2. Iterate through `predictions` and `ground_truth` simultaneously.
3. If the current predicted label matches the true label, increment
   `correct`.
4. Return `correct / len(predictions)` as the accuracy (`len(ground_truth)`)
   works as well since both lists are of the same length.
```

---

**Edge case — what if both lists are empty?**

```
If both lists are empty, return early an accuracy of 0.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Correct labels: 0, 1
Correct label count: 2
Total label count: 4
Returned accuracy: 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does

Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter      | Type        | Description                               |
| -------------- | ----------- | ----------------------------------------- |
| `predictions`  | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order.        |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec Fields

**What does "correct" mean for a given class?**

```
A sample in a given class is correctly classified if the predicted class matches
the true class. For example, for the class "interview", only the labels that are
predicted to be "interview" are considered correct.
```

---

**What does "total" mean for a given class?**

```
The total number of samples in a given class from the ground truth, NOT the
number of samples predicted as this class.
```

---

**Step-by-step logic:**

```
1. Initialize the required dictionary called `stats` to return for each valid
   label and with the total, correct, and accuracy all set to 0:
2. For each pair `(predicted, truth)`:
   3. Increment the total count in `stats[truth]`.
   4. If `predicted == truth`, then increment the correct count in `stats[truth]`.
5. Iterate over the dictionary keys to calculate and store the accuracy for each
   label: correct count / total count.
6. Return the dictionary.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set any empty class's accuracy to 0.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0
```

<!-- ---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes? -->
