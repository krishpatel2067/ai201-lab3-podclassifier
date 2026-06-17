# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does

Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter          | Type         | Description                                                                                                          |
| ------------------ | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description`      | `str`        | The episode description to classify.                                                                                 |

### Output

| Return value | Type  | Description                                        |
| ------------ | ----- | -------------------------------------------------- |
| prompt       | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output.
```

---

**What output format should you request from the LLM?**

```
A simple output format makes the response parsable:

Label: {label}
Reasoning: {reasoning}

This format balances between being minimalistic (e.g. just 2 lines without
"Label" or "Reasoning" markers), which is token efficient but unreliable to
parse due to the possibility of extra words, and overly structured (e.g. JSON),
which is not token efficient but reliable to parse.
```

---

**Edge cases to handle in the prompt:**

```
If `labeled_examples` is empty, return an empty string and/or an error message,
stating that this parameter cannot be empty (since the model can't "learn"
without examples).

An overly short description of a new episode to classify doesn't give the model
much to work with even with ample labels and guidelines. To combat very short
descriptions, use a word count threshold under which to reject the description,
returning an empty string and/or a helpful errorm essage.
```

---

## classify_episode(description, labeled_examples)

### What it does

Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter          | Type         | Description                                               |
| ------------------ | ------------ | --------------------------------------------------------- |
| `description`      | `str`        | The episode description to classify.                      |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type   | Description                                                                                         |
| ------------ | ------ | --------------------------------------------------------------------------------------------------- |
| result       | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
Rely on regex to parse for the output format specified above. Strip
whitespace and perform case-insensitive parsing to allow some flexibility in
the format. If this doesn't work, directly try scanning for one of the four
labels in the 1st line (the 2nd line has the description, which may have
multiple of those valid labels).
```

---

**Step 4 — Validate the label:**

```
Set the label to "unknown" if the parsed content doesn't contain a valid label.
```

---

**Step 5 — Handle errors gracefully:**

```
Use "unknown" as a catchall for unparsable label and any LLM API errors. When
the label is "unknown", use the "reasoning" field to send back error messages,
e.g. "Error: unparsable label" or "Error: LLM API call failed".
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM or error message if
                       # "unknown" label
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

_Fill this in after implementing and testing both functions._

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested:
"Marine Biologist Dr. Amara Diallo on What Coral Bleaching Actually Looks Like"

Raw response text:
Label: interview
Reasoning: The episode features a conversation between the host and Dr. Amara
Diallo, a guest with expertise in coral reefs, indicating a one-on-one
discussion that fits the definition of an interview.
```

**How did you parse the label out of the response?**

```
1. `string.strip()` - strip each line individually and store non-empty lines.
2. `re.match()` - Perform case-insensitive regex matching for each line
   separately
3. `match.group(1)` - Get the capture groups to extract the label and reasoning
   from the separate Match objects
```

**Did any episodes return `"unknown"`? If so, why?**

```
No, mostly because the LLM temperature is set to 0 when prompting. But I'm sure
that further (automated) testing could surface some results with deviant
formatting.
```

**One thing about the output format that surprised me:**

```
I was surprised to learn that the simplest output format without markers, like

{label}
{reasoning}

isn't very robust when parsing due to LLM noise ("chatter" words that it may
sometimes add unnecessarily). The model has to rely solely on position to follow
the format. However, markers help by providing more semantic structure and
"anchoring" the model:

Label: {label}
Reasoning: {reasoning}
```
