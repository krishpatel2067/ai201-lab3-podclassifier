# AI201 Lab 3 — Pod Classifier

A few-shot podcast episode classifier. Given an episode description, classifies it
as `interview`, `solo`, `panel`, or `narrative` using labeled examples and an LLM.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate     # Mac/Linux
# or: .venv\Scripts\activate  # Windows

pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

## Run

```bash
python app.py
```

## Project Structure

```
root/
├── app.py              # Gradio UI
├── classifier.py       # Few-shot classification logic
├── evaluate.py         # Evaluation metrics
├── config.py           # Settings and constants
├── requirements.txt
├── .env.example
├── data/
│   ├── train_episodes.json   # 20 labeled episodes
│   ├── test_episodes.json    # 20 pre-labeled episodes (held-out test set)
│   ├── my_labels.json        # Labeled episodes matching `train_episodes.json`
│   └── taxonomy.md           # Label definitions and edge cases
└── specs/
    ├── system-design.md      # Architecture overview
    ├── classifier-spec.md    # Spec for classifier.py
    └── evaluation-spec.md    # Spec for evaluate.py
```
