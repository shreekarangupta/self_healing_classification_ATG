# Self-Healing Classification DAG - Task 2

## Description
This project implements a sentiment classification pipeline using a DAG structure with fallback logic.
- Fine-tuned DistilBERT model on IMDb dataset using LoRA method.
- DAG handles main model classification and fallback rule-based classification.
- CLI interface for demo interaction.

## Files:
- `fine_tune.py`: Script to fine-tune the model (run once)
- `classify_with_dag.py`: Runs the DAG classification and CLI demo
- `requirements.txt`: List of required Python packages
- `fine_tuned_model/`: Saved pretrained model and tokenizer

## Setup Instructions:
1. Create a Python virtual     environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

2. Install dependencies:

   pip install -r requirements.txt

3.To run the fine-tuning (optional if model already trained):

python fine_tune.py

4.To run the classification demo:

python classify_with_dag.py

5.Enter reviews in CLI to see predictions. 
Then enter text reviews like:

"The movie was amazing"

"It was the worst film ever"

Use Ctrl+C to exit.

Notes:
Model fine-tuning takes time; please run fine_tune.py once before running the demo.

The DAG uses fallback logic to handle unknown or uncertain predictions.

No external checkpointing module required; memory-based runtime state is used.