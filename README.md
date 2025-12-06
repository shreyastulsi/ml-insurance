# Insurance Charges ML Analysis

Minimal steps to run the full pipeline from scratch.

## Clone Repo and CD
```bash
git clone https://github.com/shreyastulsi/ml-insurance.git
cd ml-insurance
```
## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
1) Place `insurance.csv` next to `ml_analysis.py`.  
   - Get it from Kaggle: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
2) Run:
```bash
python ml_analysis.py
```

Outputs (plots, metrics, report) will be created in a fresh `ml_results/` folder.
