# Data

This folder is intentionally empty in the repository. The dataset 
files are too large for GitHub and are therefore gitignored.

## How to Reproduce the Master Dataset

Download the following three datasets from Kaggle and place the 
extracted files directly in this folder:

**1. Primary dataset**
https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset
Files needed: perfumes.csv, perfumes.jsonl

**2. FragDB reference tables**
https://www.kaggle.com/datasets/eriklindqvist/fragdb-fragrance-database
Files needed: notes.csv, accords.csv, brands.csv, perfumers.csv

**3. Supplementary dataset**
https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes
Files needed: fra_perfumes.csv

Once all files are in this folder, open and run the notebook at:
notebooks/beyond_fragrancy_data_pipeline.ipynb

Running all cells top to bottom will generate:
- master_dataset.csv
- master_dataset_lean.csv

These two files are what the app uses for recommendations.