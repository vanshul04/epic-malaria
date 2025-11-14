import pandas as pd
from pathlib import Path

IN = Path('data/multi_disease_with_location.csv')
OUT = Path('data/cleaned.csv')
df = pd.read_csv(IN)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# Ensure symptom columns are binary
symptoms = [c for c in df.columns if c not in ('location','disease')]
for s in symptoms:
    df[s] = df[s].fillna(0).astype(int).clip(0,1)

# Standardize disease labels
df['disease'] = df['disease'].astype(str).str.strip().str.title()

# Drop duplicates
df = df.drop_duplicates()

# Save
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print("Saved cleaned dataset to", OUT)
