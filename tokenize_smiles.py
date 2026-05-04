import deepchem as dc
import os

# --- BLOCK 1: Setup Tokenizer ---
print("Initializing SMILES Tokenizer...")
tokenizer = dc.feat.BasicSmilesTokenizer()

# --- BLOCK 2: Load SMILES from ESOL ---
print("Loading ESOL SMILES...")
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='Raw', data_dir='./esol_data')
train_dataset, valid_dataset, test_dataset = datasets

sample_smiles = train_dataset.ids[:5]

# --- BLOCK 3: Tokenization ---
print("\n" + "="*40)
print("TOKENIZATION RESULTS")
print("="*40)

for smiles in sample_smiles:
    tokens = tokenizer.tokenize(smiles)
    print(f"SMILES: {smiles}")
    print(f"Tokens: {tokens}")
    print(f"Token Count: {len(tokens)}")
    print("-" * 40)