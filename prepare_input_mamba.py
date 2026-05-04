import deepchem as dc
import numpy as np

# 1. Load Data
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='Raw', data_dir='./esol_data')
all_smiles = datasets[0].ids # Training SMILES

# 2. Tokenize
tokenizer = dc.feat.BasicSmilesTokenizer()
tokenized_smiles = [tokenizer.tokenize(s) for s in all_smiles]

# 3. Create Custom Vocabulary (The Alphabet)
# Saare unique characters dhundo jo hamare dataset mein hain
unique_tokens = sorted(list(set([t for sublist in tokenized_smiles for t in sublist])))
token_to_id = {t: i+1 for i, t in enumerate(unique_tokens)} # i+1 taaki 0 padding ke liye rahe
token_to_id['<PAD>'] = 0

print(f"Vocabulary Size: {len(token_to_id)}")
print(f"Unique Tokens: {unique_tokens}")

# 4. Convert to Fixed-Length Sequences
MAX_LEN = 64
encoded_data = []

for tokens in tokenized_smiles:
    # Strings ko IDs mein badlo
    ids = [token_to_id[t] for t in tokens[:MAX_LEN]]
    # Padding (0 add karna)
    ids += [0] * (MAX_LEN - len(ids))
    encoded_data.append(ids)

encoded_data = np.array(encoded_data)
print(f"\nShape of Mamba Input: {encoded_data.shape}") # (902, 64)
print(f"First molecule encoded: {encoded_data[0]}")