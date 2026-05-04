import deepchem as dc
import os

os.chdir('/home/kamakshi.rautela/NeuroChem_Minor')

# Load Lipophilicity dataset
print("Loading Lipophilicity dataset...")
tasks, datasets, transformers = dc.molnet.load_lipo(featurizer='Raw', data_dir='./lipo_data')

train_ds, valid_ds, test_ds = datasets

print(f"\n{'='*50}")
print("LIPOPHILICITY DATASET INFORMATION")
print(f"{'='*50}")
print(f"Tasks: {tasks}")
print(f"Number of Tasks: {len(tasks)}")
print(f"\nTrain samples: {len(train_ds)}")
print(f"Valid samples: {len(valid_ds)}")
print(f"Test samples: {len(test_ds)}")
print(f"\nTrain data shape (X): {train_ds.X.shape if hasattr(train_ds.X, 'shape') else 'N/A'}")
print(f"Train labels shape: {train_ds.y.shape}")
print(f"Train label type: {type(train_ds.y[0])}")
print(f"\nFirst 5 SMILES:")
for i, smiles in enumerate(train_ds.ids[:5]):
    print(f"  {i+1}. {smiles}")
print(f"\nFirst 5 labels: {train_ds.y[:5].flatten()}")
