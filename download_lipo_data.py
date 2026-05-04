import deepchem as dc
import os

print("Step 1: Download raw Lipophilicity data on login node...")
print("This will attempt internet download and cache locally.")

os.chdir('/home/kamakshi.rautela/NeuroChem_Minor')

# Download with Raw featurizer (just SMILES)
try:
    tasks, datasets, transformers = dc.molnet.load_lipo(
        featurizer='Raw',
        data_dir='./lipo_data'
    )
    print("✓ Raw data downloaded successfully")
except Exception as e:
    print(f"✗ Error downloading raw: {e}")

# Download with RDKit descriptors
try:
    featurizer = dc.feat.RDKitDescriptors()
    tasks, datasets, transformers = dc.molnet.load_lipo(
        featurizer=featurizer,
        data_dir='./lipo_data_desc'
    )
    print("✓ RDKit descriptors featurized successfully")
except Exception as e:
    print(f"✗ Error featurizing: {e}")

print("\nData is now cached locally. Safe to use on compute nodes.")
