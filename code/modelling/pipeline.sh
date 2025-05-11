#!/bin/bash

# Step 1: Run embedding_generator.py
echo "Step 1/4: Generating embeddings..."
if ! python embedding_generator.py; then
    echo "Error in Step 1" >&2
    exit 1
fi

# Step 2: Modify config.yaml to set load_from_local to True
echo "Step 2/4: Updating config.yaml..."
# Modify the YAML file
sed -i 's/load_from_local: false/load_from_local: true/g' config.yaml
sed -i 's/save_to_local: true/save_to_local: false/g' config.yaml

echo "Verification: load_from_local is now set to:"
grep "load_from_local:" config.yaml

# Step 3: Run tune_hdbscan.py
echo "Step 3/4: Tuning HDBSCAN..."
python tune_hdbscan.py

# Step 4: Run main.py for clustering
echo "Step 4/4: Clustering embeddings..."
python main.py

echo "Pipeline completed successfully!"