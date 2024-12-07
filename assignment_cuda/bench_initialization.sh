#!/bin/bash

# Array of datasets to iterate over
DATASETS=("SMALL_DATASET" "STANDARD_DATASET" "LARGE_DATASET" "EXTRALARGE_DATASET")

# Loop through each dataset
for dataset in "${DATASETS[@]}"
do
  echo "Running make for dataset $dataset"
  
  # Execute make command with specified dataset
  make initialization.cu EXT_CFLAGS="-D $dataset" clean all run
  
  echo "Completed make for dataset $dataset"
  echo "----------------------------------------"
done

echo "All builds and runs have been completed."
