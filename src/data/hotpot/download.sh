#!/bin/bash

# Download Hotpot Data
curl -O http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

# Generate corpus from Wikipedia
python utils/generate_corpus.py