#!/bin/bash

# Run training
# Reads data from 'data/' directory, saves model to 'outputs/ner_model'
python train.py \
    --dataset_path "data" \
    --model_save_path "outputs/ner_model" \
    --num_train_epoch 3 \
    --batch_size 4 \
    --learning_rate 1e-4

# Run pipeline
# Loads model from 'outputs/ner_model', reads the test split created by train.py, saves predictions
python pipeline.py \
    --model_load_path "outputs/ner_model" \
    --input_file "data/dataset1_test_split.json" \
    --output_file "outputs/predictions.json"
