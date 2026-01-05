@echo off

REM Run training
echo Starting training...
python train.py ^
    --dataset_path "data" ^
    --model_save_path "outputs/ner_model" ^
    --num_train_epoch 3 ^
    --batch_size 4 ^
    --learning_rate 1e-4

IF %ERRORLEVEL% NEQ 0 (
    echo Training failed!
    exit /b %ERRORLEVEL%
)

REM Run pipeline
echo Starting pipeline...
python pipeline.py ^
    --model_load_path "outputs/ner_model" ^
    --input_file "data/dataset1_test_split.json" ^
    --output_file "outputs/predictions.json"

IF %ERRORLEVEL% NEQ 0 (
    echo Pipeline failed!
    exit /b %ERRORLEVEL%
)

echo Done!
