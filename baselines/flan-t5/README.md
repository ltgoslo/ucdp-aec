## Environment
- Python==3.11
- datasets==3.2.0
- torch==2.6.0
- transformers==4.48.3
- accelerate==1.3.0
- tensorboard==2.18.0

## Datasets

This setup is designed for UCDP-AEC, with input and output templates defined in `spec/aec_setup.py`. Templates can be modified to adapt to different event argument structures.

## Training

Training parameters can be specified either in `aec_setup.py` or provided as command-line arguments (--parameters). The following configuration options are available:
- `eval`: Whether to only run evaluation.
- `transformer`: Defines the HF model to run. Must be any of T5 family.
- `num_train_epochs`: Number of training epochs.
- `learning_rate`: The initial learning rate.
- `model.generation_max_length`: Maximum number of tokens to generate during evaluation.
- `dataset.name`: Name of the dataset to train on. It must be a directory in `$DATA_PATH`.
- `dataset.input_template`: Template to use for the generative representation of input.
- `dataset.output_template`: Template to use for the generative representation of output.

Before starting, ensure the following environment variables are set:
- `DATA_PATH`: Root directory for dataset arrow files.
- `MODEL_PATH`: Directory for model checkpoints.
- `LOG_PATH`: Directory for evaluation results.

Execute training with:
```sh
python -m t5 t5/spec/aec_setup.py [--parameters]
```

## Evaluation

If everything went correctly, evaluation results will be saved in `LOG_PATH` under a directory named after the run's parameters and timestamp, containing the files `validation_output.jsonl` and `test_output.jsonl`, which can be used with `process_output.py` to format results for `aec/evaluate.py`.
