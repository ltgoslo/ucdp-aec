# Baselines

We detail how to run each baseline below.

## Text2Event

The main modification of Text2Event was the addition of a new `data_format` called `list` that works similar to the standard Text2Event one but does not include an event type nor trigger and does not constrain event arguments to occur in the source sentence.

To run the baseline, first convert the UCDP-AEC dataset to the Text2Event format:
```sh
python data_convert/convert_aec_dataset.py data/text2list/aec "$DATA_PATH/UCDP-AEC"
```

Then follow the Text2Event README for setting up its environment. Note that Text2Event's `requirements.txt` file is underspecified, we slighly modified it to get a working version of their code, but this will certainly break in the future. Furthermore we had to fix a `transformers` bug by changing the version number of `tokenizer` in `$CONDA_PATH/envs/text2event/lib/python3.8/site-packages/transformers/dependency_versions_table.py` to `tokenizers==0.10.3` (this "fix" is also likely to break in the future).
Once the environment is setup, you can start training using the following command:
```sh
./run_seq2seq_verbose.bash -f list -i aec
```

You can then convert the Text2Event predictions files to our AEC format and run the evaluation script (outside the Text2Event environment, from the root of the UCDP-AEC repository):
```sh
python baselines/Text2Event/data_convert/convert_aec_preds.py predictions/Text2Event.test baselines/Text2Event/models/*/test_preds_seq2seq.txt "$DATA_PATH/UCDP-AEC/test"
python aec/evaluate.py "$DATA_PATH/UCDP-AEC/test" predictions/Text2Event.test
```


## DEGREE

Most modifications are inside the `_ucdp` files, we also had to slightly modify `degree/data.py` in order to pass the surface form of the arguments around and not only its position and to clip input documents instead of discarding them. We also had to remove BART limitation on the number of repetitions amongst generated tokens.
Note that in order to get best results, we had to implement some fuzzy template matching, so that arguments can be extracted even if the model's output does not strictly abide by the template.

To run the model, follow the DEGREE's README for environment setup and dataset creation, using `process_ucdp` and the EAE version of the model:
```sh
./scripts/process_ucdp.sh
python degree/generate_data_degree_eae.py -c config/config_degree_eae_ucdp.json
python degree/train_degree_eae.py -c config/config_degree_eae_ucdp.json
```

Then convert the predictions to our `jsonl` format and evaluate:
```sh
python baselines/DEGREE/convert_predictions.py predictions/DEGREE.test baselines/DEGREE/output/degree_eae_ucdp/*/pred.test.json "$DATA_PATH/UCDP-AEC/test"
python aec/evaluate.py "$DATA_PATH/UCDP-AEC/test" predictions/DEGREE.test
```


## T5

We fine-tuned several Flan-T5 (instruction finetuned version of T5) checkpoints of different sizes: flan-t5-small, flan-t5-base and flan-t5-large without any special modifications. Our input template is passed as input text and the output template as output text. The special token `<extra_id_0>` is used to separate arguments and later capturing them after being decoded.

Again, follow the README inside of T5 directory for setup. Remember to specify a `DATA_PATH`, `MODEL_PATH` and `LOG_PATH`. Resulting predictions stored in `LOG_PATH` can be adapted to the format for our evaluation script through `process_output`. This scripts takes a directory containing .jsonl validation and test set log files produced by the model after running it.
```
python flan-t5/t5/process_output.py predictions
python aec/evaluate.py "$DATA_PATH/UCDP-AEC/test" predictions/T5.test
```
