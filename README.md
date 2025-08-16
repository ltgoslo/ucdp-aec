# UCDP-AEC (Abstractive Event analysis Corpus)

This repository contains the software and data associated with the paper:<br>
**Abstractive Event Analysis of Armed Conflicts: Introducing the UCDP-AEC Dataset**

## Dataset Preparation

The [data directory](data/UCDP-AEC-ids) contains the dataset splits in two formats: huggingface datasets (e.g. `datasets.load_from_disk("train")`) and `jsonl`.
In both cases, the `source_article` field contains [HPLT](https://hplt-project.org/datasets/v2.0) document IDs.
The easiest way to work with this dataset is to first replace those IDs by the actual HPLT documents.
To that end the following script is provided:
```sh
python aec/ids_to_documents.py data/UCDP-AEC data/UCDP-AEC-ids
```
If you don't want to install huggingface datasets library, you can convert only the `jsonl` files by adding the `-J` argument to that command:
```sh
python aec/ids_to_documents.py -J data/UCDP-AEC data/UCDP-AEC-ids
```

## Evaluation Script

For model evaluation, generate a jsonl files with one prediction per line such as:
```json
{"id": 442069, "side_a_name": "Government of Myanmar (Burma)", "side_b_name": "ULA", "start_date": "2022-05-26", "end_date": "2022-05-26", "location_root_name": "Myanmar (Burma)", "location_adm1_name": "Chin state", "location_adm2_name": "Mindat district", "location_where_name": "Paletwa town", "deaths_side_a": 2, "deaths_side_b": 0, "deaths_civilian": 0, "deaths_unknown": 0, "deaths_low": 2, "deaths_high": 3}
```
Note that id and deaths fields are typed as integers, everything else is typed as strings.

It's a good practice to drop those fields (except `id`) from the test set after loading it to make sure you're using generate and not teacher forcing. Then use [aec/evaluate.py](aec/evaluate.py) to evaluate the model.

## Other Code Released

The `hplt_align` directory contains code used for HPLT document matching.

The `analysis` directory contains scripts we used to generate the statistics given in the paper.

The `baselines` directory contains model code used in the experiments, some subdirectories are modified version of existing code: [Text2Event](https://github.com/luyaojie/Text2Event) and [DEGREE](https://github.com/PlusLabNLP/DEGREE). See the dedicated [README](baselines/README.md) for details on how to run the models.

## Licence

We release all the code in this repository under GNU AGPL licence with the exception of the content of the `baselines/Text2Event` and `baselines/DEGREE` directories which keep the licences of their original authors. We release our modifications to these directories under the same licence as the original code, that is MIT for Text2Event and Apache 2.0 for DEGREE.
