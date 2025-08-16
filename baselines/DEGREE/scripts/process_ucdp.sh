export UCDP_PATH="./UCDP-AEC"
export OUTPUT_PATH="./processed_data/ucdp_bart"

mkdir $OUTPUT_PATH

python preprocessing/process_ucdp.py -i "$UCDP_PATH/train.jsonl" -o $OUTPUT_PATH/train.oneie.json -b facebook/bart-large
python preprocessing/process_ucdp.py -i "$UCDP_PATH/validation.jsonl" -o $OUTPUT_PATH/dev.oneie.json -b facebook/bart-large
python preprocessing/process_ucdp.py -i "$UCDP_PATH/test.jsonl" -o $OUTPUT_PATH/test.oneie.json -b facebook/bart-large
