"""Format AEC-t5 output data for evaluate.py. Takes a Path containing .jsonl files as argument."""
import json
import argparse
import pathlib


def process_output_t5_jsonl_files(directory: pathlib.Path):
    for filepath in directory.glob("*.jsonl"):
        # Ignore newly processed outputs
        if filepath.name.endswith("_evaluate.jsonl"):
            continue
        # Declare write path
        output_path = filepath.with_name(filepath.stem + "_evaluate.test")

        with (filepath.open('r', encoding='utf-8') as infile, output_path.open('w', encoding='utf-8') as outfile):
            for line in infile:
                # Read each line containing sample+prediction
                data = json.loads(line.strip())
                # Format prediction dictionary and include sample identifier
                try:
                    prediction = data["prediction"]
                    prediction["id"] = data["sample"]["id"]
                    del prediction["generated_text"]
                except KeyError:
                    print(f"{filepath.name} doesn't match the AEC-t5 output format. A sample and a prediction dictionary are expected for each line.")
                    return
                # Ensure all numeric (deaths) values are converted to integers
                prediction = {k: int(v) if  k.startswith("deaths_") and isinstance(v, str) and v.isdigit() else v for k, v in prediction.items()}
                for field in prediction:
                    # Force 0 in case of string.
                    # For all runs, strings have consistently returned hallucinations and not potentially correct results such as "5 people"
                    if field.startswith("deaths_") and isinstance(prediction[field], str):
                        prediction[field] = 0
                json.dump(prediction, outfile)
                outfile.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AEC-t5 output .jsonl files in a given directory.")
    parser.add_argument("directory", type=str, help="Path to the directory containing AEC-t5 output .jsonl files.")
    args: argparse.Namespace = parser.parse_args()

    process_output_t5_jsonl_files(pathlib.Path(args.directory))
