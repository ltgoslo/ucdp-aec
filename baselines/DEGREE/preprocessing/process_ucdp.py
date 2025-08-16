import json
from argparse import ArgumentParser
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer


ROLES = [
	 "side_a_name",
	 "side_b_name",
	 "start_date",
	 "end_date",
	 "location_root_name",
	 "location_adm1_name",
	 "location_adm2_name",
	 "location_where_name",
	 "deaths_side_a",
	 "deaths_side_b",
	 "deaths_civilian",
	 "deaths_unknown",
	 "deaths_low",
	 "deaths_high",
]

ROLE_TO_TYPE = {
	 "side_a_name": "Actor",
	 "side_b_name": "Actor",
	 "start_date": "Date",
	 "end_date": "Date",
	 "location_root_name": "Place",
	 "location_adm1_name": "Place",
	 "location_adm2_name": "Place",
	 "location_where_name": "Place",
	 "deaths_side_a": "Deaths",
	 "deaths_side_b": "Deaths",
	 "deaths_civilian": "Deaths",
	 "deaths_unknown": "Deaths",
	 "deaths_low": "Deaths",
	 "deaths_high": "Deaths",
}


def convert(input_file, output_file, tokenizer):
    with open(input_file, "r", encoding="utf-8") as r, open(output_file, "w", encoding="utf-8") as w:
        for line in r:
            sample = json.loads(line)
            entities = [
                    {
                        "id": f"{sample['id']}-{irole}",
                        "start": 2*irole,
                        "end": 2*irole+1,
                        "entity_type": ROLE_TO_TYPE[role],
                        "mention_type": "UNK",
                        "text": str(sample[role]),
                    }
                    for irole, role in enumerate(ROLES)
                ]
            event_mentions = [{
                    "event_type": "UCDPDeath",
                    "id": str(sample["id"]),
                    "trigger": {
                        "start": 0,
                        "end": 1,
                        "text": "UNK",
                    },
                    "arguments": [
                        {
                            "entity_id": f"{sample['id']}-{irole}",
                            "role": role,
                            "text": str(sample[role]),
                        }
                        for irole, role in enumerate(ROLES)
                    ]
                }]
            input_text = f"{sample['source_date']} {sample['source_article']}"
            input_tokenized = tokenizer.tokenize(input_text)
            print(json.dumps({
                    "doc_id": sample["id"],
                    "wnd_id": 0,
                    "entity_mentions": entities,
                    "relation_mentions": [],
                    "event_mentions": event_mentions,
                    "entity_coreference": [],
                    "event_coreference": [],
                    "pieces": input_tokenized,
                    "tokens": [input_text],
                    "token_lens": [len(input_tokenized)],
                    "sentence": input_text,
                }), file=w)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input dataset split')
    parser.add_argument('-o', '--output', help='Path to the output file')
    parser.add_argument('-b', '--bert', help='BERT model name', default='bert-large-cased')
    args = parser.parse_args()
    model_name = args.bert
    if model_name.startswith('bert-'):
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert,
                                                       do_lower_case=False)
    elif model_name.startswith('roberta-'):
        bert_tokenizer = RobertaTokenizer.from_pretrained(args.bert,
                                                       do_lower_case=False)
    else:
        bert_tokenizer = AutoTokenizer.from_pretrained(args.bert, do_lower_case=False, use_fast=False)
    
    convert(args.input, args.output, bert_tokenizer)
