"""Define a simple template for generative models."""

transformer = "google/flan-t5-small"

dataset = {
    "name": "UCDP-AEC",
    "processing": "generative",
    "tokenizer": transformer,
    "input_template": """
{source_date} <extra_id_0>
{source_article}
""".strip(),
    "output_template": """
side A: {side_a_name} <extra_id_0>
side B: {side_b_name} <extra_id_0>
start date: {start_date} <extra_id_0>
end date: {end_date} <extra_id_0>
country: {location_root_name} <extra_id_0>
ADM1: {location_adm1_name} <extra_id_0>
ADM2: {location_adm2_name} <extra_id_0>
where: {location_where_name} <extra_id_0>
deaths side A: {deaths_side_a} <extra_id_0>
deaths side B: {deaths_side_b} <extra_id_0>
deaths civilian: {deaths_civilian} <extra_id_0>
deaths unknown: {deaths_unknown} <extra_id_0>
deaths low: {deaths_low} <extra_id_0>
deaths high: {deaths_high}
""".strip().replace("\n", " "),
}

model = {
    "transformer": transformer,
    "generation_max_length": 150,
}
