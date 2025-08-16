from typing import List

from extraction.event_schema import EventSchema
from extraction.predict_parser.predict_parser import Metric
from extraction.predict_parser.tree_predict_parser import TreePredictParser, ListPredictParser
from data_convert.ucdp_fields import UCDP_FIELDS

decoding_format_dict = {
    'tree': TreePredictParser,
    'list': ListPredictParser,
    'treespan': TreePredictParser,
}


def get_predict_parser(format_name):
    return decoding_format_dict[format_name]


def eval_pred(predict_parser, gold_list, pred_list, text_list=None, raw_list=None):
    well_formed_list, counter = predict_parser.decode(
        gold_list, pred_list, text_list, raw_list)

    event_metric = Metric()
    role_metric = Metric()
    ucdp_metric = {role: 0 for role, _, _ in UCDP_FIELDS}

    for instance in well_formed_list:
        event_metric.count_instance(instance['gold_event'],
                                    instance['pred_event'])
        role_metric.count_instance(instance['gold_role'],
                                   instance['pred_role'],
                                   verbose=False)
        gold_dict = {role: argument for etype, role, argument in instance["gold_role"]}
        pred_dict = {role: argument for etype, role, argument in instance["pred_role"]}
        for role, _, _ in UCDP_FIELDS:
            if role not in gold_dict:
                continue
            if gold_dict[role] == pred_dict.get(role):
                ucdp_metric[role] += 1

    for role, count in ucdp_metric.items():
        ucdp_metric[role] = count / len(well_formed_list)
    ucdp_agg = {category: 0 for _, _, category in UCDP_FIELDS}
    for role, _, category in UCDP_FIELDS:
        ucdp_agg[category] += ucdp_metric[role]
    for category, total in ucdp_agg.items():
        ucdp_agg[category] = total / [category for _, _, category in UCDP_FIELDS].count(category)
    aggregate = sum(ucdp_agg.values()) / len(ucdp_agg)
    ucdp_metric.update(ucdp_agg)
    ucdp_metric["aggregate"] = aggregate


    trigger_result = event_metric.compute_f1(prefix='trigger-')
    role_result = role_metric.compute_f1(prefix='role-')

    result = dict()
    result.update(trigger_result)
    result.update(role_result)
    result['AVG-F1'] = trigger_result.get('trigger-F1', 0.) + \
        role_result.get('role-F1', 0.)
    result.update(counter)

    for role, accuracy in ucdp_metric.items():
        result["UCDP-" + role + "-acc"] = accuracy

    return result


def get_extract_metrics(pred_lns: List[str], tgt_lns: List[str], label_constraint: EventSchema, decoding_format='tree'):
    predict_parser = get_predict_parser(format_name=decoding_format)(
        label_constraint=label_constraint)
    return eval_pred(
        predict_parser=predict_parser,
        gold_list=tgt_lns,
        pred_list=pred_lns
    )
