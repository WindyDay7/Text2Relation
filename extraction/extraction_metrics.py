from typing import List

from extraction.event_schema import EventSchema
from extraction.predict_parser.predict_parser import Metric
from extraction.predict_parser.tree_predict_parser import TreePredictParser

decoding_format_dict = {
    'tree': TreePredictParser,
    'treespan': TreePredictParser,
}


def get_predict_parser(format_name):
    return decoding_format_dict[format_name]


def eval_pred(predict_parser, gold_list, pred_list, text_list=None, raw_list=None):
    well_formed_list, counter = predict_parser.decode(
        gold_list, pred_list, text_list, raw_list)

    relation_metric = Metric()

    for instance in well_formed_list:
        relation_metric.count_instance(instance['gold_relation'],
                                   instance['pred_relation'],
                                   verbose=False)

    role_result = relation_metric.compute_f1(prefix='relation-')

    result = dict()
    result.update(role_result)
    result.update(counter)
    return result

def eval_pred_with_decoding(gold_list, pred_list, text_list=None, raw_list=None):

    relation_metric = Metric()

    relation_metric.count_instance(gold_list, pred_list,verbose= False)

    role_result = relation_metric.compute_f1(prefix='relation-')

    result = dict()
    result.update(role_result)
    return result

def get_extract_metrics(pred_lns: List[str], tgt_lns: List[str], label_constraint: EventSchema, decoding_format='tree'):
    # predict_parser is the TreePredictParser, because decoding_format is tree
    predict_parser = get_predict_parser(format_name=decoding_format)(label_constraint=label_constraint)
    return eval_pred_with_decoding(
        gold_list=tgt_lns,
        pred_list=pred_lns
    )
