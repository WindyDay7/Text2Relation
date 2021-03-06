#!/usr/bin/env python
# -*- coding:utf-8 -*-
from copy import deepcopy
from typing import List, Counter, Tuple


class PredictParser:
    def __init__(self, label_constraint):
        self.relation_set = label_constraint.type_list

    def decode(self, gold_list, pred_list, text_list=None, raw_list=None) -> Tuple[List, Counter]:
        """

        :param gold_list:
        :param pred_list:
        :param text_list:
        :param raw_list:
        :return:
            dict:
                pred_event -> [(type1, trigger1), (type2, trigger2), ...]
                gold_event -> [(type1, trigger1), (type2, trigger2), ...]
                pred_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
                gold_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
            Counter:
        """
        pass

    @staticmethod
    def count_multi_event_role_in_instance(instance, counter):
        if len(instance['gold_relation']) != len(set(instance['gold_relation'])):
            counter.update(['multi-same-role-gold'])

        if len(instance['pred_relation']) != len(set(instance['pred_relation'])):
            counter.update(['multi-same-role-pred'])


class Metric:
    def __init__(self):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {prefix + 'tp': tp,
                prefix + 'gold': gold_num,
                prefix + 'pred': pred_num,
                prefix + 'P': p * 100,
                prefix + 'R': r * 100,
                prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100
                }

    def count_instance(self, gold_list, pred_list, verbose=False):
        if verbose:
            print("Gold:", gold_list)
            print("Pred:", pred_list)
        for i in range(0, len(gold_list)):
            gold_list[i] = gold_list[i].replace(" ","")
            pred_list[i] = pred_list[i].replace(" ","")
        
        for i in range(0, len(gold_list)):
            gold_relations = gold_list[i].split("<Relation_S>")
            gold_labels = list()
            for gold_relation in gold_relations:
                one_relations = gold_relation.split("<End>")
                if len(one_relations) == 1:
                    continue
                relation_type = one_relations[0]
                one_relations[-1] = one_relations[-1].replace("<Relation_E>","")
                one_relations[-1] = one_relations[-1].replace("<Temp_E>","")
                for j in range(1, len(one_relations)):
                    gold_labels.append(one_relations[0]+"_"+one_relations[j])
        #         print("gold_labels",gold_labels)
                
            pred_relations = pred_list[i].split("<Relation_S>")
            pred_labels = list()
            for pred_relation in pred_relations:
                one_relations = pred_relation.split("<End>")
                if len(one_relations) == 1:
                    continue
                relation_type = one_relations[0]
                one_relations[-1] = one_relations[-1].replace("<Relation_E>","")
                one_relations[-1] = one_relations[-1].replace("<Temp_E>","")
                for k in range(1, len(one_relations)):
                    pred_labels.append(one_relations[0]+"_"+one_relations[k])
        #         print("pred_labels",pred_labels)
            
            self.gold_num += len(gold_labels)
            self.pred_num += len(pred_labels)
            # print("Gold Labels are: ", gold_labels)
            for pred in pred_labels:
                if pred in gold_labels:
                    self.tp += 1
