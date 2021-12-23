#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

from data_convert.task_format.task_format import TaskFormat


class Conll04(TaskFormat):
    def __init__(self, doc_json):
        self.Array = doc_json

    def generate_relations(self):
        for array in self.Array:
            sentence = array['tokens']
            relations = array['relations']
            entities = array['entities']
            relations_result = list()
            for relation in relations:
                first_entity = [entities[relation['head']]['start'], entities[relation['head']]['end']]
                second_entity = [entities[relation['tail']]['start'], entities[relation['tail']]['end']]
                entity_type1 = entities[relation['head']]['type']
                entity_type2 = entities[relation['tail']]['type']
                arguments = [list(range(first_entity[0], first_entity[1])), entity_type1, list(
                    range(second_entity[0], second_entity[1])), entity_type2]
                relation_type = relation['type']
                flag = False
                for old_relation in relations_result:
                    if relation_type == old_relation['type']:
                        old_relation['arguments'].append(arguments)
                        flag = True
                        break

                if not flag:
                    relations_result += [{'type': relation_type,
                                   'arguments': [arguments]}]
            
            mentions = list()
            for entity in entities:
                mention = [
                    list(range(entity['start'], entity['end'])), entity['type']]
                mentions += [mention]

            yield {'tokens': sentence, 'relations': relations_result, 'entities': mentions}



class DyIEPP(TaskFormat):
    def __init__(self, doc_json):
        self.sentences = doc_json['sentences']
        self.ner = doc_json['ner']
        self.relations = doc_json['relations']
        if 'sentence_start' in doc_json:
            self.sentence_start = doc_json.get('sentence_start', doc_json['_sentence_start'])
        else:
            start = 0
            self.sentence_start = []
            for one_sentence in doc_json['sentences']:
                self.sentence_start.append(start)
                start = start + len(one_sentence)
    def generate_relations(self):
        for relations_in_sentence, sentence, entities, start_pos in zip(self.relations, self.sentences, self.ner, self.sentence_start):
            relations = list()
            for relation in relations_in_sentence:
                # 'arguments': [['Arg-1', [9]], ['Arg-2', [14]]]
                first_entity = [relation[0], relation[1]]
                second_entity = [relation[2], relation[3]]
                entity_type1 = ""
                entity_type2 = ""
                for entity in entities:
                    if(entity[0] == first_entity[0] and entity[1] == first_entity[1]):
                        entity_type1 = entity[2]
                    if(entity[0] == second_entity[0] and entity[1] == second_entity[1]):
                        entity_type2 = entity[2]
                
                arguments = [list(range(first_entity[0]-start_pos, first_entity[1]+1-start_pos)), entity_type1, list(
                    range(second_entity[0]-start_pos, second_entity[1]+1-start_pos)), entity_type2]
                relation_type = relation[4]
                flag = False
                for old_relation in relations:
                    if relation_type == old_relation['type']:
                        old_relation['arguments'].append(arguments)
                        flag = True
                        break
                
                if not flag:
                    relations += [{'type': relation_type, 'arguments': [arguments]}]
        
            mentions = list()
            for entity in entities:
                mention = [list(range(entity[0]-start_pos, entity[1]+1-start_pos)), entity[2]]
                mentions += [mention]
            yield {'tokens': sentence, 'relations': relations, 'entities': mentions}

class Event(TaskFormat):
    """
    {
        "doc_id": "NYT_ENG_20130914.0094",
        "sent_id": "NYT_ENG_20130914.0094-1",
        "tokens": ["LARGO", "\u2014", "A", "judge", "on", "Friday", "refused", "to", "stop", "''", "Hiccup", "Girl", "\u2019'", "Jennifer", "Mee", "from", "giving", "interviews", "to", "the", "media", "in", "the", "final", "days", "before", "her", "murder", "trial", "."],
        "entities": [
            {"entity_id": "NYT_ENG_20130914.0094-1-8-42", "entity_type": "PER", "mention_type": "NOM", "start": 3, "end": 4, "text": "judge"},
            {"entity_id": "NYT_ENG_20130914.0094-1-39-2096", "entity_type": "PER", "mention_type": "NAM", "start": 10, "end": 12, "text": "Hiccup Girl"},
            {"entity_id": "NYT_ENG_20130914.0094-1-39-48", "entity_type": "PER", "mention_type": "NAM", "start": 13, "end": 15, "text": "Jennifer Mee"},
            {"entity_id": "NYT_ENG_20130914.0094-1-39-54", "entity_type": "PER", "mention_type": "PRO", "start": 26, "end": 27, "text": "her"}
        ],
        "relations": [],
        "events": [
            {
                "event_id": "NYT_ENG_20130914.0094-1-25-998", "event_type": "contact", "event_subtype": "broadcast",
                "trigger": {"text": "interviews", "start": 17, "end": 18},
                "arguments": [
                    {"entity_id": "NYT_ENG_20130914.0094-1-39-48", "role": "entity", "text": "Jennifer Mee"}
                ]
            },
            {
                "event_id": "NYT_ENG_20130914.0094-1-1040-1019", "event_type": "justice", "event_subtype": "trialhearing",
                "trigger": {"text": "trial", "start": 28, "end": 29},
                 "arguments": [{"entity_id": "NYT_ENG_20130914.0094-1-39-54", "role": "defendant", "text": "her"}
                 ]
            }
        ],
        "start": 231, "end": 380,
         "text": "LARGO \u2014 A judge on Friday refused to stop ''Hiccup Girl\u2019' Jennifer Mee from giving interviews to the media in the final days before her murder trial."
    }
    """

    def __init__(self, doc_json):
        self.doc_key = doc_json['doc_id']
        self.sentence = doc_json['tokens']
        self.entities = {
            entity['id']: entity for entity in doc_json['entity_mentions']}
        self.relations = doc_json['relation_mentions']
        self.events = doc_json['event_mentions']
        # self.sentence_start = doc_json['start']
        # self.sentence_end = doc_json['end']
        # self.text = doc_json['text']

    def generate_relations(self, type_format='subtype'):
        relations = list()
        for relation in self.relations:
            arguments = list()
            relation_type = relation['relation_type']
            # there are two arguments in one relation
            for argument in relation['arguments']:
                argument_entity = self.entities[argument['entity_id']]
                arguments += [list(range(argument_entity['start'], argument_entity['end'])), argument_entity['entity_type']]
            flag = False
            for old_relation in relations:
                if relation_type == old_relation['type']:
                    old_relation['arguments'].append(arguments)
                    flag = True
                    break
            if not flag:
                relations += [{'type': relation_type, 'arguments': [arguments]}]
        
        mentions = list()
        for entity in self.entities:
            mention = [list(range(self.entities[entity]['start'], self.entities[entity]['end'])), self.entities[entity]['entity_type']]
            mentions += [mention]

        yield {'tokens': self.sentence, 'relations': relations, 'entities': mentions}


def ace2005_en_file_tuple(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    conll_2012_folder = "data/raw_data/ace05-EN/"

    file_tuple = [
        (conll_2012_folder + "/train.oneie.json", output_folder + '/train'),
        (conll_2012_folder + "/dev.oneie.json", output_folder + '/val'),
        (conll_2012_folder + "/test.oneie.json", output_folder + '/test'),
    ]

    return file_tuple


def sci_file_tuple(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    conll_2012_folder = "data/raw_data/sci_relation"

    file_tuple = [
        (conll_2012_folder + "/train.json", output_folder + '/train'),
        (conll_2012_folder + "/dev.json", output_folder + '/val'),
        (conll_2012_folder + "/test.json", output_folder + '/test'),
    ]

    return file_tuple


def conll04_file_tuple(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    conll_04_folder = "data/datasets/conll04"

    file_tuple = [
        (conll_04_folder + "/conll04_train.json", output_folder + '/train'),
        (conll_04_folder + "/conll04_dev.json", output_folder + '/val'),
        (conll_04_folder + "/conll04_test.json", output_folder + '/test'),
    ]

    return file_tuple

if __name__ == "__main__":
    pass
