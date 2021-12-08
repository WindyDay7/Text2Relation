#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

from data_convert.task_format.task_format import TaskFormat


class DyIEPP(TaskFormat):
    def __init__(self, doc_json):
        self.doc_key = doc_json['doc_key']
        self.sentences = doc_json['sentences']
        self.ner = doc_json['ner']
        self.relations = doc_json['relations']
        start = 0
        self.sentence_start = []
        for one_sentence in doc_json['sentences']:
            self.sentence_start.append(start)
            start = start + len(one_sentence)

    def generate_relations(self):
        for relations_in_sentence, sentence_start, sentence in zip(self.relations, self.sentence_start, self.sentences):
            relations = list()
            for relation in relations_in_sentence:
                # 'arguments': [['Arg-1', [9]], ['Arg-2', [14]]]
                arguments = [list(range(relation[0]-sentence_start, relation[1]+1-sentence_start)),
                              list(range(relation[2]-sentence_start, relation[3]+1-sentence_start))]
                relation_type = relation[4].split('.')[0]
                flag = False
                for old_relation in relations:
                    if relation_type == old_relation['type']:
                        old_relation['arguments'].append(arguments)
                        flag = True
                        break
                
                if not flag:
                    relations += [{'type': relation_type, 'arguments': [arguments]}]
                
        
            yield {'tokens': sentence, 'relations': relations}

    def generate_sentence(self, type_format='subtype'):
        for _ner, sentence, events_in_sentence, sentence_start in zip(self.ner, self.sentences, self.events, self.sentence_start):
            events = list()
            for event in events_in_sentence:
                trigger, event_type = event[0]
                trigger -= sentence_start

                suptype, subtype = event_type.split('.')

                if type_format == 'subtype':
                    event_type = subtype
                elif type_format == 'suptype':
                    event_type = suptype
                else:
                    event_type = suptype + type_format + subtype

                arguments = list()
                for start, end, role in event[1:]:
                    start -= sentence_start
                    end -= sentence_start
                    arguments += [[role, list(range(start, end + 1))]]

                # add the name_entity_reconganition
                for argument in arguments:
                    for ner_pos in _ner:
                        if((ner_pos[0]-sentence_start) == argument[1][0]):
                            argument.insert(1, ner_pos[2])
                    if(len(argument) != 3):
                        print("Wrong, this argument is not in the ner")

                event = {'type': event_type, 'tokens': [
                    trigger], 'arguments': arguments}

                events += [event]
            yield {'tokens': sentence, 'events': events}


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
            

        yield {'tokens': self.sentence, 'relations': relations}

    def generate_sentence(self, type_format='subtype'):
        events = list()

        for event in self.events:
            arguments = list()
            for argument in event['arguments']:
                argument_entity = self.entities[argument['entity_id']]
                arguments += [[argument['role'], argument_entity['entity_type'],
                               list(range(argument_entity['start'], argument_entity['end']))]]

            suptype, subtype = event['event_type'].split(':')

            if type_format == 'subtype':
                event_type = subtype
            elif type_format == 'suptype':
                event_type = suptype
            else:
                event_type = suptype + type_format + subtype

            events += [{
                'type': event_type,
                'tokens': list(range(event['trigger']['start'], event['trigger']['end'])),
                'arguments': arguments
            }]

        yield {'tokens': self.sentence, 'events': events}


def DyIEPP_ace2005_file_tuple(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    conll_2012_folder = "data/raw_data/dyiepp_ace2005"

    file_tuple = [
        (conll_2012_folder + "/train.json", output_folder + '/train'),
        (conll_2012_folder + "/dev.json", output_folder + '/val'),
        (conll_2012_folder + "/test.json", output_folder + '/test'),
    ]

    return file_tuple


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


if __name__ == "__main__":
    pass
