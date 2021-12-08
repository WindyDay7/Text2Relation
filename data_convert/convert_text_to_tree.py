#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import json
from collections import Counter, defaultdict
from data_convert.format.text2tree import Text2Tree
from data_convert.task_format.event_extraction import Event
from data_convert.utils import read_file, check_output, data_counter_to_table, get_schema, output_schema
from nltk.corpus import stopwords

english_stopwords = set(stopwords.words('english') + ["'s", "'re", "%"])


def convert_file_tuple(file_tuple, data_class=Event, target_class=Text2Tree,
                       output_folder='data/text2tree/framenet',
                       ignore_nonevent=False, zh=False,
                       mark_tree=False, type_format='subtype'):
    counter = defaultdict(Counter)
    data_counter = defaultdict(Counter)

    relation_schema_set = set()

    span_output_folder = output_folder + '_span'

    if not os.path.exists(span_output_folder):
        os.makedirs(span_output_folder)

    # in_filename a example is "data/raw_data/dyiepp_ace2005/train.json"
    # out_filename a example is 'data/text2tree/ace2005_event/train'
    for in_filename, output_filename in file_tuple(output_folder):
        span_output_filename = output_filename.replace(
            output_folder, span_output_folder)

        relation_output = open(output_filename + '.json', 'w')
        span_relation_output = open(span_output_filename + '.json', 'w')

        for line in read_file(in_filename):
            document = data_class(json.loads(line.strip()))
            for sentence in document.generate_relations():
                if ignore_nonevent and len(sentence['relations']) == 0:
                    continue
                # souce is the sentence text tokens
                # target is the corresponding relations annotations
                source, target = target_class.annotate_predicate_arguments(
                    tokens=sentence['tokens'],
                    predicate_arguments=sentence['relations'],
                    zh=zh
                )
                # The event knowledge schema, used in constrained decoder
                # sentence['tokens'] is the sentence schema information, event['tokens']
                # is the event trigger text span index
                for relation in sentence['relations']:
                    relation_schema_set.add(relation['type'])
                    sep = '' if zh else ' '
                    counter['type'].update([relation['type']])
                    data_counter[in_filename].update(['relation'])
                    for argument in relation['arguments']:
                        data_counter[in_filename].update(['argument'])

                data_counter[in_filename].update(['sentence'])

                relation_output.write(json.dumps(
                    {'text': source, 'relation': target}, ensure_ascii=False) + '\n')

                # span_source, span_target = target_class.annotate_span(
                #     tokens=sentence['tokens'],
                #     predicate_arguments=sentence['relations'],
                #     zh=zh,
                #     mark_tree=mark_tree
                # )

                # span_relation_output.write(
                #     json.dumps({'text': span_source, 'relation': span_target}, ensure_ascii=False) + '\n')

        relation_output.close()
        span_relation_output.close()

        check_output(output_filename)
        check_output(span_output_filename)
        print('\n')
    relation_type_list = list(set([schema for schema in relation_schema_set]))
    schema_output_file=os.path.join(output_folder, 'relation.schema')
    with open(schema_output_file, 'w') as output:
        output.write(json.dumps(relation_type_list) + '\n')
    # output_schema(event_schema_set, output_file=os.path.join(
    #     span_output_folder, 'event.schema'))
    print('Pred:', len(counter['pred']), counter['pred'].most_common(10))
    print('Type:', len(counter['type']), counter['type'].most_common(10))
    print('Role:', len(counter['role']), counter['role'].most_common(10))
    print(data_counter_to_table(data_counter))
    print('\n\n\n')


def convert_ace2005_event(output_folder='data/new_text2tree/ace2005_event', type_format='subtype',
                          ignore_nonevent=False, mark_tree=False):
    from data_convert.task_format.event_extraction import ace2005_en_file_tuple
    convert_file_tuple(file_tuple=ace2005_en_file_tuple,
                       output_folder=output_folder,
                       ignore_nonevent=ignore_nonevent,
                       mark_tree=mark_tree,
                       type_format=type_format,
                       )


def convert_dyiepp_event(output_folder='data/new_text2tree/ace2005_event', type_format='subtype',
                         ignore_nonevent=False, mark_tree=False):
    from data_convert.task_format.event_extraction import DyIEPP_ace2005_file_tuple
    convert_file_tuple(file_tuple=DyIEPP_ace2005_file_tuple,
                       output_folder=output_folder,
                       ignore_nonevent=ignore_nonevent,
                       mark_tree=mark_tree,
                       type_format=type_format,
                       data_class=DyIEPP
                       )


def convert_sci_event(output_folder='data/new_text2tree/sci_relastion_', type_format='subtype',
                      ignore_nonevent=False, mark_tree=False):
    from data_convert.task_format.event_extraction import sci_file_tuple
    convert_file_tuple(file_tuple=sci_file_tuple,
                       output_folder=output_folder,
                       ignore_nonevent=ignore_nonevent,
                       mark_tree=mark_tree,
                       type_format=type_format,
                       data_class=DyIEPP
                       )


if __name__ == "__main__":
    type_format_name = 'subtype'
    convert_ace2005_event("data/new_text2tree/one_ie_ace2005_%s" % type_format_name,
                          type_format=type_format_name,
                          ignore_nonevent=False,
                          mark_tree=False
                          )
    """
    convert_sci_event("data/new_text2tree/sci_relation_%s" % type_format_name,
                      type_format=type_format_name,
                      ignore_nonevent=False,
                      mark_tree=False)
    """
    
