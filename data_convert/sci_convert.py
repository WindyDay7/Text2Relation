#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import json
from data_convert.utils import read_file, check_output, data_counter_to_table, get_schema, output_schema

output_folder = 'data/new_text2tree/sci_relation'
conll_2012_folder = "data/raw_data/sci_relation"
span_output_folder = output_folder + '_span'
file_tuple = [
    (conll_2012_folder + "/train.json", output_folder + '/train'),
    (conll_2012_folder + "/dev.json", output_folder + '/val'),
    (conll_2012_folder + "/test.json", output_folder + '/test'),
]


relation_schema_set = set()
for in_filename, output_filename in file_tuple:
    if not os.path.exists(output_filename):
        os.makedirs(output_filename)
    span_output_filename = output_filename.replace(
        output_folder, span_output_folder)

    if not os.path.exists(span_output_folder):
        os.makedirs(span_output_folder)

    relation_output = open(output_filename + '.json', 'w')
    span_relation_output = open(span_output_filename + '.json', 'w')

    for line in read_file(in_filename):
        sentence = json.loads(line.strip())
        one_sentences = ' '.join(sentence["sentences"])
        
        print(sentence["sentences"])
        break
