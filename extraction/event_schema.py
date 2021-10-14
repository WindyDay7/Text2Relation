#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
from collections import defaultdict
from typing import List


class EventSchema:
    def __init__(self, type_list):
        self.type_list = type_list

    @staticmethod
    def read_from_file(filename):
        lines = open(filename).readlines()
        type_list = json.loads(lines[0])
        return EventSchema(type_list)

    def write_to_file(self, filename):
        with open(filename, 'w') as output:
            output.write(json.dumps(self.type_list) + '\n')

