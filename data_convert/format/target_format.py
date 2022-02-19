#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Dict


class TargetFormat:
    @staticmethod
    def annotate_predicate_entities(tokens: List[str], entities:List[Dict] , zh=False): pass

    @staticmethod
    def annotate_predicate_arguments(
        tokens: List[str], predicate_arguments: List[Dict], entities:List[Dict] , zh=False): pass
