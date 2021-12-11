#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Dict

from data_convert.format.text2tree import Relation_start, Relation_end, Entity_Type, Entity_End
from extraction.label_tree import get_label_name_tree

import os

debug = True if 'DEBUG' in os.environ else False
debug_step = True if 'DEBUG_STEP' in os.environ else False


def match_sublist(the_list, to_match):
    """
    :param the_list: [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
    :param to_match: [1, 2]
    :return:
        [(0, 1), (6, 7)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list


def find_bracket_position(generated_text, _type_start, _type_end):
    """
    Find the bracket position in generated text, return a dictionary,
    bracket and their corresponding position list
    """
    bracket_position = {_type_start: list(), _type_end: list()}
    for index, char in enumerate(generated_text):
        if char in bracket_position:
            bracket_position[char] += [index]
    return bracket_position




def generated_search_src_sequence(generated, src_sequence, end_sequence_search_tokens=None):
    # which tokens cloud be generated in next step
    print(generated, src_sequence) if debug else None

    if len(generated) == 0:
        # It has not been generated yet. All SRC are valid.
        return src_sequence

    matched_tuples = match_sublist(the_list=src_sequence, to_match=generated)

    valid_token = list()
    # generated text have appear in the scource sentence
    # generate the next token in the source sentence
    for _, end in matched_tuples:
        next_index = end + 1
        if next_index < len(src_sequence):
            valid_token += [src_sequence[next_index]]

    # next token might be the end token
    if end_sequence_search_tokens:
        # print(end_sequence_search_tokens)
        # print(valid_token)
        valid_token += end_sequence_search_tokens

    return valid_token


def get_constraint_decoder(tokenizer, type_schema, decoding_schema, source_prefix=None):
    if decoding_schema == 'tree':
        return TreeConstraintDecoder(tokenizer=tokenizer, type_schema=type_schema, source_prefix=source_prefix)
    elif decoding_schema == 'treespan':
        return SpanConstraintDecoder(tokenizer=tokenizer, type_schema=type_schema, source_prefix=source_prefix)
    else:
        raise NotImplementedError(
            'Type Schema %s, Decoding Schema %s do not map to constraint decoder.' % (
                decoding_schema, decoding_schema)
        )


class ConstraintDecoder:
    """
    source_prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    """
    def __init__(self, tokenizer, source_prefix):
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix
        self.source_prefix_tokenized = tokenizer.encode(source_prefix,
                                                        add_special_tokens=False) if source_prefix else []

    def get_state_valid_tokens(self, src_sentence: List[str], tgt_generated: List[str]) -> List[str]:
        pass

    def constraint_decoding(self, src_sentence, tgt_generated):
        if self.source_prefix_tokenized:
            # Remove Source Prefix for Generation
            src_sentence = src_sentence[len(self.source_prefix_tokenized):]

        if debug:
            print("Src:", self.tokenizer.convert_ids_to_tokens(src_sentence))
            print("Tgt:", self.tokenizer.convert_ids_to_tokens(tgt_generated))

        valid_token_ids = self.get_state_valid_tokens(
            src_sentence.tolist(),
            tgt_generated.tolist()
        )

        if debug:
            print('========================================')
            print('valid tokens:', self.tokenizer.convert_ids_to_tokens(
                valid_token_ids), valid_token_ids)
            if debug_step:
                input()

        # return self.tokenizer.convert_tokens_to_ids(valid_tokens)
        return valid_token_ids


class TreeConstraintDecoder(ConstraintDecoder):
    """
    rewrite constraint_decoding method 
    """
    def __init__(self, tokenizer, type_schema, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.Prediction_end = '<Prediction-end>'
        self.relation_tree = get_label_name_tree(
            type_schema.type_list, self.tokenizer, end_symbol=self.tree_end)
        self.relation_start = self.tokenizer.convert_tokens_to_ids([Relation_start])[0]
        self.relation_end = self.tokenizer.convert_tokens_to_ids([Relation_end])[0]
        self.entity_token = dict()
        for entity in Entity_Type:
            self.entity_token[entity] = self.tokenizer.convert_tokens_to_ids([Entity_Type[entity]])[0]
        self.entity_end = self.tokenizer.convert_tokens_to_ids([Entity_End])[0]
    def check_state(self, tgt_generated):
        """
        return the generated tree state and the last special token's index, 
        """
        if tgt_generated[-1] == self.tokenizer.pad_token_id:
            return 'start', -1

        special_token_set = {self.relation_start, self.relation_end}
        # the indexes of special tokens in the generated sentences
        # and the kind of special token
        special_index_token = list(
            filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))

        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        entity_token_set = set()
        for entity in self.entity_token:
            entity_token_set.add(self.entity_token[entity])
        special_entity_index_token = list(
            filter(lambda x: x[1] in entity_token_set, list(enumerate(tgt_generated))))
        special_end_index_token = list(
            filter(lambda x: x[1] == self.entity_end, list(enumerate(tgt_generated))))
        
        bracket_position = find_bracket_position(
            tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        start_number, end_number = len(bracket_position[self.type_start]), len(
            bracket_position[self.type_end])

        entity_number = len(special_entity_index_token)
        entity_end = len(special_end_index_token)

        # New kind of relation or the end of the prediction 
        if start_number == end_number:
            return 'new_or_end', -1
        # generate the first relation
        if start_number == end_number + 1:
            # relation type and start of the first entity pair,
            if(entity_number == (2*entity_end)):
                state = 'start_entity'
                if entity_number != 0:
                    last_special_index, last_special_token = special_end_index_token[-1]
            # spredicit the first entity text span or second Entity type special token
            elif (entity_number == (entity_end*2 + 1)):
                state = 'entity1_spans'
                last_special_index, last_special_token = special_entity_index_token[-1]
            # predict entity mentions(text span) or Entity_end special token
            elif (entity_number == (entity_end*2 + 2)):
                state = 'entity2_spans'
                last_special_index, last_special_token = special_entity_index_token[-1]
            else:
                state = 'error'
        else:
            state = 'error'
        return state, last_special_index

    def search_prefix_tree_and_sequence(self, generated: List[str], prefix_tree: Dict):
        """
        This part is for relation text, for example, "PART-WHOLE"
        :param generated: from the type token to the end, for example, because, after tokenizer
        it's length might longer than 2
        :param prefix_tree: relation tree
        :param src_sentence: source sentence token ids
        :param end_sequence_search_tokens: start or end token 
        :return:
        """
        tree = prefix_tree
        # generated is from "<" to the end, the first is the event type
        for index, token in enumerate(generated):
            # then tree is event type token id
            tree = tree[token]
            # sometimes the end of event type is 1, this means token == 1
            # wheather the event type text is end, means event type text have been generated
            is_tree_end = len(tree) == 1 and self.tree_end in tree
            if is_tree_end:
                valid_token = list(self.entity_token.values())
                return valid_token
            # if end trigger token appears in the generated text 
            if self.tree_end in tree:
                try:
                    valid_token = [self.type_start, self.type_end]
                    return valid_token
                except IndexError:
                    # Still search tree
                    continue
        
        valid_token = list(tree.keys())
        return valid_token

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """
        :param src_sentence is List of ids, there are token ids
        :param tgt_generated is also List of ids
        :return:
            List[str], valid token list
        """
        if self.tokenizer.eos_token_id in src_sentence:
            src_sentence = src_sentence[:src_sentence.index(self.tokenizer.eos_token_id)]

        state, index = self.check_state(tgt_generated)

        print("State: %s" % state) if debug else None

        if state == 'error':
            print("Error:")
            print("Src:", src_sentence)
            print("Tgt:", tgt_generated)
            valid_tokens = [self.tokenizer.eos_token_id]

        elif state == 'new_or_end':
            valid_tokens = [self.relation_start, self.tokenizer.eos_token_id]

        elif state == 'start_entity':
            if(tgt_generated[-1] == self.relation_start):
                valid_tokens = list(self.relation_tree.keys())
            elif (tgt_generated[-1] == self.entity_end):
                valid_tokens = list(self.entity_token.values()) + [self.relation_end]
            else:
                valid_tokens = self.search_prefix_tree_and_sequence(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.relation_tree,
                )

        elif state == 'entity1_spans':
            # this part might be the first entity text span, or next entity start token
            if tgt_generated[-1] in list(self.entity_token.values()):
                # in text span
                valid_tokens = self.search_prefix_tree_and_sequence(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.relation_tree,
                )
            else:
                # in the entity text span or next start entity token
                valid_tokens = self.search_prefix_tree_and_sequence(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.relation_tree,
                )
                valid_tokens += list(self.entity_token.values())

        elif state == 'entity2_spans':
            if tgt_generated[-1] in list(self.entity_token.values()):
                # in text span
                valid_tokens = self.search_prefix_tree_and_sequence(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.relation_tree,
                )
            else:
                # end of the two entities
                valid_tokens = self.search_prefix_tree_and_sequence(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.relation_tree,
                )
                valid_tokens += [self.entity_end]
        else:
            raise NotImplementedError(
                'State `%s` for %s is not implemented.' % (state, self.__class__))

        print("Valid: %s" % self.tokenizer.convert_ids_to_tokens(
            valid_tokens)) if debug else None
        return valid_tokens


class SpanConstraintDecoder(ConstraintDecoder):
    def __init__(self, tokenizer, type_schema, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tree_end = '<tree-end>'
        self.type_tree = get_label_name_tree(type_schema.type_list + type_schema.role_list,
                                             tokenizer=self.tokenizer,
                                             end_symbol=self.tree_end)
        self.type_start = self.tokenizer.convert_tokens_to_ids([type_start])[0]
        self.type_end = self.tokenizer.convert_tokens_to_ids([type_end])[0]

    def check_state(self, tgt_generated):
        if tgt_generated[-1] == self.tokenizer.pad_token_id:
            return 'start', -1

        special_token_set = {self.type_start, self.type_end}
        special_index_token = list(
            filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))

        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        bracket_position = find_bracket_position(
            tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        start_number, end_number = len(bracket_position[self.type_start]), len(
            bracket_position[self.type_end])

        if start_number == end_number:
            return 'end_generate', -1
        if start_number == end_number + 1:
            state = 'start_first_generation'
        elif start_number == end_number + 2:
            state = 'generate_span'
        else:
            state = 'error'
        return state, last_special_index

    def search_prefix_tree_and_sequence(self, generated: List[str], prefix_tree: Dict, src_sentence: List[str],
                                        end_sequence_search_tokens: List[str] = None):
        """
        Generate Type Name + Text Span
        :param generated:
        :param prefix_tree:
        :param src_sentence:
        :param end_sequence_search_tokens:
        :return:
        """
        tree = prefix_tree
        for index, token in enumerate(generated):
            tree = tree[token]
            is_tree_end = len(tree) == 1 and self.tree_end in tree

            if is_tree_end:
                valid_token = generated_search_src_sequence(
                    generated=generated[index + 1:],
                    src_sequence=src_sentence,
                    end_sequence_search_tokens=end_sequence_search_tokens,
                )
                return valid_token

            if self.tree_end in tree:
                try:
                    valid_token = generated_search_src_sequence(
                        generated=generated[index + 1:],
                        src_sequence=src_sentence,
                        end_sequence_search_tokens=end_sequence_search_tokens,
                    )
                    return valid_token
                except IndexError:
                    # Still search tree
                    continue

        valid_token = list(tree.keys())
        return valid_token

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """
        :param src_sentence:
        :param tgt_generated:
        :return:
            List[str], valid token list
        """
        if self.tokenizer.eos_token_id in src_sentence:
            src_sentence = src_sentence[:src_sentence.index(
                self.tokenizer.eos_token_id)]

        state, index = self.check_state(tgt_generated)

        print("State: %s" % state) if debug else None

        if state == 'error':
            print("Error:")
            print("Src:", src_sentence)
            print("Tgt:", tgt_generated)
            valid_tokens = [self.tokenizer.eos_token_id]

        elif state == 'start':
            valid_tokens = [self.type_start]

        elif state == 'start_first_generation':
            valid_tokens = [self.type_start, self.type_end]

        elif state == 'generate_span':

            if tgt_generated[-1] == self.type_start:
                # Start Event Label
                return list(self.type_tree.keys())

            elif tgt_generated[-1] == self.type_end:
                raise RuntimeError('Invalid %s in %s' %
                                   (self.type_end, tgt_generated))

            else:
                valid_tokens = self.search_prefix_tree_and_sequence(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.type_tree,
                    src_sentence=src_sentence,
                    end_sequence_search_tokens=[self.type_end]
                )

        elif state == 'end_generate':
            valid_tokens = [self.tokenizer.eos_token_id]

        else:
            raise NotImplementedError(
                'State `%s` for %s is not implemented.' % (state, self.__class__))

        print("Valid: %s" % valid_tokens) if debug else None
        return valid_tokens
