#!/usr/bin/env python
# -*- coding:utf-8 -*-
from data_convert.format.target_format import TargetFormat


def get_str_from_tokens(tokens, sentence, separator=' '):
    start, end_exclude = tokens[0], tokens[-1] + 1
    return separator.join(sentence[start:end_exclude])


type_start = '<extra_id_0>'
type_end = '<extra_id_1>'
role_end = '<extra_id_3>'


class Text2Tree(TargetFormat):

    @staticmethod
    def annotate_predicate_arguments(tokens, predicate_arguments, mark_tree=False, multi_tree=False, zh=False):
        """

        :param tokens:
            US President George W. Bush told Canadian Prime Minister Jean Chretien by telephone Monday that he looked forward
            to seeing him at the upcoming summit of major industrialized nations and Russia , the White House said Tuesday .
        :param predicate_arguments:

        :param mark_tree False
            (
                (Meet summit (Entity Russia) (Entity major industrialized nations))
                (Phone-Write told (Entity US President George W. Bush) (Entity Canadian Prime Minister Jean Chretien) (Time-Within Monday))
            )
       :param  multi_tree  True
            (Meet summit [Entity Russia] [Entity major industrialized nations])
            (Phone-Write told [Entity US President George W. Bush] [Entity Canadian Prime Minister Jean Chretien] [Time-Within Monday])

        :return:
        """
        token_separator = '' if zh else ' '
        relation_list = ["COMPARE", "PART-OF", "HYPONYM-OF"]
        relation_str_rep_list = list()
        # if(len(predicate_arguments)>2):
        #     print(predicate_arguments)
        
        for define_relation_type in relation_list:
            flag = False
            for predicate_argument in predicate_arguments:
                relation_type = predicate_argument['type']

                # predicate_argument['tokens'] is the trigger index
                # tokens is the sentence tokens, we get the trigger text span here
                # predicate_text = get_str_from_tokens(predicate_argument['tokens'], tokens, separator=token_separator)

                # prefix_tokens[predicate_argument['tokens'][0]] = ['[ ']
                # suffix_tokens[predicate_argument['tokens'][-1]] = [' ]']

                # print(predicate_argument)
                # role_name is the argument role, role_tokens are corresponding text span index
                role_str_list = list()
                if relation_type in define_relation_type:
                    flag = True
                    for relation_pair in predicate_argument['arguments']:
                        # get the role text span from role tokens index
                        # print(role_tokens)
                        
                        first_entity = get_str_from_tokens(relation_pair[0], tokens, separator=token_separator)
                        second_entity = get_str_from_tokens(relation_pair[1], tokens, separator=token_separator)
                        
                        one_role_str = ' '.join([type_start, first_entity, type_start, second_entity, type_end, type_end])
                        role_str_list += [one_role_str]

                    role_str_list_str = ' '.join(role_str_list)
                    relation_str_rep = f"{type_start} {relation_type} {role_str_list_str} {relation_type} {type_end}"
                    relation_str_rep_list += [relation_str_rep]
                    break
            if not flag:
                relation_type
            source_text = token_separator.join(tokens)
            target_text = ' '.join(relation_str_rep_list)

        if not multi_tree:
            target_text = f'{type_start} ' + \
                          ' '.join(relation_str_rep_list) + f' {type_end}'

        return source_text, target_text

    @staticmethod
    def annotate_span(tokens, predicate_arguments, mark_tree=False, zh=False):
        """
        :param tokens:
            US President George W. Bush told Canadian Prime Minister Jean Chretien by telephone Monday that he looked forward
            to seeing him at the upcoming summit of major industrialized nations and Russia , the White House said Tuesday .
        :param predicate_arguments:

        :return:
        mark_tree False
            (
                (Meet summit (Entity Russia) (Entity major industrialized nations))
                (Phone-Write told (Entity US President George W. Bush) (Entity Canadian Prime Minister Jean Chretien) (Time-Within Monday))
            )
        mark_tree  True
            (
                (Meet summit [Entity Russia] [Entity major industrialized nations])
                (Phone-Write told [Entity US President George W. Bush] [Entity Canadian Prime Minister Jean Chretien] [Time-Within Monday])
            )
        """

        token_separator = '' if zh else ' '

        relation_str_rep_list = list()

        for predicate_argument in predicate_arguments:
            relation_type = predicate_argument['type']

            span_str_list = [' '.join([type_start, relation_type, type_end])]

            for relation_pair in predicate_argument['arguments']:
                # get the role text span from role tokens index
                # print(role_tokens)

                first_entity = get_str_from_tokens(
                    relation_pair[0], tokens, separator=token_separator)
                second_entity = get_str_from_tokens(
                    relation_pair[1], tokens, separator=token_separator)

                one_role_str = ' '.join(
                    [type_start, first_entity, type_middle, second_entity, type_end])
                span_str_list += [one_role_str]

            relation_str_rep_list += [' '.join(span_str_list)]

        source_text = token_separator.join(tokens)
        target_text = f'{type_start} ' + ' '.join(relation_str_rep_list) + f' {type_end}'

        return source_text, target_text


if __name__ == "__main__":
    pass
