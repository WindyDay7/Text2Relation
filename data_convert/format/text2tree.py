#!/usr/bin/env python
# -*- coding:utf-8 -*-
from data_convert.format.target_format import TargetFormat


def get_str_from_tokens(tokens, sentence, separator=' '):
    start, end_exclude = tokens[0], tokens[-1] + 1
    return separator.join(sentence[start:end_exclude])


Temp_start = '<Temp_S>'
Temp_end = '<Temp_E>'
Relation_start = '<Relation_S>'
Relation_end = '<Relation_E>'
Entity_Type = {"ORG":"<ORG>", "VEH":"<VEH>","WEA":"<WEA>", "LOC":"<LOC>", "FAC":"<FAC>","PER":"<PER>", "GPE":"<PER>"}
Entity_End = "<End>"


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
        relation_str_rep_list = list()
        # if(len(predicate_arguments)>2):
        #     print(predicate_arguments)
        for predicate_argument in predicate_arguments:
            relation_type = predicate_argument['type']

            # predicate_argument['tokens'] is the trigger index
            # tokens is the sentence tokens, we get the trigger text span here
            # predicate_text = get_str_from_tokens(predicate_argument['tokens'], tokens, separator=token_separator)

            # prefix_tokens[predicate_argument['tokens'][0]] = ['[ ']
            # suffix_tokens[predicate_argument['tokens'][-1]] = [' ]']
            if relation_type != "PART-WHOLE" and relation_type != "ORG-AFF" and relation_type != "GEN-AFF":
                continue
            # print(predicate_argument)
            # role_name is the argument role, role_tokens are corresponding text span index
            role_str_list = list()
            for relation_pair in predicate_argument['arguments']:
                # get the role text span from role tokens index
                # print(role_tokens)
                
                first_entity = get_str_from_tokens(relation_pair[0], tokens, separator=token_separator)
                second_entity = get_str_from_tokens(relation_pair[2], tokens, separator=token_separator)
                entity1 = Entity_Type[relation_pair[1]]
                entity2 = Entity_Type[relation_pair[3]]
                one_role_str = ' '.join([entity1, first_entity, entity2, second_entity, Entity_Type["End"]])
                role_str_list += [one_role_str]

            role_str_list_str = ' '.join(role_str_list)
            relation_str_rep = f"{Relation_start} {relation_type} {role_str_list_str} {relation_type} {Relation_end}"
            relation_str_rep_list += [relation_str_rep]

        source_text = token_separator.join(tokens)
        target_text = ' '.join(relation_str_rep_list)
        if not multi_tree:
            target_text = f'{Relation_start} ' + \
                          ' '.join(relation_str_rep_list) + f' {Relation_end}'

        return source_text, target_text

    


if __name__ == "__main__":
    pass
