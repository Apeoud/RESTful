# -*- coding: utf-8 -*-
import re
import jieba.posseg as pseg

from nltk import word_tokenize
from .Tuple import Tuple

not_valid = [",", "(", ")", ";", "''", "``", "'s", "-", "vs.", "v", "'", ":",
             ".", "--"]


def tokenize_entity(entity):
    parts = word_tokenize(entity)
    if parts[-1] == '.':
        replace = parts[-2] + parts[-1]
        del parts[-1]
        del parts[-1]
        parts.append(replace)
    return parts


def find_locations(entity_string, text_tokens):
    locations = []
    e_parts = tokenize_entity(entity_string)
    for i in range(len(text_tokens)):
        if text_tokens[i:i + len(e_parts)] == e_parts:
            locations.append(i)
    return e_parts, locations


class EntitySimple:
    def __init__(self, _e_string, _e_parts, _e_type, _locations):
        self.string = _e_string
        self.parts = _e_parts
        self.type = _e_type
        self.locations = _locations

    def __hash__(self):
        return hash(self.string) ^ hash(self.type)

    def __eq__(self, other):
        return self.string == other.string and self.type == other.type


class EntityLinked:
    def __init__(self, _e_string, _e_parts, _e_type, _locations, _url=None):
        self.string = _e_string
        self.parts = _e_parts
        self.type = _e_type
        self.locations = _locations
        self.url = _url

    def __hash__(self):
        return hash(self.url)

    def __eq__(self, other):
        return self.url == other.url


class Sentence(object):
    """对单个sentence建模，统计其中符合要求的pair以及相应的上下文，在本程序的数据结构中叫做tuple"""

    def __init__(self, sentence, config, pos_tagger=None):
        self.tuples = []
        self.tuple_extraction(sentence, config, pos_tagger)

    def tuple_extraction(self, sentence, config, pos_tagger):
        """
        利用sentence(string类型，包含用<LOC></LOC>包裹的实体 抽取可能的Tuple
        :param sentence: string类型，其中的实体用<ENTITY_TYPE></ENTITY_TYPE>包裹
        :param config: 配置信息，包含一些超参数。
        :return: 无返回值，但会更新对象的tuples属性
        """
        entity_regex = config.regex_simple if config.tag_type == "simple" else config.regex_linked
        entitys = [match for match in re.finditer(entity_regex, sentence)]

        if len(entitys) < 2:
            return

        entity_clean_regex = config.regex_clean_simple if config.tag_type == "simple" else config.regex_clean_linked
        sentence_no_tag = re.sub(entity_clean_regex, "", sentence)

        text_tokens = word_tokenize(sentence_no_tag)

        tagged_text = pos_tagger.tag(text_tokens)

        entities_info = set()
        for x in range(0, len(entitys)):
            if config.tag_type == "simple":
                entity = entitys[x].group()
                e_string = re.findall(config.regex_entity_text_simple, entity)[0]
                e_type = re.findall(config.regex_entity_type, entity)[0]
                e_parts, locations = find_locations(e_string, text_tokens)
                e = EntitySimple(e_string, e_parts, e_type, locations)
                entities_info.add(e)

        locations = dict()
        for e in entities_info:
            for start in e.locations:
                locations[start] = e

        sorted_keys = list(sorted(locations))
        for i in range(len(sorted_keys) - 1):
            for j in range(i + 1, len(sorted_keys)):
                # 两个实体的distance是抛去实体外的token的长度
                if j - i == 1:
                    distance = sorted_keys[j] - (sorted_keys[i] + len(locations[sorted_keys[i]].parts))
                else:
                    distance = 0
                    for k in range(i, j):
                        distance += sorted_keys[k + 1] - (sorted_keys[k] + len(locations[sorted_keys[k]].parts))
                if distance < config.min_tokens_away or distance > config.max_tokens_away:
                    break

                e1 = locations[sorted_keys[i]]
                e2 = locations[sorted_keys[j]]

                if e1.type == config.relationship.e1_type and e2.type == config.relationship.e2_type:
                    # ignore relationships between the same entity
                    if config.tag_type == "simple":
                        if e1.string == e2.string:
                            continue
                    elif config.tag_type == "linked":
                        if e1.url == e2.url:
                            continue
                    before = tagged_text[:sorted_keys[i]][-config.context_window_size:]
                    if j - i == 1:
                        between = tagged_text[sorted_keys[i] + len(locations[sorted_keys[i]].parts): sorted_keys[j]]
                    else:
                        between = []
                        for k in range(i, j):
                            between += tagged_text[sorted_keys[k] + len(locations[sorted_keys[k]].parts):sorted_keys[j]]

                    after = tagged_text[sorted_keys[j] + len(e2.parts):][:config.context_window_size]

                    # 过滤掉所有token
                    if all(token in not_valid for token, _ in between):
                        continue

                    # 添加到句子的提取实体中
                    self.tuples.append(Tuple(sentence, e1.string, e2.string, before, between, after, config))
        return


class EntityInfo(object):
    def __init__(self, entity_name, entity_type, begin, end):
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.begin = begin
        self.end = end

    def __unicode__(self):
        return self.entity_name + u'\t' + self.entity_type

    def __str__(self):
        return unicode(self).encode('utf-8')


class SentenceZh(object):
    def __init__(self, sentence, config, pos_tagger=None):
        self.tuples = []
        self.extract_tuples(sentence, config, pos_tagger)

    def extract_tuples(self, sentence, config, pos_tagger):
        entity_regex = config.regex_simple if config.tag_type == "simple" else config.regex_linked
        entity = [match for match in re.finditer(entity_regex, sentence)]
        if len(entity) < 2:
            return
        entity_info = []
        for i in range(len(entity)):
            ent = entity[i].group()
            e_string = re.findall(config.regex_entity_text_simple, ent)[0]
            e_type = re.findall(config.regex_entity_type, ent)[0]
            entity_info.append(EntityInfo(e_string, e_type, entity[i].start(), entity[i].end()))

        if len(entity_info) != len(entity):
            print "wrong "
            return

        for i in range(len(entity) - 1):
            for j in range(i + 1, len(entity)):
                e1 = entity_info[i]
                e2 = entity_info[j]

                # 判断类型是否符合
                if not e1.entity_type == config.relationship.e1_type or not e2.entity_type == config.relationship.e2_type:
                    print "type not compatible"
                    continue

                # bet
                bet = ""
                if j - i == 1:
                    bet = sentence[e1.end:e2.begin]
                elif j - i > 1:
                    for k in range(i, j):
                        bet += sentence[entity_info[k].end:entity_info[k + 1].begin]
                bet_tag = [(word, flag) for word, flag in pseg.cut(bet)]

                # 判断间隔是否符合
                if len(bet_tag) > config.max_tokens_away:
                    # print "distance not compatible"
                    continue

                # bef
                bef = sentence[:e1.begin]
                bef_tag = [(word, flag) for word, flag in pseg.cut(bef)]
                bef_tag = bef_tag[-config.bef_token_nums:]

                # aft
                aft = sentence[e2.end:]
                aft_tag = [(word, flag) for word, flag in pseg.cut(aft)]
                aft_tag = aft_tag[:config.aft_token_nums]

                self.tuples.append(Tuple(sentence, e1.entity_name, e2.entity_name, bef_tag, bet_tag, aft_tag, config))
