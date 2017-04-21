#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import json
import codecs
import json

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from .ReVerb import ReVerb


class Seed(object):
    def __init__(self, _e1, _e2):
        self.e1 = _e1
        self.e2 = _e2

    def __hash__(self):
        return hash(self.e1) ^ hash(self.e2)

    def __eq__(self, other):
        return self.e1 == other.e1 and self.e2 == other.e2

    def toDict(self):
        return [self.e1, self.e2]


class Relationship(object):
    def __init__(self, rel_config):
        # 内聚性太强 需要修改
        self.name = rel_config["name"]
        self.e1_type = rel_config["e1_type"]
        self.e2_type = rel_config["e2_type"]
        self.positive_seed = set([Seed(seed[0], seed[1]) for seed in rel_config["positive_seed"]])
        self.negative_seed = set([Seed(e1, e2) for e1, e2 in rel_config["negative_seed"]])

    def toDict(self):
        return {
            'name': self.name,
            'e1_type': self.e1_type,
            'e2_type': self.e2_type,
            'positive_seeds': [seed.toDict() for seed in self.positive_seed],
            'negative_seeds': [seed.toDict() for seed in self.negative_seed]
        }


class Config(object):
    def __init__(self, config_file):
        config = json.load(open(config_file, 'r'))

        # 词性过滤和tag map
        self.filter_pos = config['filter_pos']
        # 正则表达式 提取
        self.regex_clean_simple = re.compile(config['re']['regex_clean_simple'], re.U)
        self.regex_clean_linked = re.compile(config['re']['regex_clean_linked'], re.U)
        self.regex_simple = re.compile(config['re']['regex_simple'], re.U)
        self.regex_linked = re.compile(config['re']['regex_linked'], re.U)
        self.regex_entity_text_simple = re.compile(config['re']['regex_entity_text_simple'])
        self.regex_entity_text_linked = re.compile(config['re']['regex_entity_text_linked'])
        self.regex_entity_type = re.compile(config['re']['regex_entity_type'])
        self.tags_regex = re.compile(config['re']['tags_regex'], re.U)

        # 待抽取的关系描述，包含关系的两个实体类型

        self.relationship = Relationship(config["relationship"])

        # hyper-parameters
        self.wUpdt = config['hyper_parameters']['wUpdt']
        self.wUnk = config['hyper_parameters']['wUnk']
        self.wNeg = config['hyper_parameters']['wNeg']
        self.number_iterations = config['hyper_parameters']['number_iterations']
        self.min_pattern_support = config['hyper_parameters']['min_pattern_support']
        self.max_tokens_away = config['hyper_parameters']['max_tokens_away']
        self.min_tokens_away = config['hyper_parameters']['min_tokens_away']

        self.alpha = config['context_weight']['alpha']
        self.beta = config['context_weight']['beta']
        self.gamma = config['context_weight']['gamma']
        self.tag_type = config['hyper_parameters']['tag_type']
        self.context_window_size = config['hyper_parameters']['context_window_size']

        self.similarity = config['similarity']['similarity']
        self.confidence = config['similarity']['confidence']

        # 工程地质
        self.project_path = os.path.dirname(os.path.dirname(__file__))
        self.word2vec_path = self.project_path + config["path"]["dirpath"] + config["path"]["word2vec"]
        self.sentences_path = self.project_path + config["path"]["dirpath"] + config["path"]["sentences"]
        self.processed_path = self.project_path + config["path"]["dirpath"] + config["path"]["processed_tuples"]

        # word2vec 模型导入
        self.word2vec = None
        self.vec_dim = None
        self.read_word2vec()

        # ReVerb词性抽取
        self.reverb = ReVerb()

        # stopwords
        self.stopwords = self.load_stopwords('en')

    def read_word2vec(self):
        print "loading word2vec model ...\n"
        self.word2vec = Word2Vec.load_word2vec_format(self.word2vec_path, binary=True)
        self.vec_dim = self.word2vec.layer1_size
        print self.vec_dim, "dimensions"

    @staticmethod
    def load_stopwords(language='en'):
        if language == 'en':
            return stopwords.words('english')
        if language == 'cn':
            with codecs.open('../data/stopwords.zh.txt') as f:
                return [word for word in f.readlines()]
