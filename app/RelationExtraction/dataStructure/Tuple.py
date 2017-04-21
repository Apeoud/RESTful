# -*- coding: utf-8 -*-
import json
from numpy import zeros


class Tuple(object):
    bef_vector = None
    bet_vector = None
    aft_vector = None

    def __init__(self, _sentence, _e1, _e2, _before, _between, _after, config, toVector=False):
        self.sentence = _sentence

        self.e1 = _e1
        self.e2 = _e2

        self.confidence = 0

        # context的词序列和词的标签
        self.bef_tags = _before
        self.bet_tags = _between
        self.aft_tags = _after

        # _before 等的格式 [("I", "ADV"),("am", "V")]
        self.bef_words = " ".join([token for token, _ in _before])
        self.bet_words = " ".join([token for token, _ in _between])
        self.aft_words = " ".join([token for token, _ in _after])

        # vector是一定维度的用来表示词语义的向量
        if toVector:
            self.construct_vector(config)
        return

    def toJson(self):
        return json.dumps(self.toDict())

    def toDict(self):
        return {
            'e1': self.e1,
            'e2': self.e2,
            'bef_tags': self.bef_tags,
            'bet_tags': self.bet_tags,
            'aft_tags': self.aft_tags,
        }

    def __str__(self):
        return str(self.e1 + '\t' +
                   self.e2 + '\t' + self.bef_words + '\t' + self.bet_words + '\t' + self.aft_words).encode("utf8")

    def __hash__(self):
        return hash(self.e1) ^ hash(self.e2) ^ hash(self.bef_words) ^ \
               hash(self.bet_words) ^ hash(self.aft_words)

    def __eq__(self, other):
        return (self.e1 == other.e1 and self.e2 == other.e2 and
                self.bef_words == other.bef_words and
                self.bet_words == other.bet_words and
                self.aft_words == other.aft_words)

    def __cmp__(self, other):
        if other.confidence > self.confidence:
            return -1
        elif other.confidence < self.confidence:
            return 1
        else:
            return 0

    def construct_vector(self, config):
        """
        
         
        :param config: 
        :return: 
        """
        reverb_pattern = config.reverb.extract_reverb_patterns_tagged_ptb(self.bet_tags)
        bet_words = reverb_pattern if len(reverb_pattern) > 0 else self.bet_tags

        bet_filtered = [token for token, tag in bet_words if
                        token.lower() not in config.stopwords and tag not in config.filter_pos]

        self.bet_vector = self.context2vector(bet_filtered, config)

        bef_no_tags = [t[0] for t in self.bef_tags if t[0].lower() not in config.stopwords]
        aft_no_tags = [t[0] for t in self.aft_tags if t[0].lower() not in config.stopwords]
        self.bef_vector = self.context2vector(bef_no_tags, config)
        self.aft_vector = self.context2vector(aft_no_tags, config)

        return

    @staticmethod
    def context2vector(tokens, config):
        """
        token列表的word2vec值的加权平均
        :param tokens: 
        :param config: 
        :return: 
        """
        vector = zeros(config.vec_dim)
        for token in tokens:
            try:
                vector += config.word2vec[token.strip()]
            except KeyError:
                continue

        return vector
