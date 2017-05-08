# -*- coding: utf-8 -*-
import os
import time
import sys
import json
import codecs
import cPickle

from numpy import dot
from nltk.data import load
from gensim import matutils
from collections import defaultdict

from app.RelationExtraction.dataStructure.Sentence import Sentence
from app.RelationExtraction.dataStructure.Config import Config, Seed, Relationship
from app.RelationExtraction.dataStructure.Pattern import Pattern
from app.RelationExtraction.dataStructure.Tuple import Tuple


class AUTORE(object):
    def __init__(self, config_file):
        self.curr_iteration = 0
        self.patterns = list()
        self.processed_tuples = list()
        self.candidate_tuples = defaultdict(list)
        self.config = Config(config_file)

    def extract_tuples(self, sentence_file, localsaved=False):

        if localsaved:
            # 先查找是否有提取出的用cPickle备份好的Tuple，如果有则从本地导入tuples，调试时使用能减少时间。
            # os.path.isfile("../data/processed_tuples.pkl")
            f = open(self.config.processed_path, 'r')
            processed_tuples = json.load(f)['processed_tuples']
            self.processed_tuples = [
                Tuple(_sentence="", _e1=tuple_dict['e1'], _e2=tuple_dict['e2'], _before=tuple_dict['bef_tags'],
                      _between=tuple_dict['bet_tags'], _after=tuple_dict['aft_tags'], config=self.config, toVector=True)
                for
                tuple_dict in processed_tuples]
            print "load tuples finished with %d tuples" % (len(self.processed_tuples))
            f.close()
        else:
            # 如果没有将Tuple保存好，则从给定的文件中按照给定的方法提取。
            print "io error occurs"
            with codecs.open(sentence_file, encoding='utf-8') as f:
                begin = time.time()
                count = 0
                tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')
                for line in f.readlines():
                    count += 1
                    if count % 1000 == 0:
                        sys.stdout.write('.')

                    sentence = Sentence(sentence=line.strip(), config=self.config, pos_tagger=tagger)

                    for tup in sentence.tuples:
                        self.processed_tuples.append(tup)
            print time.time() - begin
            print "extract tuples finished with %d tuples " % (len(self.processed_tuples))
            f = open(self.config.processed_path, 'wb')
            json.dump({"processed_tuples": [pro_tuple.toDict() for pro_tuple in self.processed_tuples]}, f)
            f.close()
            print "dump tuples successfully"

        return

    def extract_matched_tuples(self):
        """
        在候选Tuple中找到与种子(POSITIVE)匹配的Tuple，并统计每一个Tuple出现的次数
        :return: matched_tuples : 是list of tuples counts : 是[e1,e2]为key的字典，value是该实体对出现的次数。
        """
        matched_tuples = list()
        counts = dict()
        for tup in self.processed_tuples:
            for s in self.config.relationship.positive_seed:
                if tup.e1 == s.e1 and tup.e2 == s.e2:
                    matched_tuples.append(tup)
                    try:
                        counts[(tup.e1, tup.e2)] += 1
                    except KeyError:
                        counts[(tup.e1, tup.e2)] = 1
        print "extract matched tuples finished with %d tuples" % (len(matched_tuples))
        return matched_tuples, counts

    def similarity_tuple_tuple(self, t1, t2):
        """
        计算两个Tuple之间的相似度，主要e1之前的部分，e1和e2之间的部分，e2之后的部分，三部分相似度的加权和
        :param t1: 
        :param t2: 
        :return: 
        """
        (bef, bet, aft) = (0, 0, 0)
        if t1.bef_vector is not None and t2.bef_vector is not None:
            bef = dot(matutils.unitvec(t1.bef_vector), matutils.unitvec(t2.bef_vector))
        if t1.bet_vector is not None and t2.bet_vector is not None:
            bet = dot(matutils.unitvec(t1.bet_vector), matutils.unitvec(t2.bet_vector))
        if t1.aft_vector is not None and t2.aft_vector is not None:
            aft = dot(matutils.unitvec(t1.aft_vector), matutils.unitvec(t2.aft_vector))

        return self.config.alpha * bef + self.config.beta * bet + self.config.gamma * aft

    def similarity_tuple_pattern(self, tup, pattern):
        """
        比较一个Tuple和一个pattern的相似度，主要是和pattern中每个tuple进行比较
        :param tup: 
        :param pattern: 
        :return: 
        """
        good = 0
        bad = 0
        max_similarity = 0

        for p in list(pattern.tuples):
            score = self.similarity_tuple_tuple(tup, p)
            max_similarity = score if score > max_similarity else max_similarity
            if score >= self.config.similarity:
                good += 1
            else:
                bad += 1
        if good >= bad:
            return True, max_similarity
        else:
            return False, 0.0

    def extract_patterns(self, matched_tuples):
        """
        利用查找到的与种子匹配的Tuple来提取一些Pattern。
        :param matched_tuples: 
        :return: 
        """
        if len(self.patterns) == 0:
            c1 = Pattern(matched_tuples[0])
            self.patterns.append(c1)

        count = 0
        for t in matched_tuples:
            count += 1
            if count % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            max_similarity = 0
            max_similarity_cluster_index = 0

            for i in range(0, len(self.patterns), 1):
                pattern = self.patterns[i]
                accept, score = self.similarity_tuple_pattern(t, pattern)
                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            if max_similarity < self.config.similarity:
                c = Pattern(t)
                self.patterns.append(c)
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(t)

        self.patterns = [p for p in self.patterns if len(p.tuples) > self.config.min_pattern_support]
        print "extract patterns finished with %d patterns and pattern distribution %s" % (
            len(self.patterns), str([len(patt.tuples) for patt in self.patterns]))
        return 0

    def extract_candidate_tuples(self):
        """
        挑选符合关系的tuple
        :return: 
        """
        for t in self.processed_tuples:
            sim_best = 0
            # 第一步是用所有的pattern对该Tuple打分，并记录最高分和对应的pattern
            for pattern in self.patterns:
                accept, score = self.similarity_tuple_pattern(t, pattern)
                if accept is True:
                    pattern.update_selectivity(t, self.config)
                    if score > sim_best:
                        sim_best = score
                        pattern_best = pattern
            # 如果最高分高于设定的阈值，则保存对应的pattern
            if sim_best >= self.config.similarity:
                patterns = self.candidate_tuples[t]
                if patterns is not None:
                    if pattern_best not in [x[0] for x in patterns]:
                        self.candidate_tuples[t].append((pattern_best, sim_best))
                else:
                    self.candidate_tuples[t].append((pattern_best, sim_best))
        print "extract candidate tuples finished with %d" % (len(self.candidate_tuples))
        return 0

    def update_pattern_confidence(self):
        for p in self.patterns:
            p.update_confidence(self.config)
        return 0

    def update_candidate_tuples(self):
        """
        根据更新的pattern得分来更新候选Tuple的得分
        :return: 
        """
        for t in self.candidate_tuples.keys():
            confidence = 1
            for p in self.candidate_tuples.get(t):
                confidence *= 1 - (p[0].confidence * p[1])
            t.confidence = 1 - confidence

        return 0

    def update_seeds(self):
        for t in self.candidate_tuples.keys():
            if t.confidence >= self.config.confidence:
                seed = Seed(t.e1, t.e2)
                self.config.relationship.positive_seed.add(seed)

    def bootstrap(self, sentence_file):
        """
        进行半自动bootstrap的迭代过程
        :return: 
        """
        self.extract_tuples(sentence_file)

        while self.curr_iteration < self.config.number_iterations:
            matched_tuples, counts = self.extract_matched_tuples()

            if len(matched_tuples) == 0:
                print "No seed matches found"
                return False

            self.extract_patterns(matched_tuples)

            self.extract_candidate_tuples()

            self.update_pattern_confidence()
            self.update_candidate_tuples()
            self.curr_iteration += 1
        pass

    def dump_model(self, filepath="test2.json"):
        dump_info = dict()
        dump_info['relation'] = self.config.relationship.toDict();
        dump_info['patterns'] = [pattern.toDict() for pattern in self.patterns]

        try:
            f = open(os.path.join(os.path.dirname(__file__), filepath), 'w')
            json.dump(dump_info, f)
        except EOFError:
            print "error occurs when dump model"

    def load_model(self, filepath="test2.json"):
        try:
            f = open(os.path.join(os.path.dirname(__file__), filepath), 'r')
            info = json.load(f)
            patterns = list()
            for patt in info['patterns']:
                match_tuples = [
                    Tuple(_sentence="", _e1=tuple_info['e1'], _e2=tuple_info['e2'], _before=tuple_info['bef_tags'],
                          _between=tuple_info['bet_tags'], _after=tuple_info['aft_tags'], config=self.config,
                          toVector=True) for tuple_info in patt['tuples']]
                p = Pattern(match_tuples)
                p.confidence = patt['confidence']
                p.positive = patt['positive']
                p.negative = patt['negative']
                p.unknown = patt['unknown']
                patterns.append(p)
            self.patterns = patterns
            relation_info = info['relation']
            self.config.relationship.e1_type = relation_info["e1_type"]
            self.config.relationship.e2_type = info['relation']["e2_type"]
            self.config.relationship.name = info['relation']["name"]
            self.config.relationship.positive_seed = [Seed(seed_info[0], seed_info[1]) for seed_info in
                                                      info['relation']["positive_seeds"]]
            self.config.relationship.negative_seed = [Seed(seed_info[0], seed_info[1]) for seed_info in
                                                      info['relation']["positive_seeds"]]

        except EOFError:
            print "error occurs when load model"

    def score(self, sentence):
        tagger = load('taggers/maxent_treebank_pos_tagger/english.pickle')
        sent = Sentence(sentence=sentence, config=self.config, pos_tagger=tagger)
        for i in range(len(sent.tuples)):
            sent.tuples[i].construct_vector(self.config)
        result = dict()
        for tup in sent.tuples:
            sim_best = 0
            for p in self.patterns:
                accept, score = self.similarity_tuple_pattern(tup, p)
                if accept is True:
                    if score > sim_best:
                        sim_best = score
            result[(tup.e1, tup.e2)] = sim_best
        return result


def main():
    sentence_ex = "<ORG>AOL</ORG> , which is based in <LOC>New York</LOC> but also has major operations " \
                  "in <LOC>Northern Virginia</LOC> , said it will take about $ 200 million in charges for " \
                  "severance and other costs related to the restructuring ."
    re = AUTORE(config_file=os.path.join(os.path.dirname(__file__), 'parameter.json'))
    re.bootstrap(
        sentence_file='/Users/duanshangfu/PycharmProjects/RESTful/app/RelationExtraction/data/sentences_zh_zgs.txt')
    # re.load_model()
    result = re.score(sentence_ex)
    # re.dump_model()
    print result


def create_instance(config_file='parameter.json'):
    re_instance = AUTORE(config_file=config_file)
    re_instance.load_model()

    return re_instance


if __name__ == "__main__":
    main()
