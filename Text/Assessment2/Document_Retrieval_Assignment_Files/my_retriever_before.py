from math import log, sqrt
from operator import itemgetter
from collections import Counter

class Retrieve:

    # Create new Retrieve object storing index and term weighting
    # scheme. (You can extend this method, as required.)
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.document_list = self.get_document_list()
        self.tfidf_document_dic = {}



    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # Method performing retrieval for a single query (which is
    # represented as a list of preprocessed terms). Returns list
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        final_results = self.calculaet_results(query)
        return final_results


    def calculaet_results(self, query):
        results_dic = {}
        final_result = []
        target_document = set()
        for word in query:
            target_document = target_document.union(set(self.index.get(word, {}).keys()))

        if self.term_weighting == "binary":
            for i in target_document:
                i = i-1
                same_count = len(set(query).intersection(set(self.document_list[i])))
                result = same_count ** 2 / sqrt((len(set(self.document_list[i])) * len(set(query))))
                results_dic[i+1] = result
            final_result = self.give_result(results_dic)

        if self.term_weighting == "tf":
            for i in target_document:
                i = i-1
                num_query, num_document, same_count = self.get_num(query, self.document_list[i])
                tf_query = self.compute_TF(num_query, query)
                tf_document = self.compute_TF(num_document, self.document_list[i])
                result = self.eval(tf_query, tf_document, same_count)
                results_dic[i+1] = result
            final_result = self.give_result(results_dic)

        if self.term_weighting == "tfidf":
            IDF_dic = self.compute_IDF(self.num_docs)

            for i in target_document:
                i = i-1
                num_query, num_document, same_count = self.get_num(query, self.document_list[i])
                tf_query = self.compute_TF(num_query, query)
                tfidf_query = self.tf_idf(tf_query,IDF_dic)
                if self.tfidf_document_dic.get(i, False):
                    tfidf_document = {}
                    for word in num_document.keys():
                        tfidf_document[word] = self.tfidf_document_dic[i].get(word, 0)
                    result = self.eval(tfidf_query, tfidf_document, same_count)

                else:
                    tf_document = self.compute_TF(num_document, self.document_list[i])
                    self.tfidf_document_dic[i] = self.tf_idf(tf_document, IDF_dic)
                    result = self.eval(tfidf_query, self.tfidf_document_dic[i], same_count)
                results_dic[i + 1] = result
            final_result = self.give_result(results_dic)

        return final_result



    def eval(self, weight_dic1, weight_dic2, same_count):
        up = 0
        down_dic1 = 0
        down_dic2 = 0
        weight_set1 = set(weight_dic1.keys())
        weight_set2 = set(weight_dic2.keys())
        cal_up = weight_set1.intersection(weight_set2)
        for word in cal_up:
            up += weight_dic1[word] * weight_dic2[word] * same_count
        for word in weight_set1:
            down_dic1 += weight_dic1[word] ** 2
        for word in weight_set2:
            down_dic2 += weight_dic2[word] ** 2
        down = sqrt(down_dic1 * down_dic2)
        result = up / down
        return result

    def give_result(self, results_dic):
        t = sorted([(v, k) for k, v in results_dic.items()], reverse=True)
        final_result = [k for v, k in t]
        return final_result[:10]

    def tf_idf(self, tf_vector, IDF_dic):
        tf_idf_dic = {}
        for word, value in tf_vector.items():
            if word in IDF_dic.keys():
                tf_idf_dic[word] = value * IDF_dic[word]
            else:
                tf_idf_dic[word] = 0
        return tf_idf_dic

    def get_num(self, query, document):
        total_words = set(query).union(set(document))
        same_words = set(query).intersection(set(document))
        same_count = len(same_words)

        num_query = dict.fromkeys(total_words, 0)
        for word in query:
            num_query[word] += 1

        num_document = dict.fromkeys(total_words, 0)
        for word in document:
            num_document[word] += 1

        return num_query, num_document, same_count

    def compute_TF(self, num_word, total_list):
        tf_dic = {}
        total = len(total_list)
        for word, count in num_word.items():
            tf_dic[word] = count / float(total)
        return tf_dic

    def compute_IDF(self, num_documents):
        IDF_dic = {}
        for word, position_dic in self.index.items():
            num_each_word = 0
            for i in position_dic.values():
                num_each_word += i
            IDF_dic[word] = log(num_documents / float(num_each_word))
        return IDF_dic

    def get_document_list(self):
        document_list = [[] for i in range(self.compute_number_of_documents())]
        for word, key in self.index.items():
            for docIndex, count in key.items():
                for i in range(count):
                    document_list[docIndex - 1].append(word)
        return document_list
