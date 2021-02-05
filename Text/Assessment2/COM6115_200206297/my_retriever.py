import math
from collections import Counter

class Retrieve:

    # Create new Retrieve object storing index and term weighting
    # scheme. (You can extend this method, as required.)
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.document_list = self.get_document_list()
        self.tf_dic = {}
        self.norm_document = {}
        self.tfidf_dic = {}

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
                result = same_count ** 2 / math.sqrt((len(set(self.document_list[i])) * len(set(query))))
                results_dic[i+1] = result
            final_result = self.give_result(results_dic)

        if self.term_weighting == "tf":
            tf_query = self.compute_TF(query)
            for doc_ind in target_document:
                doc_ind = doc_ind - 1
                tf_document = self.tf_dic.get(doc_ind, False)
                if not tf_document:
                    tf_document = self.compute_TF(self.document_list[doc_ind])
                    self.tf_dic[doc_ind] = tf_document
                inter = set(query).intersection(set(self.document_list[doc_ind]))
                result = self.eval(inter, tf_query, tf_document, doc_ind)
                results_dic[doc_ind + 1] = result
            final_result = self.give_result(results_dic)

        if self.term_weighting == "tfidf":
            tfidf_query = self.compute_tfidf(query)
            for doc_ind in target_document:
                doc_ind = doc_ind -1
                tfidf_document = self.tfidf_dic.get(doc_ind, False)
                if not tfidf_document:
                    tfidf_document = self.compute_tfidf(self.document_list[doc_ind])
                    self.tfidf_dic[doc_ind] = tfidf_document
                inter = set(query).intersection(set(self.document_list[doc_ind]))
                result = self.eval(inter, tfidf_query, tfidf_document, doc_ind)
                results_dic[doc_ind + 1] = result
            final_result = self.give_result(results_dic)

        return final_result


    def eval(self, inter, target_query, target_document, doc_ind):
        up = down_query = 0
        for word in inter:
            up += target_query[word] * target_document[word]
        for word in target_query.keys():
            down_query += target_query[word] ** 2
        down_document = self.norm_document.get(doc_ind, False)
        if not down_document:
            for word in target_document.keys():
                down_document += target_document[word] ** 2
                self.norm_document[doc_ind] = down_document
        result = len(inter) * up / (down_query * down_document) ** 0.5
        return result

    def give_result(self, results_dic):
        t = sorted([(v, k) for k, v in results_dic.items()], reverse=True)
        final_result = [k for v, k in t]
        return final_result[:10]

    def compute_TF(self, target):
        collect = Counter(target)
        n = len(target)
        for word in collect.keys():
            collect[word] = collect[word] / n
        return collect

    def compute_tfidf(self, target):
        tfidf_dict = self.compute_TF(target)
        for word in tfidf_dict.keys():
            tfidf_dict[word] *= math.log(self.num_docs / (len(self.index.get(word, {}))+1))
        return tfidf_dict

    def get_document_list(self):
        document_list = [[] for i in range(self.compute_number_of_documents())]
        for word, key in self.index.items():
            for docIndex, count in key.items():
                for i in range(count):
                    document_list[docIndex - 1].append(word)
        return document_list