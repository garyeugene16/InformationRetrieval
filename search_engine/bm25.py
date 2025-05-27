# pakai library
# from rank_bm25 import BM25Okapi
# from .preprocessing import preprocess

# class BM25Engine:
#     def __init__(self, raw_docs):
#         self.tokenized_docs = [preprocess(doc) for doc in raw_docs]
#         self.bm25 = BM25Okapi(self.tokenized_docs)

#     def search(self, query, top_k=5):
#         query_tokens = preprocess(query)
#         scores = self.bm25.get_scores(query_tokens)
#         ranked_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
#         return ranked_ids[:top_k], scores

#Manual
import math
from .preprocessing import preprocess
from collections import defaultdict, Counter

class BM25Engine:
    def __init__(self, raw_docs, k=1.5, b=0.75):
        self.k = k
        self.b = b

        # Preprocessing
        self.tokenized_docs = [preprocess(doc) for doc in raw_docs]
        self.doc_lens = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens)

        self.doc_freqs = defaultdict(int)   # df: berapa dokumen mengandung term
        self.term_freqs = []                # tf: list of Counter per dokumen

        # Hitung df dan tf - indexing
        for tokens in self.tokenized_docs:
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1

        self.N = len(self.tokenized_docs)  # jumlah dokumen

    def idf(self, term):
        # Inverse Document Frequency (menggunakan formula BM25)
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_tokens, doc_idx):
        score = 0
        doc_tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        for term in query_tokens:
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            idf = self.idf(term)
            numerator = tf * (self.k + 1)
            denominator = tf + self.k * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (numerator / denominator)
        return score

    def search(self, query, top_k=5):
        query_tokens = preprocess(query)
        scores = [self.score(query_tokens, i) for i in range(len(self.tokenized_docs))]
        ranked_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked_ids[:top_k], scores
