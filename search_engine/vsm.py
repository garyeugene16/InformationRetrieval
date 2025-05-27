#Library
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from .preprocessing import preprocess

# class VSMEngine:
#     def __init__(self, raw_docs):
#         self.raw_docs = raw_docs
#         self.cleaned_docs = [" ".join(preprocess(doc)) for doc in raw_docs]
#         self.vectorizer = TfidfVectorizer(sublinear_tf=True)
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_docs)

#     def search(self, query, top_k=5):
#         query_clean = " ".join(preprocess(query))
#         query_vec = self.vectorizer.transform([query_clean])
#         scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
#         ranked_ids = scores.argsort()[::-1][:top_k]
#         return ranked_ids, scores

#Manual
import math
from collections import defaultdict, Counter
from .preprocessing import preprocess

class VSMEngine:
    def __init__(self, raw_docs):
        self.tokenized_docs = [preprocess(doc) for doc in raw_docs]
        self.doc_count = len(self.tokenized_docs)

        # Bangun term frequency (TF) dan document frequency (DF) - indexing
        self.tf_list = []
        self.df = defaultdict(int)
        for tokens in self.tokenized_docs:
            tf = Counter(tokens)
            self.tf_list.append(tf)
            for term in tf:
                self.df[term] += 1

        # Buat vocabulary dan term index
        self.vocab = sorted(self.df.keys())
        self.term_index = {term: idx for idx, term in enumerate(self.vocab)}

        # Hitung IDF
        self.idf = {}
        for term in self.vocab:
            df = self.df[term]
            self.idf[term] = math.log((self.doc_count) / (df + 1)) + 1  # smoothed idf

        # Buat matrix TF-IDF untuk dokumen
        self.doc_vectors = [self._compute_vector(tf) for tf in self.tf_list]

    def _compute_vector(self, tf_counter):
        vec = [0.0] * len(self.vocab)
        for term, tf in tf_counter.items():
            if term in self.term_index:
                idx = self.term_index[term]
                tf_weight = 1 + math.log(tf) if tf > 0 else 0  # sublinear tf
                vec[idx] = tf_weight * self.idf[term]
        return self._normalize(vec)

    def _normalize(self, vec):
        norm = math.sqrt(sum(x ** 2 for x in vec))
        return [x / norm for x in vec] if norm != 0 else vec

    def _cosine_similarity(self, vec1, vec2):
        return sum(a * b for a, b in zip(vec1, vec2))

    def search(self, query, top_k=5):
        query_tokens = preprocess(query)
        query_tf = Counter(query_tokens)
        query_vec = self._compute_vector(query_tf)

        scores = [self._cosine_similarity(query_vec, doc_vec) for doc_vec in self.doc_vectors]
        ranked_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked_ids[:top_k], scores
