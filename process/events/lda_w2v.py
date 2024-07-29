# from typing import List, Tuple
# from gensim import corpora
# from gensim.models.coherencemodel import CoherenceModel
# from gensim.models.ldamodel import LdaModel
# from helper.time_util import cost_time


# class LDAEncoder:
#     def __init__(self, num_topics=10, num_words=10, passes=30, random_state=1, minimum_probability=1e-8):
#         self.num_topics = num_topics
#         self.num_words = num_words
#         self.passes = passes
#         self.random_state=random_state
#         self.minimum_probability = minimum_probability

#         self.data_set: List[List[str]] = None
#         self.dictionary: corpora.Dictionary = None
#         self.corpus: List[List[int]] = None
#         self.ldamodel: LdaModel = None

#     def print_topics(self):
#         return self.ldamodel.print_topics(num_topics=self.num_topics, num_words=self.num_words)


#     def get_conference(self):
#         ldacm = CoherenceModel(model=self.ldamodel, texts=self.data_set, dictionary=self.dictionary, coherence='c_v')
#         return ldacm.get_coherence()


#     def get_perplexity(self):
#         return self.ldamodel.log_perplexity(self.corpus)

#     def fit(self, data_set: List[List[str]], labels) -> LdaModel:
#         self.data_set = data_set
#         self.dictionary = corpora.Dictionary(self.data_set)
#         self.corpus = [self.dictionary.doc2bow(text) for text in self.data_set]

#         self.ldamodel = LdaModel(self.corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=self.passes,
#                                  random_state=self.random_state, minimum_probability=self.minimum_probability)
#         return self.ldamodel

#     def update(self, data_set: List[List[str]]) -> LdaModel:
#         if self.data_set is None:
#             return self.fit(data_set)

#         self.data_set.extend(data_set)
#         self.ldamodel.id2word.add_documents(data_set)
#         corpus = [self.ldamodel.id2word.doc2bow(text) for text in data_set]
#         self.ldamodel.update(corpus)
#         return self.ldamodel


#     def get_sentence_embedding(self, text: List[str]) -> List[float]:
#         corpus = self.ldamodel.id2word.doc2bow(text)
#         pre_result = self.ldamodel[corpus]
#         result = [0.0 for i in range(self.num_topics)]
#         for topic_id, topic_value in pre_result:
#             result[topic_id] = topic_value
#         return result