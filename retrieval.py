from typing import Any
import pandas as pd
from datasets import load_dataset
import time
import json
import numpy as np
import pickle
import torch
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
import string

class Retriever:
    def __init__(self, 
                corpus_path: str = 'documents_chunk/documents_chunk.json', 
                corpus_emb_path: str = 'documents_chunk/documents_embedding.pkl',
                model_embedding: str = 'model/bi-encoder-2epochs', 
                eval_path: str = 'data/test.json'):
        
        print("-----------------------------------------------------------------")
        print("--------------Read the corpus and embedding corpus---------------")
        print("-----------------------------------------------------------------")

        self.meta_corpus = load_dataset(
                "json",
                data_files=corpus_path,
                split="train"
            ).to_list()
        
        with open(corpus_emb_path, 'rb') as f:
            self.corpus_embs = pickle.load(f)
        print("---> Read the corpus and embedding corpus sucessfull.")

        print("-----------------------------------------------------------------")
        print("-----------------Initialize the BM25 retriever-------------------")
        print("-----------------------------------------------------------------")
        self.bm25 = self.init_bm25()
        print("---> Initialize the BM25 retriever sucessfull.")


        print("---------------------------------------------------------------------")
        print("--------------------Loading the embedding model----------------------")
        print("---------------------------------------------------------------------")
        self.embedder = SentenceTransformer(model_embedding)
        if torch.cuda.is_available() == True:
           self.embedder.cuda()
        print("---> Load embedding model sucessfull.")
        
        self.eval_path = eval_path

        
    def split_text(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.lower().split()
        words = [word for word in words if len(word.strip()) > 0]
        return words
    
    def init_bm25(self):
        tokenized_corpus = [self.split_text(doc["chunk"]) for doc in tqdm(self.meta_corpus)]
        bm25 = BM25Plus(tokenized_corpus)
        return bm25

    def retrieve(self, question, topk=50, w_bm25 = 0.7, w_emb = 0.3):
        """
        Get most relevant chunks to the question using combination of BM25 and semantic scores.
        """
        ## initialize query for each retriever (BM25 and semantic)
        tokenized_query = self.split_text(question)
        segmented_question = tokenize(question)
        question_emb = self.embedder.encode([segmented_question])
        question_emb /= np.linalg.norm(question_emb, axis=1)[:, np.newaxis]
        # question_emb = co.embed([segmented_question], input_type="search_query", model="embed-multilingual-v3.0").embeddings
        ## get BM25 and semantic scores
        bm25_scores = self.bm25.get_scores(tokenized_query)
        semantic_scores = question_emb @ self.corpus_embs.T
        semantic_scores = semantic_scores[0]

        ## update chunks' scores. 
        max_score = max(bm25_scores)
        min_score = min(bm25_scores)

        def normalize(x):
            return (x - min_score + 0.1) / \
            (max_score - min_score + 0.1)
            
        corpus_size = len(self.meta_corpus)
        for i in range(corpus_size):
            self.meta_corpus[i]["bm25_score"] = bm25_scores[i]
            self.meta_corpus[i]["bm25_normed_score"] = normalize(bm25_scores[i])
            self.meta_corpus[i]["semantic_score"] = semantic_scores[i]

        ## compute combined score (BM25 + semantic)
        for passage in self.meta_corpus:
            passage["combined_score"] = (passage["bm25_normed_score"] * w_bm25 + \
                                        passage["semantic_score"] * w_emb)

        ## sort passages by the combined score
        sorted_passages = sorted(self.meta_corpus, key=lambda x: x["combined_score"], reverse=True)
        return sorted_passages[:topk]
    
    def evaluate(self, topk = 10, w_bm25 = 0.3, w_emb = 0.7):
        print("-----------------------------------------------------------------")
        print("----------------------------Evaluate-----------------------------")
        print("-----------------------------------------------------------------")
        with open(self.eval_path, 'r', encoding='utf-8') as file:
            test = json.load(file)
        test = pd.DataFrame(test)

        mrr = [0] * topk
        recall = [0] * topk
        # print(accuracy)
        number_test = 0

        for idx in tqdm(range(len(test))):
            que = test['question'][idx]
            number_test += 1
            top_passages = self.retrieve(que, topk=topk, w_bm25=w_bm25, w_emb=w_emb)
            top_chunks_id = list(pd.DataFrame(top_passages)['chunk_id'])
            try:
                first_occurrence = top_chunks_id.index(int(test['label_chunk_id'][idx]))
            except ValueError:
                continue
            
            for j in range(first_occurrence, topk):
                mrr[j] += 1 / (first_occurrence + 1)
                recall[j] += 1

        mrr = list(map(lambda x: x / number_test, mrr))
        recall = list(map(lambda x: x / number_test, recall))
        print(f"Weight = {w_bm25:.1f} * BM25 + {w_emb:.1f} Embedding")
        print("| MRR@k | Value | Recall@k | Value |")
        print("|-------|-------|----------|-------|")
        for k in range(topk):
            print(f"| MRR@{k + 1} | {mrr[k]:.4f} | Recall@{k + 1} | {recall[k]:.4f} |")

        return mrr, recall
    def __call__(self, question, topk = 1, w_bm25 = 0.3, w_emb = 0.7):
        top_passages = self.retrieve(question, topk=topk, w_bm25 = w_bm25, w_emb = w_emb)
        return top_passages
    
if __name__ == "__main__":
    re = Retriever()
    # print(re("Bộ mộn Toán tin có bao nhiều cán bộ?"))
    for a in range(0, 11):
        a /= 10
        b = 1 - a
        re.evaluate(topk=10, w_bm25=a, w_emb=b)
        print('\n')
    