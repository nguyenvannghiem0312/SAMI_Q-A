from typing import Any
import pandas as pd
from retrieval import Retriever
from generate import GenerateModel
from tqdm import tqdm
import json
import csv
import os
from rouge_score import rouge_scorer
  ###
        # model_list = ["model/seallms/seallm-7b-chat.q4_k_s.gguf",
        #  "model/vinallama/vinallama-2.7b-chat_q5_0.gguf",
        #  "model/vinallama/vinallama-7b-chat_q5_0.gguf"]
    ###
class RAGModel:
    def __init__(self, corpus_path: str = 'documents_chunk/documents_chunk.json', 
                corpus_emb_path: str = 'documents_chunk/documents_embedding.pkl',
                model_embedding: str = 'model/bi-encoder-2epochs',
                model_path: str = 'model/vinallama/vinallama-7b-chat_q5_0.gguf',
                prompt:str='VINALLAMA', #SEALLM, VINALLAMA
                eval_path: str = 'data/test.json'):
        
        self.retriever = Retriever(corpus_path=corpus_path,
                                  corpus_emb_path=corpus_emb_path,
                                  model_embedding=model_embedding)
        self.generator = GenerateModel(model_path=model_path,
                                       prompt=prompt)
        self.eval_path = eval_path
    
    def evaluate_retriever(self, topk = 10, w_bm25 = 0.3, w_emb = 0.7):
        print("-----------------------------------------------------------------")
        print("-----------------------Evaluate retrieve--------------------------")
        print("-----------------------------------------------------------------")
        with open(self.eval_path, 'r', encoding='utf-8') as file:
            test = json.load(file)
        test = pd.DataFrame(test)

        mrr = [0] * topk
        recall = [0] * topk
        number_test = 0

        for idx in tqdm(range(len(test))):
            que = test['question'][idx]
            number_test += 1
            top_passages = self.retriever(que, topk=topk, w_bm25=w_bm25, w_emb=w_emb)
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
    
    def evaluate_llm(self):
        print("-----------------------------------------------------------------")
        print("-----------------------Evaluate LLM------------------------------")
        print("-----------------------------------------------------------------")
        with open(self.eval_path, 'r', encoding='utf-8') as file:
            test = json.load(file)
        test = pd.DataFrame(test[:50])

        output_file_path = "results_llm_2.7B.csv"
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL'], use_stemmer=True)
        number_test = 0
        results = []

        with open(output_file_path, 'a', newline='\n', encoding='utf-8') as csv_file:
            fieldnames = ["answer", "label_answer", "time", "rouge1", "rouge2", "rougeL"] 
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not os.path.isfile(output_file_path):
                writer.writeheader()

            for idx in tqdm(range(len(test))):
                que = test['question'][idx]
                ans = test['label_answer'][idx]
                context = test['chunk'][idx]
                number_test += 1
                answer, finish_reason, t = self.generator(question=que, context=context)
                scores = scorer.score(ans, answer)
                results.append({"answer": answer,
                                "label_answer": ans, 
                                "time": t,
                                'rouge1': scores['rouge1'][2],
                                'rouge2': scores['rouge2'][2],
                                'rougeL': scores['rougeL'][2]})

                writer.writerow({"answer": answer, "label_answer": ans, "time": t, 'rouge1': scores['rouge1'][2], 'rouge2': scores['rouge2'][2], 'rougeL': scores['rougeL'][2]})

        return results
            
    def __call__(self, question, topk=1, w_bm25=0.3, w_emb=0.7, temperature=0, use_full_context=True):
        top_passages = self.retriever(question=question, topk=topk, w_bm25=w_bm25, w_emb=w_emb)
        context = top_passages[0]['chunk'] if use_full_context == False else top_passages[0]['full_passage']
        answer, finish_reason, t = self.generator(question=question, context=context, temperature=temperature)
        return answer, finish_reason, t

if __name__ == "__main__":
    rag = RAGModel()
    # rag.evaluate_retriever()
    # rag.evaluate_llm().
    while True:
        question = input("Question: ")
        answer, finish_reason, t = rag(question=question)
        print(f"User: {question}.\nSystem: {answer}\nTime: {t}")
        