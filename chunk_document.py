import os
import json
from typing import Any
from tqdm import tqdm
import sys
import stat
import numpy as np
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
import torch
import pickle

class CorpusProcessor:
    def __init__(self, data_dir: str="documents/", 
                 output_dir: str="documents_chunk/documents_chunk.json", 
                 chunk_size=None, 
                 window_size=None,
                 model_embedding=None,
                 output_embedding_dir: str="documents_embedding/documents_embedding.pkl"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.corpus = []
        self.meta_corpus = []
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.model_embedding = model_embedding
        self.output_embedding_dir = output_embedding_dir

    def split_text_into_chunks(self, text):
        words = text.split()
        num_words = len(words)
        chunks = []
        start_idx = 0

        while True:
            end_idx = start_idx + self.chunk_size
            chunk = " ".join(words[start_idx:end_idx])
            chunks.append(chunk)
            if end_idx >= num_words:
                break
            start_idx += self.window_size

        return chunks

    def get_corpus(self):
        filenames = os.listdir(self.data_dir)
        filenames = sorted(filenames)
        # print(filenames)
        _id = 0
        for filename in tqdm(filenames):
            filepath = self.data_dir + filename
            title = filename.strip(".txt")
            with open(filepath, "r", encoding="utf8") as f:
                text = f.read()
                paragraph = text
                text = text.lstrip(title).strip()

                if self.chunk_size != None and self.window_size != None:
                    chunks = self.split_text_into_chunks(text)
                else: chunks = text.split('\n\n')
                
                chunks = [f"Title: {title}\n\n{chunk}" for chunk in chunks]
                meta_chunks = [{
                    "title": title,
                    "full_passage": paragraph,
                    "chunk": chunks[i],
                    "full_passage_id": _id,
                    "chunk_id": _id + i,
                    "len": len(chunks[i].split())
                } for i in range(len(chunks))]
                _id += len(chunks)
                self.meta_corpus.extend(meta_chunks)
                self.corpus.extend(chunks)

    def save_corpus_to_files(self):
        with open(self.output_dir, "w+", encoding="utf8") as outfile:
            for chunk in self.meta_corpus:
                d = json.dumps(chunk, ensure_ascii=False) + "\n"
                outfile.write(d)

    def embedding(self):
        model = SentenceTransformer(self.model_embedding)
        if torch.cuda.is_available() == True:
           model.cuda()
        segmented_corpus = [tokenize(example["chunk"]) for example in tqdm(self.meta_corpus)]
        segmented_corpus = [example["chunk"] for example in tqdm(self.meta_corpus)]
        embeddings = model.encode(segmented_corpus)
        embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]


        with open(self.output_embedding_dir, 'wb+') as f:
            pickle.dump(embeddings, f)    

    def fit(self):
        print("-----------------------------------------------------------------")
        print("-------------------Read the corpus and chunk--------------------")
        print("-----------------------------------------------------------------")
        self.get_corpus()
        print(f"---> Read the corpus succesfull, corpus has {len(self.meta_corpus)} chunks. Now save the corpus.")
        self.save_corpus_to_files()
        print("---> Save corpus sucessfull.")

        if self.model_embedding != None:
            print("---------------------------------------------------------------------")
            print("---------Now, loading the embedding model and text embedding---------")
            print("---------------------------------------------------------------------")
            self.embedding()
            print("---> Text embedding sucessfull.")
        else:
            print("Warning: You have no embeding model.")


if __name__ == "__main__":
    # if len(sys.argv) < 3 or (len(sys.argv) > 3 and len(sys.argv) < 5):
    #     print("Usage: python script.py <data_dir> <output_file> <chunk_size> <window_size> <model_embedding> <output_embedding>")
    #     print("Or usage: python script.py <data_dir> <output_file>")
    #     sys.exit(1)
        
    # data_dir = sys.argv[1]
    # output_file = sys.argv[2]
    # chunk_size = window_size = None
    # if len(sys.argv) == 5:
    #     chunk_size = sys.argv[3]
    #     window_size = sys.argv[4]

    corpus_processor = CorpusProcessor(data_dir='documents/', 
                                       output_dir='documents_chunk/documents_chunk.json',
                                       chunk_size=None,
                                       window_size=None, 
                                       model_embedding='model/bi-encoder-2epochs',
                                       output_embedding_dir='documents_chunk/documents_embedding.pkl')
    corpus_processor.fit()
