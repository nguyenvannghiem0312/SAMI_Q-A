import os
import json
from tqdm import tqdm
import sys

class CorpusProcessor:
    def __init__(self, data_dir="documents/", output_file="documents_chunk/", chunk_size=None, window_size=None):
        self.data_dir = data_dir
        self.output_dir = output_file
        self.corpus = []
        self.meta_corpus = []
        self.chunk_size = chunk_size
        self.window_size = window_size

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
        with open(output_file, "w+", encoding="utf8") as outfile:
            for chunk in self.meta_corpus:
                d = json.dumps(chunk, ensure_ascii=False) + "\n"
                outfile.write(d)


if __name__ == "__main__":
    if len(sys.argv) < 3 or (len(sys.argv) > 3 and len(sys.argv) < 5):
        print("Usage: python script.py <data_dir> <output_file> <chunk_size> <window_size>")
        print("Or usage: python script.py <data_dir> <output_file>")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    output_file = sys.argv[2]
    chunk_size = window_size = None
    if len(sys.argv) == 5:
        chunk_size = sys.argv[3]
        window_size = sys.argv[4]

    corpus_processor = CorpusProcessor(data_dir, output_file, chunk_size, window_size)
    corpus_processor.get_corpus()
    corpus_processor.save_corpus_to_files()
