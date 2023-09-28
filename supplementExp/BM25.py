"""
Apply BM25 as baseline for tool selection
"""

from rank_bm25 import BM25Okapi
import numpy as np
import sys

sys.path.append('../')
from gear.prompt import calculator_description, wiki_description
from gear.read_dataset import *

tool_description = [calculator_description, wiki_description]
tokenized_corpus = [doc.split(" ") for doc in tool_description]

bm25 = BM25Okapi(tokenized_corpus)


path = "../datasets/ASDiv/ASDiv.xml"
inputs, _, _ = read_ASDiv(path)

len = len(inputs)
count = 0
for input in inputs:
    tokenized_input = input.split(" ")
    scores = bm25.get_scores(tokenized_input)
    label = np.argmax(scores)
    print(scores)
    if label == 0:
        count += 1
    
