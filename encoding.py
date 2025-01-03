import json as js
import torch
import os
import pickle as pk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse

def remove_xml(string):
    """
    去除xml标签及其内容 
    例如：<a> bcd </a> -> ""
    """
    import re
    pattern = re.compile(r'<[^>]+>', re.S)
    return pattern.sub('', string)

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="dataset/pid_to_info_all.json")
parser.add_argument("--save_path", type = str, default = "dataset/roberta_embeddings.pkl")
args = parser.parse_args()

with open(args.path, "r", encoding="utf-8") as f:
    papers = js.load(f)

batch_size = 5000
device = torch.device("cuda:0")

# Initialize RoBERTa tokenizer and model
model_path = './roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
dic_paper_embedding = {}
papers = [[key, value] for key,value in papers.items()]
for ii in tqdm(range(0, len(papers), batch_size), total=len(papers)//batch_size):
    
    batch_papers = papers[ii: ii + batch_size]
    texts = [paper[1]["title"] + paper[1]["abstract"] for paper in batch_papers]
    tmp = []
    for text in texts:
        text = remove_xml(text)
        tmp.append(text)
    texts = tmp
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=50,add_special_tokens=True)

    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    # take mean pooling of all tokens as title embedding
    embedding = outputs.last_hidden_state[:,:-1,:]
    attention_mask = inputs["attention_mask"][:,1:]
    embedding  = (embedding * attention_mask.unsqueeze(-1)).sum(dim = 1)/attention_mask.sum(dim = 1).unsqueeze(-1)
    assert torch.any(torch.isnan(embedding)) == False
    embedding = embedding.cpu().numpy()
    
    # or take cls token
    # embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  

    tt = 0
    for jj in range(ii, ii+len(batch_papers)):
        paper_id = papers[jj][0]
        paper_vec = embedding[tt]
        tt+=1
        dic_paper_embedding[paper_id] = paper_vec

with open(args.save_path, "wb") as f:
    pk.dump(dic_paper_embedding, f)
