import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# 初始化
model_path = './roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')

def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        # 使用 [CLS] 标记的嵌入或取平均池化
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def compute_similarity(sentence1, sentence2):
    emb1 = get_embedding(sentence1)
    emb2 = get_embedding(sentence2)
    similarity = F.cosine_similarity(emb1, emb2)
    return similarity.item()

# 示例
sentence_a = "我是学生。"
sentence_b = "我在上学。"

similarity_score = compute_similarity(sentence_a, sentence_b)
print(f"相似度得分: {similarity_score:.4f}")