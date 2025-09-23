# 文本模态预处理与可视化分析代码
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 下载 NLTK 资源
nltk.download('stopwords')

# 加载数据集
data = pd.read_csv("./Mental-Health-Twitter.csv")

# 在其他导入语句之后，绘图之前添加以下字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题# 文本标准化处理

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return " ".join(tokens)

data['clean_text'] = data['post_text'].apply(preprocess_text)

# 情感分析 (VADER)
sia = SentimentIntensityAnalyzer()
data['sentiment'] = data['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# TF-IDF 特征提取
tfidf = TfidfVectorizer(max_features=5000)
tfidf_features = tfidf.fit_transform(data['clean_text'])

# BERT 特征提取
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 检查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 将模型移至 GPU (如果可用)
model = model.to(device)

# 批处理以提高效率
batch_size = 32  # 可根据您的 GPU 内存调整
bert_features = []

for i in range(0, len(data), batch_size):
    batch_texts = data['clean_text'].iloc[i:i+batch_size].tolist()
    
    # 显示进度
    if i % (batch_size * 10) == 0:
        print(f"处理 BERT 特征: {i}/{len(data)}")
    
    # 批量处理输入
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                      truncation=True, max_length=512)
    
    # 将输入移至同一设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取 CLS token 嵌入
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    bert_features.extend(cls_embeddings)

bert_features = np.array(bert_features)
print(f"BERT 特征提取完成，形状: {bert_features.shape}")

# 可以将 BERT 特征保存为文件，避免重复计算
np.save("bert_features.npy", bert_features)

# 可视化：情感分数分布
plt.figure(figsize=(8, 4))
sns.histplot(data['sentiment'], bins=30, kde=True)
plt.title("VADER 情感得分分布")
plt.xlabel("情感得分")
plt.ylabel("频率")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.close()

# 可视化：情感得分与抑郁标签关系
plt.figure(figsize=(6, 4))
sns.boxplot(x='label', y='sentiment', data=data)
plt.title("情感得分 vs 抑郁标签")
plt.xlabel("是否抑郁 (0=否, 1=是)")
plt.ylabel("情感得分")
plt.tight_layout()
plt.savefig("sentiment_vs_label.png")
plt.close()

# TF-IDF 可视化（PCA）
pca = PCA(n_components=2)
tfidf_reduced = pca.fit_transform(tfidf_features.toarray())
plt.figure(figsize=(6, 4))
sns.scatterplot(x=tfidf_reduced[:, 0], y=tfidf_reduced[:, 1], hue=data['label'][:len(tfidf_reduced)])
plt.title("TF-IDF 降维可视化 (PCA)")
plt.tight_layout()
plt.savefig("tfidf_pca.png")
plt.close()

print("文本处理、特征提取与可视化完成！")