# 文本模态特征融合策略对比分析：TF-IDF vs BERT vs 融合
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, accuracy_score
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')

# 在其他导入语句之后，绘图之前添加以下字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题


# =================== 数据加载与清洗 ===================
data = pd.read_csv("./Mental-Health-Twitter.csv")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return " ".join(tokens)

data['clean_text'] = data['post_text'].apply(preprocess_text)

# 使用全部数据
texts = data['clean_text'].tolist()
labels = data['label'].values

# =================== 特征提取 ===================
# TF-IDF 特征
print("正在提取TF-IDF特征...")
tfidf = TfidfVectorizer(max_features=1000)
tfidf_features = tfidf.fit_transform(texts).toarray()
print(f"TF-IDF特征形状: {tfidf_features.shape}")

# BERT 特征提取
print("正在提取BERT特征...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 检查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
model = model.to(device)
model.eval()  # 设置为评估模式

# 批处理以提高效率
batch_size = 32  # 可根据GPU内存调整
bert_features = []

for i in range(0, len(texts), batch_size):
    # 显示进度
    if i % (batch_size * 10) == 0:
        print(f"处理 BERT 特征: {i}/{len(texts)}")
        
    # 获取当前批次的文本
    batch_texts = texts[i:i+batch_size]
    
    # 批量处理输入
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                      truncation=True, max_length=128)
    
    # 将输入移至同一设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 禁用梯度计算以加速处理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取 CLS token 嵌入（第一个token的最后隐藏状态）
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    bert_features.extend(cls_embeddings)

bert_features = np.array(bert_features)
print(f"BERT 特征提取完成，形状: {bert_features.shape}")

# 可以将特征保存为文件，避免重复计算
np.save("bert_features_full.npy", bert_features)
np.save("tfidf_features_full.npy", tfidf_features)

# 融合特征
print("正在融合特征...")
fused_features = np.hstack([tfidf_features, bert_features])
print(f"融合特征形状: {fused_features.shape}")

# =================== 修改后的模型训练函数 ===================
def train_and_eval(X, y, name_prefix):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"训练 {name_prefix} 模型...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # 报告
    print(f"\n{name_prefix} 分类报告:\n", classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name_prefix} AUC 值: {auc:.4f}")
    
    # 返回评估结果，而不是生成单独的图表
    return {
        'name': name_prefix,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'auc': auc,
        'accuracy': accuracy_score(y_test, y_pred)
    }

# =================== 三种模型训练对比 ===================
print("\n开始模型训练和评估...")
results = []
results.append(train_and_eval(tfidf_features, labels, "TF-IDF"))
results.append(train_and_eval(bert_features, labels, "BERT"))
results.append(train_and_eval(fused_features, labels, "融合模型"))

# =================== 绘制对比图 ===================
# 1. ROC曲线对比图
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
line_styles = ['-', '--', '-.']
for i, result in enumerate(results):
    fpr, tpr, _ = roc_curve(result['y_test'], result['y_prob'])
    plt.plot(fpr, tpr, color=colors[i], linestyle=line_styles[i],
             label=f"{result['name']} (AUC = {result['auc']:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("假阳性率 (False Positive Rate)")
plt.ylabel("真阳性率 (True Positive Rate)")
plt.title("三种模型 ROC 曲线对比")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("models_roc_comparison.png")
plt.close()

# 2. 混淆矩阵对比图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, result in enumerate(results):
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f"{result['name']} 混淆矩阵")
    axes[i].set_xlabel("预测")
    axes[i].set_ylabel("真实")

plt.tight_layout()
plt.savefig("models_confusion_matrix_comparison.png")
plt.close()

# 3. 性能指标对比条形图
metrics = {
    'Model': [r['name'] for r in results],
    'AUC': [r['auc'] for r in results],
    'Accuracy': [r['accuracy'] for r in results]
}

plt.figure(figsize=(10, 6))
x = np.arange(len(metrics['Model']))
width = 0.35

# 绘制条形图
plt.bar(x - width/2, metrics['AUC'], width, label='AUC', color='cornflowerblue')
plt.bar(x + width/2, metrics['Accuracy'], width, label='准确率', color='lightcoral')

# 添加文本标签
plt.xlabel('模型')
plt.ylabel('得分')
plt.title('模型性能指标对比')
plt.xticks(x, metrics['Model'])
plt.legend()
plt.ylim(0.5, 1.0)  # 设置y轴范围，突出差异
plt.grid(axis='y', alpha=0.3)

# 在条形上方添加数值标签
for i, v in enumerate(metrics['AUC']):
    plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
for i, v in enumerate(metrics['Accuracy']):
    plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig("models_performance_comparison.png")
plt.close()

print("TF-IDF、BERT 和融合模型对比分析完成，对比图表已生成。")