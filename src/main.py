# 多模态抑郁倾向预测完整训练脚本（可本地运行，支持GPU）

from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel
from imblearn.over_sampling import SMOTE
import re
import shap
import emoji
import matplotlib
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

nltk.download('vader_lexicon')
print("PyTorch 版本:", torch.__version__)

# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


# 在其他导入语句之后，绘图之前添加以下字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei',
                                          'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题# 文本标准化处理

# ========== 数据准备 ==========
print("开始数据准备...")

# 数据加载
DATA_PATH = "../data/Mental-Health-Twitter.csv"
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['post_text'])
print(f"数据加载完成，共 {len(df)} 条记录")


def clean_text(text):
    ''' 清洗文本数据 '''
    text = str(text)  # 确保是字符串
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"[^A-Za-z0-9 ]+", ' ', text)
    return text.lower()


df['clean_text'] = df['post_text'].apply(clean_text)
print("文本清洗完成")

# 情感得分（VADER）
print("进行情感分析...")
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['clean_text'].apply(
    lambda x: sia.polarity_scores(x)['compound'])

# 表情情绪比例（正/中/负）
print("分析表情符号...")


def emoji_sentiment_ratio(text):
    pos, neg, neu = 0, 0, 0
    for ch in text:
        if ch in emoji.EMOJI_DATA:
            desc = emoji.EMOJI_DATA[ch]['en']
            if any(word in desc for word in ['sad', 'cry', 'angry', 'frown']):
                neg += 1
            elif any(word in desc for word in ['smile', 'happy', 'joy']):
                pos += 1
            else:
                neu += 1
    total = pos + neg + neu if pos + neg + neu > 0 else 1
    return [pos/total, neu/total, neg/total]


df[['pos_emoji', 'neu_emoji', 'neg_emoji']] = df['post_text'].apply(
    emoji_sentiment_ratio).apply(pd.Series)

# 时间模态处理
print("处理时间特征...")


def extract_time_features(ts):
    try:
        # API 标准时间格式
        dt = datetime.strptime(ts, '%a %b %d %H:%M:%S %z %Y')
        is_night = 1 if dt.hour < 6 or dt.hour >= 22 else 0
        return [dt.hour/24, is_night]
    except ValueError:
        # 记录错误并返回默认值
        return [0.5, 0]  # 返回默认值


df[['hour_norm', 'is_night']] = df['post_created'].apply(
    extract_time_features).apply(pd.Series)

# 社交数值特征 + 标准化
print("标准化社交特征...")
social_features = ['followers', 'friends',
                   'favourites', 'statuses', 'retweets']
scaler = StandardScaler()
df[social_features] = scaler.fit_transform(df[social_features])

# TF-IDF 特征
print("提取TF-IDF特征...")
tfidf = TfidfVectorizer(max_features=500)
tfidf_matrix = tfidf.fit_transform(df['clean_text']).toarray()
print(f"TF-IDF特征形状: {tfidf_matrix.shape}")

# 启用 BERT 表征
print("开始加载BERT模型...")
# 检查是否存在保存的BERT特征
bert_features_file = "bert_features_full.npy"
if os.path.exists(bert_features_file):
    print(f"加载已保存的BERT特征: {bert_features_file}")
    bert_features = np.load(bert_features_file)
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.to(DEVICE)
    bert_model.eval()  # 设置为评估模式

    # 批处理 BERT 特征提取
    batch_size = 32
    bert_features = []

    print("开始提取BERT特征...")
    for i in range(0, len(df), batch_size):
        if i % (batch_size * 10) == 0:
            print(f"处理BERT特征: {i}/{len(df)}")

        batch_texts = df['clean_text'].iloc[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           return_tensors="pt", max_length=128)

        # 将输入移至设备
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bert_model(**inputs)

        # 获取 CLS token 嵌入
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        bert_features.extend(cls_embeddings)

    bert_features = np.array(bert_features)
    print(f"BERT特征提取完成，形状: {bert_features.shape}")

    # 保存 BERT 特征以便重复使用
    np.save(bert_features_file, bert_features)

# 标签
y = df['label'].values

# 拼接所有特征
print("拼接所有特征...")
X_all = np.concatenate([
    tfidf_matrix,  # TF-IDF特征
    df[['sentiment']].values,  # VADER情感得分
    df[['hour_norm', 'is_night']].values,  # 时间特征
    df[['pos_emoji', 'neu_emoji', 'neg_emoji']].values,  # 表情特征
    df[social_features].values,  # 社交特征
    bert_features  # BERT特征
], axis=1)

print(f"总特征维度: {X_all.shape}")

# SMOTE 过采样
print("进行SMOTE过采样...")
X_resampled, y_resampled = SMOTE().fit_resample(X_all, y)
print(f"过采样后样本分布: {np.bincount(y_resampled.astype(int))}")

# 数据划分 - 添加验证集
print("划分数据集...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print(f"训练集: {X_train.shape[0]}个样本")
print(f"验证集: {X_val.shape[0]}个样本")
print(f"测试集: {X_test.shape[0]}个样本")


class DepressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = DepressionDataset(X_train, y_train)
val_dataset = DepressionDataset(X_val, y_val)
test_dataset = DepressionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ========== 模型定义 ==========


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


model = Classifier(X_train.shape[1]).to(DEVICE)
print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")

# ========== 训练过程 ==========
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=3, factor=0.5, verbose=True)

# 记录训练历史
history = {
    'train_loss': [],
    'val_loss': [],
    'val_auc': []
}

EPOCHS = 30
best_auc = 0
best_model_state = None
patience = 5
no_improve = 0

print("开始训练模型...")
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)

    # 验证阶段
    model.eval()
    val_preds, val_targets = [], []
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item()
            val_preds.extend(out.cpu().numpy())
            val_targets.extend(yb.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_auc = roc_auc_score(val_targets, val_preds)
    history['val_loss'].append(avg_val_loss)
    history['val_auc'].append(val_auc)

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # 更新学习率
    scheduler.step(avg_val_loss)

    # 早停检查
    if val_auc > best_auc:
        best_auc = val_auc
        best_model_state = model.state_dict().copy()
        no_improve = 0
        print(f"✓ 模型改善! 最佳验证AUC: {best_auc:.4f}")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"早停: {patience} 轮未改善")
        break

# 使用最佳模型进行测试
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"加载最佳模型 (验证AUC: {best_auc:.4f})")

# ========== 测试评估 ==========
print("\n开始测试评估...")
model.eval()
preds, targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        out = model(xb).cpu().numpy()
        preds.extend(out)
        targets.extend(yb.numpy())

pred_labels = [1 if p > 0.5 else 0 for p in preds]
accuracy = accuracy_score(targets, pred_labels)
auc = roc_auc_score(targets, preds)
print(f"\n测试结果:")
print(f"准确率: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"\n分类报告:")
print(classification_report(targets, pred_labels))
print(f"混淆矩阵:")
print(confusion_matrix(targets, pred_labels))

# ========== 修改后的模型可解释性分析部分 ==========
print("\n开始模型可解释性分析（使用全部测试数据）...")

try:
    # 使用全部测试数据进行SHAP分析
    model.eval()  # 确保模型处于评估模式

    print(f"SHAP分析使用测试样本: {len(X_test)}")

    # 拼接特征名（确保顺序与原始特征顺序一致）
    feature_names = [f"TFIDF_{i}" for i in range(tfidf_matrix.shape[1])] + [
        "sentiment", "hour_norm", "is_night", "pos_emoji", "neu_emoji", "neg_emoji"
    ] + social_features + [f"BERT_{i}" for i in range(bert_features.shape[1])]

    # 确保特征名称长度与特征数量一致
    assert len(
        feature_names) == X_all.shape[1], f"特征名称数量 {len(feature_names)} 与特征数量 {X_all.shape[1]} 不匹配"

    # 改用 GradientExplainer 而不是 DeepExplainer
    # 准备一个小批量数据作为背景
    background = torch.tensor(X_train[:100], dtype=torch.float32).to(DEVICE)

    # 创建一个包装类，使模型输出单个标量值
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

    wrapped_model = ModelWrapper(model).to(DEVICE)

    # 使用 Kernel SHAP 而不是 Deep SHAP (性能稍慢但更稳定)
    # 取一小部分数据进行演示
    sample_size = min(500, len(X_test))  # 最多使用500个样本用于展示
    X_sample = X_test[:sample_size]

    # 创建 KernelExplainer (更稳定但计算成本更高)
    # 定义一个预测函数
    def f(x):
        with torch.no_grad():
            tensor_x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
            return model(tensor_x).cpu().numpy()

    # 使用一小部分背景数据
    explainer = shap.KernelExplainer(f, shap.kmeans(X_train, 50))

    print("开始计算SHAP值...")
    # 计算SHAP值 (这可能需要一些时间)
    shap_values = explainer.shap_values(X_sample)

    # 保存SHAP值
    np.save("shap_values.npy", shap_values)

    print(f"SHAP分析完成，形状: {np.shape(shap_values)}")

    # SHAP 值总结图
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    plt.title("特征重要性 SHAP 值分布")
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 特征重要性条形图
    plt.figure(figsize=(14, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20
    )
    plt.title("Top 20 重要特征")
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("已保存 SHAP 图：shap_summary_plot.png 和 shap_feature_importance.png")

    # 获取并打印前20个最重要特征
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(-mean_abs_shap)
    print("\n模型最重要的20个特征:")
    for i, idx in enumerate(top_indices[:20]):
        print(f"{i+1}. {feature_names[idx]} (重要性: {mean_abs_shap[idx]:.6f})")

    # 根据模态分组计算特征重要性
    feature_groups = {
        'TF-IDF': (0, tfidf_matrix.shape[1]),
        'VADER情感': (tfidf_matrix.shape[1], tfidf_matrix.shape[1] + 1),
        '时间特征': (tfidf_matrix.shape[1] + 1, tfidf_matrix.shape[1] + 3),
        '表情符号': (tfidf_matrix.shape[1] + 3, tfidf_matrix.shape[1] + 6),
        '社交特征': (tfidf_matrix.shape[1] + 6, tfidf_matrix.shape[1] + 11),
        'BERT特征': (tfidf_matrix.shape[1] + 11, X_all.shape[1])
    }

    group_importance = {}
    for group_name, (start_idx, end_idx) in feature_groups.items():
        group_importance[group_name] = np.abs(
            shap_values[:, start_idx:end_idx]).mean()

    # 绘制模态重要性图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(group_importance.keys(), group_importance.values())
    plt.xlabel('特征模态')
    plt.ylabel('平均|SHAP|值')
    plt.title('各模态特征对模型预测的重要性')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 在每个柱状图顶部添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=9)

    plt.savefig('modality_importance.png', dpi=300)
    plt.close()

except Exception as e:
    import traceback
    print(f"SHAP分析出错: {str(e)}")
    print("错误详情:")
    traceback.print_exc()
    print("跳过SHAP分析，继续执行后续代码")
# ========== 可视化结果 ==========
# 训练历史可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('训练和验证损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_auc'], label='Val AUC')
plt.title('验证AUC')
plt.xlabel('轮次')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(targets, preds)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC 曲线')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png')
plt.close()

# 混淆矩阵可视化
plt.figure(figsize=(8, 6))
cm = confusion_matrix(targets, pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('confusion_matrix.png')
plt.close()

print(f"\n训练完成! 结果已保存为图表。")
print(f"最终测试AUC: {auc:.4f}")
