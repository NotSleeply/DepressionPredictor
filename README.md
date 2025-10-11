# DepressionPredictor

本项目旨在基于多模态数据（如文本、音频等）进行抑郁倾向预测，结合自然语言处理、机器学习与深度学习等方法，探索社交媒体数据中的心理健康信号。

## 项目特色

- **多模态特征融合**：支持文本（如推文）、音频等多种数据源的特征提取与融合。
- **多模型对比**：集成传统机器学习（如SVM、随机森林）与深度学习（如BERT、Transformer、神经网络）模型，进行效果对比。
- **可解释性分析**：集成SHAP等工具，分析模型决策的关键特征。
- **可视化丰富**：输出多种可视化图表，辅助理解模型表现与数据分布。

## 快速启动

```shell
  uv venv .venv     // 创建虚拟环境
  uv activate venv  // 激活虚拟环境
  uv sync           // 安装依赖
  uv run start      // 运行main.py
```

## 参考资料

- [Mental-Health-Twitter.csv](./data/Mental-Health-Twitter.csv) : 来源为[Kaggle](https://www.kaggle.com/)公开数据集的[Depression: Twitter Dataset + Feature Extraction](https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media?resource=download)，包含推文文本及其抑郁标签。

- [暗知识：机器认知如何颠覆商业和社会](https://weread.qq.com/web/bookDetail/faa32030717f58b3faa40a7) ： 理解机器学习是什么的入门书籍。

## 成就

- 获取[全国大学生统计建模大赛](http://tjjmds.ai-learning.net)国赛资格(2025年)

## 许可证

- 本项目仅供学术研究与学习使用，禁止用于商业用途；
- 采用 MIT 许可证，详情请查看 [LICENSE](LICENSE) 文件。
