# 第3章 传统NLP任务

# 3.1 文本分类

## 3.1.1 文本分类

文本分类任务其实我们应该是相对比较熟悉的。毕竟学习机器学习的第一课就是做分类任务嘛，把文本分成若干类。它和聚类还有些不同的地方，这个我们后面会讲。那分类把文本能分成哪几类呢？比如，邮箱可以把邮件分成正常邮件和垃圾邮件；新闻会把文章分区为体育/娱乐/政治/经济/军事/科技等；情感分析可以把文本分解成积极情绪、中性情绪、负面情绪等……对于分类任务，它的应用是广泛的，并且标签是根据具体任务来确定。

在文本分类任务中，常用的评价指标包括**准确率（Accuracy）**、**精确率（Precision）**、**召回率（Recall）**、**F1值（F1-Score）**以及**混淆矩阵（Confusion Matrix）**等。以下是这些指标的定义及其计算公式：

1. **准确率（Accuracy）**
   准确率是分类任务中最直观的指标，表示模型预测正确的样本占总样本的比例。  
   **公式**：

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

- **TP（True Positive）**：正确预测为正类的样本数  
- **TN（True Negative）**：正确预测为负类的样本数  
- **FP（False Positive）**：错误预测为正类的样本数  
- **FN（False Negative）**：错误预测为负类的样本数  

**适用场景**：类别分布均衡时，准确率是一个有效的指标；但在类别不平衡时，准确率可能会失真。

2. **精确率（Precision）**
   精确率表示模型预测为正类的样本中，实际为正类的比例，衡量模型的预测准确性。  
   **公式**：

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

**适用场景**：关注减少误报（False Positive）时使用，例如垃圾邮件分类中，避免将正常邮件误判为垃圾邮件。

3. **召回率（Recall）**
   召回率表示实际为正类的样本中，被模型正确预测为正类的比例，衡量模型的覆盖能力。  
   **公式**：

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

**适用场景**：关注减少漏报（False Negative）时使用，例如疾病诊断中，避免将患病样本误判为健康。

4. **F1值（F1-Score）**
   F1值是精确率和召回率的调和平均值，用于平衡精确率和召回率。  
   **公式**：

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**适用场景**：当精确率和召回率需要同时考虑时，F1值是一个综合指标，特别适用于类别不平衡的数据集。

5. **混淆矩阵（Confusion Matrix）**
   混淆矩阵是一个表格，用于直观展示分类模型的预测结果与实际结果的对比情况。  
   **结构**：

|              | **实际正类** | **实际负类** |
| ------------ | ------------ | ------------ |
| **预测正类** | TP           | FP           |
| **预测负类** | FN           | TN           |

**适用场景**：用于分析分类错误的具体类型（如FP和FN），帮助改进模型。

6. **ROC曲线与AUC值**

- **ROC曲线（Receiver Operating Characteristic Curve）**：以假正率（FPR）为横轴，真正率（TPR）为纵轴绘制的曲线，用于评估模型在不同阈值下的表现。  

  - **假正率（FPR）**：  
    $$
    \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
    $$

  - **真正率（TPR）**：  
    $$
    \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \text{Recall}
    $$

- **AUC值（Area Under Curve）**：ROC曲线下的面积，AUC值越接近1，模型性能越好。  
  **适用场景**：适用于二分类任务，尤其是类别不平衡的情况。

7. **宏平均（Macro-Average）与微平均（Micro-Average）**
   在多分类任务中，精确率、召回率和F1值可以通过宏平均或微平均计算：

- **宏平均（Macro-Average）**：对每个类别的指标单独计算后取平均。  
  $$
  \text{Macro-Precision} = \frac{1}{N} \sum_{i=1}^N \text{Precision}_i
  $$

  $$
  \text{Macro-Recall} = \frac{1}{N} \sum_{i=1}^N \text{Recall}_i
  $$

  $$
  \text{Macro-F1} = \frac{1}{N} \sum_{i=1}^N \text{F1}_i
  $$

- **微平均（Micro-Average）**：将所有类别的TP、FP、FN汇总后计算全局指标。  
  $$
  \text{Micro-Precision} = \frac{\sum_{i=1}^N \text{TP}_i}{\sum_{i=1}^N (\text{TP}_i + \text{FP}_i)}
  $$

  $$
  \text{Micro-Recall} = \frac{\sum_{i=1}^N \text{TP}_i}{\sum_{i=1}^N (\text{TP}_i + \text{FN}_i)}
  $$

  $$
  \text{Micro-F1} = 2 \times \frac{\text{Micro-Precision} \times \text{Micro-Recall}}{\text{Micro-Precision} + \text{Micro-Recall}}
  $$

**适用场景**：  

- 宏平均：适用于类别重要性相同的情况。  
- 微平均：适用于类别不平衡的情况，更关注整体性能。

8. **Kappa系数（Cohen's Kappa）**
   Kappa系数用于衡量分类结果与随机分类的一致性，考虑了随机因素的影响。  
   **公式**：

$$
\text{Kappa} = \frac{\text{Accuracy} - \text{Random Accuracy}}{1 - \text{Random Accuracy}}
$$

其中，\(\text{Random Accuracy}\) 是随机分类的准确率。  
**适用场景**：适用于评估分类器的实际性能是否优于随机分类。



早期的一些文本分类是通过机器学习来进行。比如SVM、朴素贝叶斯等，经常被用于文本分类任务当中来。而传统机器学习需要一些特征，这可以通过高频词的TF-IDF值来刻画。TF-IDF本身可以用来衡量一个词在文本中的重要性。TF-IDF（词频-逆文档频率）是衡量词语在文档集合中重要程度的经典指标，其核心思想在于：一个词语的重要性与其在单个文档中的出现频率成正比，与在整个语料库中的出现频率成反比。数学表达式为：

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) = \left( \frac{\text{词t在文档d中出现的次数}}{\text{文档d的总词数}} \right) \times \log\left( \frac{\text{总文档数}}{\text{包含词t的文档数 + 1} } \right)
$$

该指标有效抑制了"的"、"是"等高频无意义词的权重，同时强化了如"发票"、"促销"等特征词的判别力。通过以下代码可快速计算TF-IDF矩阵：

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例文档集
documents = [
    "优惠促销 限时折扣 点击领取",
    "财务报表 年度审计 税务核算",
    "垃圾邮件 虚假链接 账号安全"
]

# 自动计算TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
print("特征词列表:", vectorizer.get_feature_names_out())
print("TF-IDF矩阵:\n", tfidf_matrix.toarray())

# 手动实现TF计算
from collections import defaultdict
import math

def compute_tf(text):
    words = text.split()
    tf_dict = defaultdict(float)
    for word in words:
        tf_dict[word] += 1
    for word in tf_dict:
        tf_dict[word] /= len(words)
    return tf_dict

# 手动实现IDF计算
def compute_idf(docs):
    N = len(docs)
    idf_dict = defaultdict(float)
    for doc in docs:
        words = set(doc.split())
        for word in words:
            idf_dict[word] += 1
    for word in idf_dict:
        idf_dict[word] = math.log(N / (idf_dict[word] + 1))
    return idf_dict
```

朴素贝叶斯算法基于特征条件独立假设，特别适合处理高维稀疏的文本数据。以下代码演示从数据预处理到模型评估的完整流程：

```python
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 示例数据集（实际需替换为真实数据）
data = {
    "text": [
        "免费领取iPhone点击链接", 
        "财务部季度报表请查收",
        "限时特惠最后一天", 
        "项目评审会议通知"
    ],
    "label": [1, 0, 1, 0]  # 1:垃圾邮件 0:正常邮件
}
df = pd.DataFrame(data)

# 特征提取
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 训练模型
model = MultinomialNB(alpha=0.1)  # 拉普拉斯平滑
model.fit(X_train, y_train)

# 预测评估
y_pred = model.predict(X_test)
print("测试准确率:", accuracy_score(y_test, y_pred))

# 新样本预测
new_emails = ["紧急通知：账号异常登录", "市场部预算表已提交"]
new_X = vectorizer.transform(new_emails)
predictions = model.predict(new_X)
print("预测结果:", ["垃圾邮件" if p ==1 else "正常邮件" for p in predictions])
```

随着深度学习的发展，深度学习技术也被广泛应用于文本分类当中来。早期用于文本分类的深度学习模型主要是基于循环神经网络（RNN）及其变体，如长短期记忆网络（LSTM）和门控循环单元（GRU），它们能够有效处理文本的序列信息，但在处理长文本时存在一定的局限性（点名TextRNN模型）。随后，卷积神经网络（CNN）也被应用于文本分类，它可以通过卷积操作提取文本的局部特征，并且在一些文本分类任务中取得了较好的效果。例如，TextCNN使用一维卷积的方法学习词向量之间的特征并卷积，形成文本级别的语义表示。CNN的特点是快速，RNN的特点是序列学习，但在面对篇章级文本分类这种长文本任务时，再好的RNN记忆力上总会表现得捉襟见肘，因为一篇文章真正的主题词、关键词分散在不同位置，需要分配注意力机制。近年来，随着Transformer架构的出现，基于注意力的模型如HAN、BERT、GPT等在文本分类任务中展现出了强大的性能，这些模型通过大规模无监督预训练学习到了丰富的语言知识和语义信息，然后在特定的文本分类任务上进行微调，能够取得显著优于传统模型的分类效果，推动了文本分类技术的进一步发展。

一个非常典型的案例是TextCNN。TextCNN的设计思想源于将卷积神经网络（CNN）应用于文本分类任务，其核心在于利用CNN强大的特征提取能力来捕捉文本中的局部特征。具体来说，TextCNN通过将文本转化为词向量序列，然后使用多个不同尺寸的卷积核在这些词向量上进行滑动窗口操作，从而提取出不同粒度的n-gram特征。这些卷积核能够捕获文本中的局部相关性，例如2-gram、3-gram等，进而生成多个特征图。随后，通过最大池化操作对每个特征图进行降维，保留最显著的特征信息，最终将这些特征拼接起来，形成一个固定长度的特征向量，用于后续的分类任务。

在作用过程中，TextCNN首先将输入的文本进行词向量转换，得到一个二维的词向量矩阵。接着，这个矩阵被输入到卷积层，卷积层中的多个卷积核会并行地在词向量矩阵上滑动，提取出不同位置和粒度的局部特征。每个卷积核都会生成一个特征图，这些特征图经过激活函数处理后，再通过最大池化层提取每个特征图中的最大值，得到一个固定长度的特征向量。最后，这个特征向量被输入到全连接层，通过softmax函数进行分类，输出文本所属类别的概率分布。
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215155242267-877287404.png)
如果写代码，可以这样写：

1. **数据加载与预处理**：

```python
import pandas as pd
import jieba
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

# 加载数据
data = pd.read_csv('dataset.csv')  # 假设数据保存为CSV文件

# 分词函数
def tokenize(text):
    return list(jieba.cut(text))

# 构建词汇表
def build_vocab(texts, min_freq=5):
    counter = Counter()
    for text in texts:
        words = tokenize(text)
        counter.update(words)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# 将文本转换为词索引序列
def text_to_sequence(text, vocab, max_len=100):
    words = tokenize(text)
    seq = [vocab.get(word, vocab['<UNK>']) for word in words]
    seq = seq[:max_len] + [vocab['<PAD>']] * (max_len - len(seq))
    return seq

# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        seq = text_to_sequence(text, self.vocab, self.max_len)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 示例数据
texts = data['原微博内容'].tolist()
labels = data['原文观点'].map({'无关': 0, '支持': 1, '反对': 2, '中立': 3}).tolist()

vocab = build_vocab(texts)
dataset = TextDataset(texts, labels, vocab)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

2. **模型定义**:

```python
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_channels, (k, embed_dim)) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_channels, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        
        # Apply convolution and max pooling
        conv_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x)).squeeze(3)  # [batch_size, num_channels, seq_len - k + 1]
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [batch_size, num_channels]
            conv_outs.append(conv_out)
        
        # Concatenate all convolution outputs
        x = torch.cat(conv_outs, dim=1)  # [batch_size, len(kernel_sizes) * num_channels]
        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, num_classes]
        return x
    
# 超参数设置
vocab_size = len(vocab)
embed_dim = 512
num_classes = 4  # 替换成数据集当中标签种类个数
kernel_sizes = [3, 4, 5]
num_channels = 100
learning_rate = 0.001
num_epochs = 10

# 初始化模型、损失函数和优化器
model = TextCNN(vocab_size, embed_dim, num_classes, kernel_sizes, num_channels)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

3. **训练与验证**：

```python
from sklearn.model_selection import train_test_split

# 假设数据已经加载到 texts 和 labels 中
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 构建训练集和验证集
train_dataset = TextDataset(train_texts, train_labels, vocab)
val_dataset = TextDataset(val_texts, val_labels, vocab)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def calculate_accuracy(model, dataloader):
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# 初始化模型、损失函数和优化器
model = TextCNN(vocab_size, embed_dim, num_classes, kernel_sizes, num_channels)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()  # 将模型设置为训练模式
    total_loss = 0

    for batch in train_dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_dataloader)
    train_accuracy = calculate_accuracy(model, train_dataloader)
    val_accuracy = calculate_accuracy(model, val_dataloader)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, '
          f'Train Acc: {train_accuracy:.4f}, '
          f'Val Acc: {val_accuracy:.4f}')

print("训练完成！")
```

预训练技术得到广泛发展以后，文本分类几乎被一统了。以BERT为例，它是一种基于Transformer架构的预训练语言模型，在文本分类任务中，通常先将预训练好的BERT模型进行微调。具体来说，首先将待分类的文本输入到BERT模型中，BERT会输出文本中每个单词的隐藏状态表示，然后取[CLS]标记对应的隐藏状态作为整个文本的表示，接着在这个表示的基础上添加一个全连接层作为分类器，通过在文本分类任务的训练数据上训练这个全连接层以及微调BERT模型的部分参数，使得模型能够学习到针对特定文本分类任务的特征表示，从而实现对文本的准确分类。比如，如果用BERT实现一个文本分类可以这么写：

```python
import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch

# 1. 加载数据
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# 2. 数据预处理
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['content']
        labels = [1.0] * 15  # 根据实际情况设置每个主题的标签

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# 3. 模型训练
def train_model(model, train_dataloader, epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = BCEWithLogitsLoss()  # 使用多标签分类的损失函数

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

# 4. 主程序
def main():
    # 设置文件夹路径和其他参数
    folder_path = 'F:\\LDA\\dd'  # 更改为您的文件夹路径
    max_length = 128
    epochs = 3
    learning_rate = 1e-5
    num_labels = 15  # 15个主题类别，根据实际情况调整

    # 加载模型和分词器
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 遍历文件夹下的所有Excel文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            file_full_path = os.path.join(folder_path, file_name)

            # 加载数据
            data = load_data(file_full_path)

            # 创建数据集和数据加载器
            dataset = CustomDataset(data, tokenizer, max_length)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

            # 训练模型
            train_model(model, dataloader, epochs, learning_rate)

if __name__ == "__main__":
    main()

```

当然了，现在大语言模型当然也能做文本分类。我们可以在deepseek-r1网页版中试探一下它的效果，关于本地部署、API调用等内容我会在后续章节再提一下。我们编写提示词做文本分类：

```
user-prompt:
**角色**：假如你是一个高性能高准确率文本分析模型，现在在做一个文本的情感分类任务。
**任务**：你将会对给定的社交媒体篇章级文本进行分析，并分析它的情感积极性。这是一个分类任务，你只需要输出分类的预测标签和置信度即可。
**输出格式定义**：
    - **类别标签**：分类结果仅允许在[积极，消极，中性]三种当中进行选择，你只需从这三个当中选择你认为最符合的分类结果。
    - **置信度**：你给出这个结果的可信度，用0-1之间的实数表示，数值越高表示你的结果越可信（或者说你有多信任你的结果）
**注意**：你只需要返回预测标签和置信度，不要输出其他结果。
> **输入文本**：【#被家暴16次女子多次逃跑与报警均以失败告终#】#2年被家暴16次女子幸存后的580天#  澎湃新闻持续跟拍自述2年被家暴16次的玉梅，记录她幸存后的580个日夜。2024年1月，检方以故意伤害罪、虐待罪起诉其前夫贺某阳，目前该案正待一审开庭。玉梅说，除第一次外，后续每次报警，她都坚持要求依法处理，但得到的答复不尽相同——你们是夫妻，夫妻和外人不一样；你们都有孩子了，要为孩子考虑。仿佛应验了贺某阳曾对她说过的话：一个女人，领了结婚证、怀孕了，就像放风筝一样，随时可以拽回来。她想不通：“难道一张结婚证就可以把我和他牢牢绑定在一起吗？”为了离开施暴环境，她选择逃跑。2023年9月26日，贺某阳离开派出所后，玉梅没有回家。她坐上了去重庆的高铁，前往朋友家中躲避。过了不到一周，贺某阳找到了她。“我朋友一开门，他直接一下子冲进来了，”玉梅回忆起当时的场景，“他把我朋友家从上到下都翻了一遍，之后就让我们两个坐在沙发上，他就坐在餐桌旁边，把我手机也抢走了。”趁贺某阳不注意时，朋友才设法联系外界帮忙报警。在重庆辗转数日后，玉梅决定回到成都，“当时想着，他一直在重庆找我，可能想不到我会回去，最危险的地方就是最安全的地方。”出于谨慎，她打了一辆顺风车，用现金支付了车费。她说，自己换掉了手机号，借用他人身份证租住在成华区一栋公寓，距离贺某阳当时的住所至少20公里。安顿下来后，她找了份在美容院上班的工作。平静的生活没有持续太久，玉梅说，2023年10月28日前后，她照常坐地铁回家，刚准备出站，就看到贺某阳在十米开外等着自己。回忆这段经历时，她带着澎湃新闻记者去到当时的地铁站，该站有四个出口，下班高峰时人流密集，“他就在我要出入的地铁口等着我。”玉梅说，她至今仍然无法想通，贺某阳究竟是怎么一次次找到自己的。无论如何，逃跑变得更加困难。“我上班，他就送我去上班。我下班的时候，他就已经提前在那里等着我。他说，只要我跑或者有别的想法，他就去我单位闹，”玉梅说。2023年4月15日，最后一次家暴发生前九天，是玉梅最接近离婚的一次。当天中午，贺某阳在砂锅店向玉梅泼去四盘滚烫的砂锅。店员报警后，玉梅在派出所坚持要求依法处理。民警再次以家庭纠纷为由劝她选择调解时，她第一次失控了。“我说，如果你们的女儿，你们的姐姐，你们的妹妹，你们的妈妈遇到这种事情了，你们还会这样轻描淡写一句话，‘你们都有孩子了，要为孩子考虑’，你们还会说这句话吗？”最终，在警方的劝说下，玉梅和贺某阳签署了一份离婚协议。她放弃分割财产，抚养权归男方所有，每月向其支付5000元的抚养费。“当时我的想法就是，只要他同意离婚，只要他放过我，让我有一条生路就可以。”她说。在办案人员的见证下，双方约定好了去民政局办理手续的时间。按照民法典规定，夫妻双方需携带离婚协议到民政局申请离婚登记，三十日离婚冷静期过后，婚姻登记机关出具离婚证，方可认定为生效。但刚出派出所的门，贺某阳便反悔了。至此，玉梅多次报警、尝试逃跑、协议离婚，试图摆脱家庭暴力的阴影，均以失败告终。
```

如果我们打开deepseek-r1进行推理，观察它的推理过程（我们能从中找到机器分析的依据）：
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215160830424-2033650624.png)
这个粒度的长文本，对于TextCNN、TextRNN等都是很难处理长距离语义关系的的，而以BERT为代表的预训练深度学习方法负载又太大，往往缺少资源。这种大语言模型利用本身的语言能力，通过小样本甚至零样本的方式直接提示词分类，是很好的方法。


## 3.1.2 情感计算

情感计算（Affective Computing）是一门通过计算机科学、心理学和认知科学的交叉研究，实现对人类情感状态的识别、分析、建模与响应的技术领域。其核心目标在于赋予机器理解、模拟甚至表达情感的能力，从而推动人机交互向更自然、更共情的维度发展。这一概念由MIT媒体实验室的Rosalind Picard于1997年首次提出，现已成为情感智能系统、心理健康监测、个性化推荐等应用的技术基石。

在情绪心理学领域，研究者通过构建理论框架解析情感的本质结构。Paul Ekman的基本情绪理论（1972）将人类情绪归纳为六种跨文化普适的类别：快乐、悲伤、愤怒、恐惧、厌恶和惊讶，这一分类成为早期情感分类模型的基础。情绪心理学家通常采用维度模型来对情感和情绪进行建模，例如Plutchik的情绪轮模型，该模型将情绪分为八种基本情绪，并通过这些情绪的组合和强度变化来描述更复杂的情感状态。而James Russell提出的环形情绪模型（1980）则从效价（Valence，愉悦-不愉悦）和唤醒度（Arousal，平静-兴奋）两个连续维度描述情绪状态，为情感量化提供了数学映射空间。PAD三维情感模型是由Mehrabian 和Russell于1974年提出的维度观测量模型。该模型认为情感具有愉悦度、激活度和优势度3个维度，其中P代表愉悦度（Pleasure-displeasure），表示个体情感状态的正负特性;A代表激活度（Arousal-nonarousal），表示个体的神经生理激活水平;D代表优势度（Dominance-submissiveness），表示个体对情景和他人的控制状态。同时也可以通过这3个维度的值来代表具体的情感，如愤怒的坐标为（-0.51,0.59,0.25）（各维度上的数值范围为-1到+1，+1表示在此维度上的值高，而-1表示在此维度上的值低）。
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215171920616-1143335899.png)
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215172026630-1982889141.png)


心理学家通过实验和观察来研究情绪的产生、表达和影响，以及情绪在不同情境下的变化规律。自然语言处理和计算语言学领域的研究者则侧重于从文本数据中提取情感信息，他们通过构建词汇表、语法规则和语义分析模型来识别和分类文本中的情感极性和强度。自然语言处理领域的情感建模则聚焦于从文本中提取情感信号。计算语言学家通过情感词典（如SentiWordNet）和语法规则捕捉显式情感词汇（如“美好”“绝望”）与隐式情感表达（如反讽“这真是个好主意”）。例如，VADER（Valence Aware Dictionary and sEntiment Reasoner）是一种基于规则的情感分析工具，它通过分析文本中的词汇和语法结构来评估情感倾向。另外，**大连理工大学情感本体词表**是我国发布的中文情感计算标注本体，对每个词汇的情感类型与强度进行了标注。随着深度学习的发展，研究者如Bo Pang（2002）开创性地将支持向量机（SVM）应用于影评情感分类，而近年来基于Transformer的预训练模型（如BERT）通过自注意力机制建模上下文情感依赖，显著提升了细粒度情感识别的性能。

情感极性和情感强度的分析是情感计算中的重要任务。情感极性通常指文本中的情感倾向，如正面、负面或中性，而情感强度则描述情感的强烈程度。研究者们开发了多种模型和算法来计算情感极性和强度，包括基于机器学习的方法，如支持向量机（SVM）、随机森林（RF）和神经网络模型，这些方法通过训练数据集来学习情感分类模型，然后对未知文本进行情感分类。此外，基于深度学习的方法，如卷积神经网络（CNN）、循环神经网络（RNN）及其变体（如LSTM和GRU），也被广泛应用于情感分析中，这些模型能够自动提取文本特征并进行情感分类。情感强度的计算通常涉及到对情感词汇的强度进行量化，例如通过构建情感词汇表并赋予每个词汇一个强度分数，然后通过加权求和的方式来计算整个文本的情感强度。

情绪分类任务将文本映射到离散的情绪类别（如Ekman六类或Plutchik的八类情绪），其数学模型本质是高维空间中的多分类问题。传统方法依赖特征工程构建情绪相关特征（如表情符号频率、情绪词分布），而现代神经网络的端到端学习通过嵌入层自动捕获情绪语义特征。例如，卷积神经网络（CNN）利用局部感受野提取情绪短语模式（如“喜极而泣”中的复合情感），而分层注意力网络（Yang et al., 2016）通过词级和句级注意力权重识别文本中的情绪焦点。评估中常采用加权F1分数以应对类别不均衡问题，例如在心理健康分析中，“抑郁”类样本可能远少于“中性”类。

对于情感分析，也有一些现成的模型来分析。比如NLTK和SnowNLP。以下是使用 NLTK 和 SnowNLP 进行情感分析的代码示例：
**使用 NLTK 进行情感分析**

```python
# 安装 nltk 和对应的 sentiment 数据包
pip install nltk

# 在 Python 中加载 nltk 数据
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

# 初始化情感分析器
sia = SentimentIntensityAnalyzer()

# 进行情感分析
text = "I love this product! It's absolutely amazing."
sentiment = sia.polarity_scores(text)

print(f"Original Text: {text}")
print(f"Sentiment Scores: {sentiment}")
print(f"Overall Sentiment: {'positive' if sentiment['compound'] > 0 else 'negative'}")
```

输出示例：

```
Original Text: I love this product! It's absolutely amazing.
Sentiment Scores: {'neg': 0.0, 'neu': 0.294, 'pos': 0.706, 'compound': 0.8316}
Overall Sentiment: positive
```

**使用 SnowNLP 进行情感分析**
SnowNLP 是一个适用于中文文本的情分析工具，安装方法如下：
``pip install snownlp``
以下是一个简单的代码示例：

```python
from snownlp import SnowNLP

# 中文文本情感分析
text = "这个产品真的很好用，我很满意！"
s = SnowNLP(text)
sentiment = s.sentiments  # 返回一个 0 到 1 之间的值，1 表示积极，0 表示消极

print(f"Original Text: {text}")
print(f"Sentiment Score: {sentiment}")
print(f"Overall Sentiment: {'positive' if sentiment > 0.5 else 'negative'}")
```

输出示例：

```
Original Text: 这个产品真的很好用，我很满意！
Sentiment Score: 0.9876543209876543
Overall Sentiment: positive
```

NLTK 是一个强大的自然语言处理工具，适用于英文文本的情感分析。SnowNLP 是一个专注于中文文本的工具，提供了丰富的功能，包括情感分析、文本相似性分析等。除了情感分析以外，SnowNLP还有更多用法。例如：

```python
from snownlp import SnowNLP

# 输入文本
text = "我非常喜欢这部电影，演员表现得很好。"
s = SnowNLP(text)

# 1. 中文分词
words = s.words
print("分词结果:", words)

# 2. 词性标注
tags = s.tags
print("词性标注:", tags)

# 3. 情感分析
sentiment_score = s.sentiments
print("情感得分:", sentiment_score)
if sentiment_score > 0.5:
    print("情感倾向: 正面")
else:
    print("情感倾向: 负面")

# 4. 关键词提取
keywords = s.keywords(3)  # 提取前3个关键词
print("关键词:", keywords)

# 5. 摘要生成
summary = s.summary(2)  # 生成2句摘要
print("摘要:", summary)

# 6. 拼音转换
pinyin = s.pinyin
print("拼音结果:", pinyin)

# 7. 简繁转换
traditional = s.han  # 转换为繁体
print("繁体转换:", traditional)
```


观点挖掘（Opinion Mining）是指从文本数据中提取和分析观点、态度和情感的过程。其任务定义包括识别文本中的观点持有者、观点目标、观点极性和观点强度等。观点挖掘的方法发展经历了从传统的基于规则的方法到现代的基于机器学习和深度学习的方法的转变。早期的方法依赖于手工构建的规则和模式来识别观点，而现代方法则利用机器学习算法和深度学习模型来自动学习观点的特征表示。例如，基于深度学习的模型，如Transformer架构的BERT，通过预训练和微调，能够有效地捕捉文本中的观点信息。此外，观点挖掘还涉及到跨领域和跨语言的观点识别，以及观点的细粒度分析，如方面级观点挖掘和句子级观点挖掘。其实说白了，观点态度主要就是一个四分类：【支持、反对、中立、无关】。

细粒度情感分析（Aspect-based Sentiment Analysis, ASA）是观点挖掘的高级形态，其目标是在句子或段落层级上，针对特定评价对象（Aspect）进行情感判断。例如，在评论“餐厅环境优雅，但服务速度慢”中，ASA需分别识别“环境→积极”和“服务速度→消极”。这一任务依赖双重注意力机制（Wang et al., 2016），其中上下文注意力定位评价对象，情感注意力聚焦相关描述词。当前最先进的模型如BART-Aspect（Xia et al., 2021）通过序列到序列框架联合生成评价对象及其情感倾向，在SemEval-2014数据集上达到92.3%的F1值，展现了细粒度建模的强大能力。关于细粒度情感分析的例子，我们可以从SemEval-2014的官网上找到对问题的定义。ASA被分解为四个主要要点：

- **子任务 1：方面术语提取**
  给定一组具有预先识别实体（例如，restaurants）的句子，识别句子中存在的 aspect 术语，并返回一个包含所有不同 aspect 术语的列表。方面术语指定目标实体的特定方面。

> 例如，“我喜欢这里的服务和工作人员，但不喜欢食物”，“食物没什么，但我喜欢这里的工作人员”。多词 aspect 术语（例如，“hard disk”）应被视为单个术语（例如，在“The hard disk is very noisy”中，唯一的 aspect 术语是 “hard disk”）。

- **子任务 2：方面项极性**
  对于句子中给定的一组方面术语，确定每个方面术语的极性是积极、消极、中性还是冲突（即积极和消极）。

> 例如：
> “我喜欢他们的法吉塔” → {fajitas： positive}
> “我讨厌他们的法吉塔，但他们的沙拉很棒” → {fajitas： negative， salads： positive}
> “法吉塔是他们的第一盘” → {fajitas： neutral}
> “法吉塔很好尝，但不好看” → {fajitas： conflict}
> understand?

- **子任务 3：方面类别检测**
  给定一组预定义的方面类别（例如，价格、食品），确定给定句子中讨论的方面类别。方面类别通常比 Subtask 1 的方面术语更粗糙，并且它们不一定在给定句子中作为术语出现。

> 例如，给定一组方面类别 {food， service， price， ambience， anecdotes/miscellaneous}：
> “餐厅太贵了” → {price}
> “餐厅很贵，但菜单很棒” → {price， food}
> understand?

- **子任务 4：方面类别极性**
  给定一组预先确定的方面类别（例如，{food， price}），确定每个方面类别的极性（积极、消极、中性或冲突）。

> 例如：
> “餐厅太贵了” → {price： negative}
> “餐厅很贵，但菜单很棒” → {price： negative， food： positive}
> understand?就是这样。

我们把这个要求告诉大模型，来看看大语言模型的细粒度情感分析能力究竟如何吧！

```
user-prompt:
**角色**：假如你是一个高性能文本处理与分析助手，现在需要对给定的社交媒体文本进行`细粒度情感分析`任务。
**任务描述**：根据SemEval-2014对细粒度情感分析（ABSA）的定义，这一任务包含以下几个子任务：
（这里把上面的要求和示例都写进来）
**输入**：如下：
> “这款手机真的太棒了！从外观设计到性能表现都让我非常满意。首先，它的外观非常时尚，机身颜色很正，拿在手里质感十足。屏幕显示效果清晰细腻，无论是看视频还是浏览图片都有很棒的视觉体验。性能方面，运行速度非常快，多任务处理毫无压力，玩大型游戏也十分流畅，几乎没有出现卡顿的情况。拍照功能也很强大，照片清晰度高，色彩还原度好，各种场景下都能拍出好看的照片。续航能力也很不错，重度使用一天下来还剩不少电量，充电速度也很快，半小时就能充到一大半。客服服务也很贴心，有问题咨询时回复及时，态度友好，解决了我不少疑惑。总之，这次购物体验非常棒，手机的品质和性能都超出了我的预期，强烈推荐给大家！”
```

我们看看它会生成什么玩意。
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215171134642-1848759598.png)

```json
{
  "Subtask1": {
    "Aspect Terms": [
      "外观设计",
      "机身颜色",
      "屏幕显示效果",
      "运行速度",
      "多任务处理",
      "拍照功能",
      "照片清晰度",
      "色彩还原度",
      "续航能力",
      "充电速度",
      "客服服务"
    ]
  },
  "Subtask2": {
    "Aspect Polarities": {
      "外观设计": "positive",
      "机身颜色": "positive",
      "屏幕显示效果": "positive",
      "运行速度": "positive",
      "多任务处理": "positive",
      "拍照功能": "positive",
      "照片清晰度": "positive",
      "色彩还原度": "positive",
      "续航能力": "positive",
      "充电速度": "positive",
      "客服服务": "positive"
    }
  },
  "Subtask3": {
    "Aspect Categories": [
      "外观",
      "屏幕",
      "性能",
      "拍照",
      "续航",
      "服务"
    ]
  },
  "Subtask4": {
    "Category Polarities": {
      "外观": "positive",
      "屏幕": "positive",
      "性能": "positive",
      "拍照": "positive",
      "续航": "positive",
      "服务": "positive"
    }
  }
}
```

这种带着明确任务定义和少数案例的提示方式叫上下文学习（context-learning），这种提示词设计相比单纯地让它做ASA要更加明智一些。比它更厉害的是思维链（CoT），这个我们会在后面提到。


## 3.1.3 主题模型

**1. 文本聚类的定义、与文本分类的区别及示例**
**定义**：
文本聚类是指将一组文本数据根据其内容的相似性自动划分为若干个类别或簇的过程，每个簇内的文本在主题、语义或特征上具有较高的相似性，而不同簇之间的文本则具有较大的差异性。例如，假设我们有一组包含多种主题的新闻文章，如体育、科技、娱乐等，但这些文章并没有预先标注类别。通过文本聚类，算法可以自动将这些文章划分为几个簇，每个簇内的文章主要集中在某一特定主题上，比如体育新闻簇、科技新闻簇等，从而帮助我们快速了解这组文本数据的主要内容分布。

文本分类与文本聚类不同，文本分类是一种有监督学习方法，需要预先标注好的训练数据。在训练过程中，算法通过学习标注数据的特征与类别之间的映射关系，构建分类模型。当遇到新的未见过的文本时，分类模型能够根据其特征将其划分到预先定义好的类别中。例如，如果我们有一组已标注为"体育""科技""娱乐"等类别的新闻文章，通过训练一个文本分类模型，当遇到一篇新的新闻文章时，模型可以将其自动分类到这些预定义的类别之一，从而实现文本的自动化分类。

**K-Means文本聚类demo**：

```python
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# 随机生成一些文本数据
sentences = [["word", "word2", "word3"], ["word4", "word5", "word6"], ...]
# 使用Word2Vec训练词向量
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
word_vectors = model.wv
# 将词向量作为文本表示
X = np.array([word_vectors[word] for word in word_vectors.index_to_key])
# 使用K-Means进行聚类
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmeans.labels_
# 使用t-SNE进行可视化
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)
# 绘制聚类结果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels)
plt.colorbar()
plt.show()
```

**2. 什么是话题模型？**
话题模型是一种统计模型，用于发现文本数据中的隐藏话题结构。它假设每篇文本是由多个话题混合而成，每个话题则是一组相关词汇的概率分布。通过话题模型，可以揭示文本集合中潜在的主题信息，有助于理解大规模文本数据的内容和结构。

**3. LDA（隐狄利克雷分布）的数学原理和推导**
**数学原理**：
LDA（Latent Dirichlet Allocation）是一种生成式概率话题模型，基于贝叶斯理论。它假设每个文档是由多个话题混合而成，每个话题则是一组词汇的概率分布。LDA使用狄利克雷分布作为话题分布的先验分布。
**推导**：

1. **生成过程**：对于每个文档，首先从狄利克雷分布中采样得到话题分布，然后对于文档中的每个词，从话题分布中采样一个话题，最后从该话题的词汇分布中采样一个词。
2. **概率公式**：给定文档集合$D$，话题数$K$，词汇表$V$，LDA的目标是估计话题分布$\theta$和词汇分布$\phi$的 posterior distribution。
3. **变分推理**：由于直接计算后验分布复杂，LDA通常采用变分推理进行近似，通过优化变分参数来逼近真实的后验分布。

**代码案例**

```python
import pandas as pd
from gensim import corpora, models
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.models.phrases import Phrases, Phraser
import os
import jieba
import warnings
warnings.filterwarnings('ignore')

# 定义一个函数来读取停用词表文件并返回一个集合
def load_stopwords(file_path):
    # 创建一个空集合用于存储停用词
    stopwords = set()
    
    # 打开文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉每行的首尾空白字符（如换行符）并添加到集合中
            stopwords.add(line.strip())
    
    return stopwords

# 假设停用词表文件的路径是 'stopwords.txt'
file_path = 'stopwords.txt'

# 调用函数加载停用词
default_stopwords = load_stopwords(file_path) | set(STOPWORDS)

# 添加自定义的停用词
custom_stopwords = {"投稿", "文章呼吁", "大妈观点","粉丝投稿",'的','了','是','在','这','个','吗','我','你','他'}


# 合并两个集合
all_stopwords = default_stopwords.union(custom_stopwords)

def simplified_preprocess(text):
    return [token for token in jieba.lcut(text) if token not in all_stopwords and len(token) > 3]

def compute_coherence_values(dictionary, corpus, texts, limit, start=4, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print('============'+str(num_topics)+'=============')
        model = gensim.models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=130,alpha='auto',eta='auto',random_state=1)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def main():
    dir_path = './数据（原创+转发'   # 待处理文件的目录
    file_paths = os.listdir(dir_path)
    all_content = []

    for file_path in file_paths:
        print(file_path)
        df = pd.read_csv(dir_path+'/'+file_path,encoding='gb18030')
        df=df.fillna('')
        #if len(df)>3000:
        #    df=df.sample(n=3000)
        all_content.extend(df['原微博内容'].dropna().tolist())

    print("数据预处理阶段")
    all_processed_docs = [simplified_preprocess(doc) for doc in all_content]
    all_dictionary = corpora.Dictionary(all_processed_docs)
    all_corpus = [all_dictionary.doc2bow(doc) for doc in all_processed_docs]
    print("主题建模阶段")
    # all_lda_model = gensim.models.LdaModel(all_corpus, num_topics=20, id2word=all_dictionary, passes=10)
    #
    # all_topics = all_lda_model.print_topics(num_words=10)
    # for topic in all_topics:
    #     print(topic)
    limit = 24
    start = 15
    step = 1
    model_list, coherence_values = compute_coherence_values(dictionary=all_dictionary, corpus=all_corpus,
                                                            texts=all_processed_docs, start=start, limit=limit,
                                                            step=step)

    # 输出结果
    for num_topics, cv in zip(range(start, limit, step), coherence_values):
        print("Num Topics =", num_topics, " has Coherence Value of", round(cv, 4))

    # 选择最佳主题数并进行可视化
    best_model_index = coherence_values.index(max(coherence_values))
    best_lda_model = model_list[best_model_index]
    print("Best Num Topics =", start + best_model_index * step)
    # 可视化
    vis = gensimvis.prepare(best_lda_model, all_corpus, all_dictionary)
    pyLDAvis.save_html(vis, './评论/主题/LDAvis.html')

if __name__ == "__main__":
    main()
```

**4. 深度学习时代的话题模型发展**
随着深度学习的兴起，话题模型也得到了新的发展。深度学习模型，如循环神经网络（RNN）、卷积神经网络（CNN）和Transformer，被引入到话题模型中，以捕捉文本的复杂结构和语义信息。其中，BERTopic是一个代表性的深度学习话题模型，它结合了BERT的预训练语言模型和传统的话题模型方法，能够更准确地发现文本中的潜在话题，并提供了更丰富的语义表示。
**5. BERTopic的代码案例demo**

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
# 加载数据
data = fetch_20newsgroups(subset='all')['data']
# 初始化BERTopic模型
topic_model = BERTopic()
# 拟合模型
topics, _ = topic_model.fit_transform(data)
# 可视化话题
topic_model.visualize_topics()
```

这个demo首先加载了20个新闻组的数据，然后使用BERTopic模型进行话题发现，并可视化展示发现的话题。BERTopic利用了BERT的强大语义理解能力，能够更有效地从文本中提取话题信息。



# 3.2 信息抽取

## 3.2.1 命名实体识别

这一节我们看看信息抽取的有关内容。信息抽取其实就是对非结构化的一大段文本，把里面关键的信息抽出来并组织成结构化查询模式。要抽的东西比如时间、地点、人物、数字、动作、事件、关系等，这也是篇章级文本分析的基础，也是知识图谱的基础。在LLM诞生之前，有很多研究都是在做信息抽取和知识图谱的。信息抽取的核心其实是个序列标注问题，我们在下面就可以看到。本节重点讨论的事情有四个：命名实体识别（NER）、事件抽取（EE）、关系抽取（RE）、知识图谱（KG）。

命名实体（Named Entity）是指文本中具有特定意义的专有名词或特定概念，通常包括人名、地名、组织名、日期、时间、数量、货币、百分比等。这些实体在文本中具有明确的语义和指代，是信息提取、问答系统、机器翻译等自然语言处理任务中的关键要素。例如，在句子“我在2024年5月1日去了北京天安门广场”中，“2024年5月1日”是一个日期实体，“北京天安门广场”是一个地名实体。命名实体的识别和分类对于理解文本内容、提取关键信息以及构建知识图谱等任务具有重要意义。
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215175508127-811622767.png)



命名实体识别（Named Entity Recognition, NER）任务是指从文本中自动识别出命名实体，并将其分类为预定义的类别，如人名、地名、组织名等。这一任务是自然语言处理中的基础任务之一，广泛应用于信息抽取、问答系统、机器翻译等领域。评测命名实体识别模型性能的常用指标包括准确率（Precision）、召回率（Recall）和F1值（F1 Score）。准确率衡量的是模型识别出的实体中正确实体的比例，召回率衡量的是文本中实际存在的实体被模型正确识别的比例，F1值则是准确率和召回率的调和平均数，用于综合评估模型的性能。

![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215175529293-1416698818.png)


BIO标注法是一种常用的命名实体识别标注方法，其中B表示实体的开始（Begin），I表示实体的内部（Inside），O表示非实体（Outside）。在命名实体识别任务中，每个单词或字符都会被标注为B、I或O，以表示其在实体中的位置。例如，对于句子“我在2024年5月1日去了北京天安门广场”，BIO标注结果可能如下：

```
我 - O
在 - O
2024 - B-date
年 - I-date
5 - B-date
月 - I-date
1 - I-date
日 - I-date
去了 - O
北京 - B-location
天安门 - I-location
广场 - I-location
```

早期的命名实体识别方法主要基于规则和词典。这些方法通过手工编写规则和构建词典来识别文本中的命名实体。例如，通过定义人名、地名、组织名等的规则模式，结合词典中的专有名词，来匹配和识别文本中的实体。然而，这些方法的局限性在于规则的覆盖范围有限，且难以处理复杂和多样化的文本数据。随着机器学习和深度学习的发展，命名实体识别方法逐渐转向基于统计和神经网络的模型。机器学习方法如隐马尔可夫模型（HMM）、条件随机场（CRF）等，通过学习文本中的特征和模式来识别命名实体。深度学习方法如循环神经网络（RNN）、长短期记忆网络（LSTM）、门控循环单元（GRU）以及Transformer架构的模型（如BERT），能够自动提取文本的上下文信息和语义特征，显著提高了命名实体识别的准确性和鲁棒性。这些模型在大规模语料上的预训练和微调，使得命名实体识别在各种自然语言处理任务中取得了显著的性能提升。

我们来动手做一个基于BiLSTM-CRF的命名实体识别模型：

**数据预处理**

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 示例标注数据
sentences = [
    "John works in New York",
    "Apple releases new iPhone in 2023",
    "The WHO is headquartered in Geneva"
]

# BIO标注
tags = [
    ["B-PER", "O", "O", "B-LOC", "I-LOC"],
    ["B-ORG", "O", "O", "B-PROD", "O", "B-DATE"],
    ["O", "B-ORG", "O", "O", "O", "B-LOC"]
]

# 构建词汇表和标记表
vocab = set()
tag_set = set()

for sentence in sentences:
    vocab.update(sentence.split())
    
for tag_list in tags:
    tag_set.update(tag_list)

vocab = sorted(list(vocab))
tag_list = sorted(list(tag_set))

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
tag_to_idx = {tag: idx for idx, tag in enumerate(tag_list)}
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

# 数据集类
class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_to_idx, tag_to_idx):
        self.sentences = sentences
        self.tags = tags
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx].split()
        tag = self.tags[idx]
        
        word_indices = [self.word_to_idx.get(word, len(self.word_to_idx)) for word in sentence]
        tag_indices = [self.tag_to_idx.get(t, len(self.tag_to_idx)) for t in tag]
        
        return torch.tensor(word_indices), torch.tensor(tag_indices), len(word_indices)
```

**模型定义**

```python
import torch.nn as nn

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_idx, embedding_dim, hidden_dim, dropout_prob=0.1):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx)
        
        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim)  # +1 for UNK
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)
        
    def _get_lstm_features(self, sentences):
        embeds = self.word_embeds(sentences)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def forward(self, sentences, tags=None):
        lstm_feats = self._get_lstm_features(sentences)
        loss = None
        if tags is not None:
            loss = self.crf(lstm_feats, tags, reduction='mean')
        return loss, lstm_feats
```

**模型训练**

```python
# 数据集和数据加载器
dataset = NERDataset(sentences, tags, word_to_idx, tag_to_idx)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# 帮助函数：将批次数据填充到相同长度
def collate_fn(batch):
    sentences, tags, lengths = zip(*batch)
    padded_sentences = pad_sequence(sentences, batch_first=True)
    padded_tags = pad_sequence(tags, batch_first=True)
    lengths = torch.tensor(lengths)
    return padded_sentences, padded_tags, lengths

# 模型训练
embedding_dim = 100
hidden_dim = 200
learning_rate = 0.01
num_epochs = 10

model = BiLSTM_CRF(vocab_size=len(word_to_idx), tag_to_idx=tag_to_idx, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    
    for batch in train_loader:
        sentences, tags, lengths = batch
        sentences = sentences.to(device)
        tags = tags.to(device)
        
        optimizer.zero_grad()
        loss, _ = model(sentences, tags)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
```

**测试模型**

```python
def test_model(model, dataloader, device, tag_to_idx, idx_to_tag):
    model.eval()
    true_tags = []
    pred_tags = []
    
    with torch.no_grad():
        for batch in dataloader:
            sentences, tags, lengths = batch
            sentences = sentences.to(device)
            tags = tags.to(device)
            
            _, lstm_feats = model(sentences)
            decoded_tags = model.crf.decode(lstm_feats)
            
            # Flatten the tags for evaluation
            true_tags.extend([tag[:length].tolist() for tag, length in zip(tags.cpu(), lengths)])
            pred_tags.extend([torch.tensor(tags, dtype=torch.long)[:length].tolist() for tags, length in zip(decoded_tags, lengths)])
    
    true_tags_flat = [tag for sublist in true_tags for tag in sublist]
    pred_tags_flat = [tag for sublist in pred_tags for tag in sublist]
    
    print("Accuracy:", accuracy_score(true_tags_flat, pred_tags_flat))
    print("Precision, Recall, F1:", precision_recall_fscore_support(true_tags_flat, pred_tags_flat, average='weighted'))
    
test_model(model, val_loader, device, tag_to_idx, idx_to_tag)

# 示例句子
test_sentence = "Google released a new Pixel phone in 2023"
test_words = test_sentence.split()
test_indices = [word_to_idx.get(word, len(word_to_idx)) for word in test_words]
test_tensor = torch.tensor([test_indices]).to(device)

model.eval()
with torch.no_grad():
    _, lstm_feats = model(test_tensor)
    decoded_tags = model.crf.decode(lstm_feats)
    
predicted_tags = [idx_to_tag[idx] for idx in decoded_tags[0]]

print("Sentence:", test_sentence)
print("Predicted Tags:", predicted_tags)
```

当然，我们也可以利用大模型进行命名实体识别。比如，我们直接提出命名实体识别的要求：

```
user-prompt:
请对下面的文本进行命名实体识别：
张伟出生在辽宁葫芦岛的一个山村，30岁前，他的人生像开挂一样，保送省里最好的高中理科实验班，保送北大，毕业后全奖直博美国专业排名前三的名校，成为全球大药厂资深首席科学家。那时，他是冉冉升起的明日之星。
```

返回结果：

```
张  B-PER
伟  I-PER
出  O
生  O
在  O
辽  B-LOC
宁  I-LOC
葫  B-LOC
芦  I-LOC
岛  I-LOC
的  O
一  O
个  O
山  O
村  O
，  O
30  B-QUANTITY
岁  I-QUANTITY
前  O
，  O
他  O
的  O
人  O
生  O
像  O
开  O
挂  O
一  O
样  O
，  O
保  O
送  O
省  O
里  O
最  O
好  O
的  O
高  O
中  O
理  O
科  O
实  O
验  O
班  O
，  O
保  O
送  O
北  B-ORG
大  I-ORG
，  O
毕  O
业  O
后  O
全  O
奖  O
直  O
博  O
美  B-LOC
国  I-LOC
专  O
业  O
排  O
名  O
前  O
三  O
的  O
名  O
校  O
，  O
成  O
为  O
全  O
球  O
大  O
药  O
厂  O
资  O
深  O
首  O
席  O
科  O
学  O
家  O
。  O
那  O
时  O
，  O
他  O
是  O
冉  O
冉  O
升  O
起  O
的  O
明  O
日  O
之  O
星  O
。  O
```

## 3.2.2 事件抽取

事件是自然语言中描述具有特定行为或状态变化的语义单元，通常由事件触发词（Event Trigger）和事件论元（Event Argument）组成。事件触发词是直接表示事件行为或状态变化的核心词汇，如“签署”“发生”“上涨”等，它是事件的核心，明确了事件的类型。事件论元则是与事件触发词相关联的实体或概念，用于补充说明事件的参与者、时间、地点等信息。例如，在句子“公司于2024年5月签署了合作协议”中，“签署”是事件触发词，表示事件的行为类型；“公司”和“合作协议”是事件论元，分别表示事件的参与者和对象；“2024年5月”是时间论元，表示事件发生的时间。事件的组成要素和论元共同构成了对事件的完整描述，是事件抽取任务中的关键信息。

事件抽取（Event Extraction）任务是指从文本中自动识别出事件触发词及其对应的事件论元，并将它们组织成结构化的事件表示。这一任务是信息抽取和知识图谱构建中的重要环节，旨在从非结构化的文本中提取出具有语义价值的事件信息。事件抽取的评测指标通常包括准确率（Precision）、召回率（Recall）和F1值（F1 Score）。准确率衡量的是模型抽取的事件中正确事件的比例，召回率衡量的是文本中实际存在的事件被模型正确抽取的比例，F1值则是准确率和召回率的调和平均数，用于综合评估模型的性能。

BIO标注法在事件抽取任务中被广泛应用，用于标注文本中每个单词或字符的语义角色。B表示事件触发词或论元的开始（Begin），I表示事件触发词或论元的内部（Inside），O表示非事件相关的部分（Outside）。例如，在句子“公司于2024年5月签署了合作协议”中，BIO标注结果可能如下：

```
公司 - B-ORG
于 - O
2024 - B-TIME
年 - I-TIME
5 - I-TIME
月 - I-TIME
签署 - B-TRIGGER
了 - O
合作 - B-ARGUMENT
协议 - I-ARGUMENT
```

所以你们看，这还是个序列标注问题。

```python
# 示例文本
text = "公司于2024年5月签署了合作协议"

# 假设模型预测结果
predicted_triggers = ["签署"]
predicted_arguments = [
    {"argument": "公司", "role": "Participant"},
    {"argument": "合作协议", "role": "Object"},
    {"argument": "2024年5月", "role": "Time"}
]

# 输出事件
event = {
    "trigger": predicted_triggers[0],
    "arguments": [
        {"role": arg["role"], "argument": arg["argument"]}
        for arg in predicted_arguments
    ]
}

print("抽取的事件：", event)
```

早期的事件抽取方法主要基于规则和模式匹配。这些方法通过手工编写规则或正则表达式来识别事件触发词和论元。例如，通过定义与特定事件类型相关的关键词和语法模式，结合领域知识，来匹配和提取文本中的事件信息。然而，这些方法的局限性在于规则的覆盖范围有限，难以适应多样化的文本数据和复杂的语言现象，且对领域知识的依赖程度较高。随着机器学习和深度学习技术的发展，事件抽取方法逐渐转向基于统计和神经网络的模型。机器学习方法如支持向量机（SVM）、条件随机场（CRF）等，通过学习文本中的特征和模式来识别事件触发词和论元。深度学习方法如DMCNN、BERT-CRF等模型框架，能够自动提取文本的上下文信息和语义特征，显著提高了事件抽取的准确性和鲁棒性。近年来，基于预训练语言模型的方法在事件抽取任务中取得了显著的性能提升，成为研究和应用的主流方向。

OmniEvent是清华大学开源的一款用于事件抽取的库，支持多种事件抽取任务，包括触发词检测、事件论元抽取等。该库提供了预训练模型和易于使用的接口，极大简化了事件抽取的实现。全面涵盖各种范式，并对广泛使用的英文和中文数据集提供公平、统一的评估。模块化实现使 OmniEvent 具有高度可扩展性。我这里给出一个demo：

```python
from OmniEvent import infer
device='cpu'


# 初始化OmniEvent模型
def init_models(device='cpu'):
    ed_model, ed_tokenizer = infer.get_pretrained("s2s-mt5-ed",device=device)
    eae_model, eae_tokenizer = infer.get_pretrained("s2s-mt5-eae",device=device)
    return (ed_model, ed_tokenizer,eae_model,eae_tokenizer)

# 进行事件抽取
def run_ee(models,news_text_list,device='cpu'):
    schemas = ["<ace>"]*len(news_text_list)
    
    ed_model, ed_tokenizer,eae_model,eae_tokenizer=models
    events = infer.do_event_detection(ed_model, ed_tokenizer, news_text_list, schemas, device)
    instances = infer.prepare_for_eae_from_pred(news_text_list, events, schemas)
    

    if len(instances[0]["triggers"]) == 0:
        results = [{
            "text": instances[0]["text"],
            "events": []
        }]
        return results
    arguments = infer.do_event_argument_extraction(eae_model, eae_tokenizer, instances, device)
    results = infer.get_eae_result(instances, arguments)
    return results


text = "“中比动漫文化周”12日在布鲁塞尔中国文化中心开幕，旨在通过动漫这一独特的艺术形式，促进两国文化的交流与理解。"
models=init_models(device)
ee=run_ee(models,[text])
# 输出结果

for events in ee:
    print("文本:", events['text'])
    for event in events['events']:
        print(f"事件类型: {event['type']}")
        print(f"触发词: {event['trigger']}")
        print(f"论元: {event['arguments']}")
```

这是事件抽取比较成熟的方案。当然，根据BIO标注格式自己处理数据然后训一个BERT-BiLSTM-CRF用于事件抽取也不是不行，但是我建议如果是直接用的话还是用工具好。另外，也可以使用大语言模型，如果我不带任何规则约束直接提问：

> user-prompt: 请对下面的社交媒体文本进行事件抽取：
> 近日，伦敦帝国理工学院结构生物学名誉教授斯蒂芬·库里在其个人博客上发表了一封致英国皇家学会会长阿德里安·史密斯的公开信，对2018年皇家学会授予马斯克院士资格一事表达了强烈不满。

![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215182624636-205057353.png)

```json
{
  "event": {
    "type": "学术抗议",
    "trigger": ["发表公开信", "表达强烈不满"],
    "participants": {
      "initiator": {
        "name": "斯蒂芬·库里",
        "affiliation": "伦敦帝国理工学院",
        "position": "结构生物学名誉教授"
      },
      "recipient": "阿德里安·史密斯",
      "organization": "英国皇家学会",
      "controversial_figure": "埃隆·马斯克"
    },
    "temporal": {
      "protest_time": "近日",
      "contested_event_time": "2018年"
    },
    "spatial": {
      "platform": "个人博客",
      "institution_location": "伦敦"
    },
    "content": {
      "core_issue": "院士资格授予决策",
      "specific_criticism": "对马斯克获选院士资格的异议",
      "manifestation": "通过正式公开信提出质疑"
    }
  }
}
```

可以看到，在R1推理模式下，即使不告诉它具体标注流程，它也可以给出比较准确的结构化信息。另外，我们在CCKS 2024的事件关系抽取赛道当中，就应用了大语言模型帮助我们进行分析，并取得了赛道第五的成绩。


## 3.2.3 关系抽取

实体之间的关系类型多种多样，具体取决于应用场景和领域需求。以下是一些常见的关系类型：

- 亲属关系：如“父亲”“母亲”“兄弟”“姐妹”等，表示实体之间的家庭关系。
- 工作关系：如“雇佣”“合作”“领导”等，表示实体之间的职业或工作关系。
- 地理位置关系：如“位于”“邻近”“属于”等，表示实体之间的地理位置关系。
- 所有权关系：如“拥有”“控制”“属于”等，表示实体之间的所有权或控制关系。
- 时间关系：如“在……之前”“在……之后”“同时”等，表示实体之间的时间顺序关系。
- 因果关系：如“导致”“引起”“因为”等，表示实体之间的因果关系。
- 部分整体关系：如“包含”“组成”“属于”等，表示实体之间的部分与整体关系。
- 比较关系：如“大于”“小于”“等于”等，表示实体之间的比较关系。

![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215201356291-388608071.png)
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215201409123-2030693631.png)


关系抽取（Relation Extraction）任务是指从文本中自动识别出实体对之间的关系类型。这一任务是信息抽取和知识图谱构建中的重要环节，旨在从非结构化的文本中提取出具有语义价值的关系信息。关系抽取的评测指标通常包括准确率（Precision）、召回率（Recall）和F1值（F1 Score）。准确率衡量的是模型抽取的关系中正确关系的比例，召回率衡量的是文本中实际存在的关系被模型正确抽取的比例，F1值则是准确率和召回率的调和平均数，用于综合评估模型的性能。
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215201340617-93874259.png)


关系抽取任务的数据标注通常采用三元组或四元组的形式，具体格式如下：

- 三元组格式：(实体1, 关系类型, 实体2)，表示实体1和实体2之间的关系类型。
- 四元组格式：(实体1, 关系类型, 实体2, 句子索引)，在三元组的基础上增加了句子索引，用于标识关系所在的句子。
  示例：假设我们有以下句子：
  ``“张伟是李华的父亲，他们住在纽约。”``
  标注结果可以是：

```
(张伟, 父亲, 李华)
(张伟, 位于, 纽约)
(李华, 位于, 纽约)
```

以下是一个简单的关系抽取示例，假设我们已经训练好一个关系抽取模型：

```python
# 示例文本
text = "张伟是李华的父亲，他们住在纽约。"

# 假设模型预测结果
predicted_relations = [
    {"entity1": "张伟", "relation": "父亲", "entity2": "李华"},
    {"entity1": "张伟", "relation": "位于", "entity2": "纽约"},
    {"entity1": "李华", "relation": "位于", "entity2": "纽约"}
]

# 输出关系
for relation in predicted_relations:
    print(f"实体1: {relation['entity1']}, 关系: {relation['relation']}, 实体2: {relation['entity2']}")
```

```
实体1: 张伟, 关系: 父亲, 实体2: 李华
实体1: 张伟, 关系: 位于, 实体2: 纽约
实体1: 李华, 关系: 位于, 实体2: 纽约
```

早期的关系抽取方法主要基于规则和模式匹配。这些方法通过手工编写规则或正则表达式来识别实体对之间的关系。例如，通过定义与特定关系类型相关的关键词和语法模式，结合领域知识，来匹配和提取文本中的关系信息。大家可以通过一个demo来感受一下基于正则式规则来抽取实体关系的结果：

```python
import re

def extract_causal_relationships(text):
    # 定义因果关系关联词的正则表达式模式
    pattern = re.compile(
        r'((\w+)(因为|由于|以便|导致|致使)\
        (\w+))|((\w+)(所以|因此|因而|从而)(\w+))|((\w+)(为了)(\w+))|(\
        (\w+)(使得)(\w+))',
        re.UNICODE
    )

    matches = pattern.findall(text)
    relationships = []

    for match in matches:
        # 处理不同的因果关系模式
        if match[0] != '':  # 因为/由于/以便/导致/致使
            cause = match[1]
            effect = match[3]
            relationships.append({'cause': cause, 'effect': effect})
        elif match[4] != '':  # 所以/因此/因而/从而
            cause = match[5]
            effect = match[7]
            relationships.append({'cause': cause, 'effect': effect})
        elif match[8] != '':  # 为了
            purpose = match[9]
            effect = match[10]
            relationships.append({'purpose': purpose, 'effect': effect})
        elif match[11] != '':  # 使得
            cause = match[12]
            effect = match[13]
            relationships.append({'cause': cause, 'effect': effect})

    return relationships

def print_result(relationships):
    for relation in relationships:
        if 'cause' in relation and 'effect' in relation:
            print(f"因果关系：'{relation['cause']}' 导致了 '{relation['effect']}'.")
        elif 'purpose' in relation and 'effect' in relation:
            print(f"目的关系：为了达到 '{relation['effect']}'，需要 '{relation['purpose']}'.")  
        else:
            print(f"其他关系：{relation}.")

# 示例文本
text = """
因为小明努力学习，所以他在考试中取得了好成绩。 
他每天早起，导致他在课堂上精神饱满。
为了提高编程能力，他参加了编程马拉松。
他努力工作，使得公司项目取得了成功。
"""

# 提取因果关系
relationships = extract_causal_relationships(text)
print_result(relationships)
```

然而，这些方法的局限性在于规则的覆盖范围有限，难以适应多样化的文本数据和复杂的语言现象，且对领域知识的依赖程度较高。随着机器学习和深度学习技术的发展，关系抽取方法逐渐转向基于统计和神经网络的模型。机器学习方法如支持向量机（SVM）、条件随机场（CRF）等，通过学习文本中的特征和模式来识别实体对之间的关系。深度学习方法如BiLSTM-Attention等，能够自动提取文本的上下文信息和语义特征，显著提高了关系抽取的准确性和鲁棒性。近年来，基于预训练语言模型的方法在关系抽取任务中取得了显著的性能提升，成为研究和应用的主流方向。

让我们来试试deepseek-r1对复杂逻辑关系的把握究竟如何吧。比如，我们输入prompt，让它来分析分析曹禺的《雷雨》中究竟有哪些人物关系（出了名的小妈文学）：

```
user-prompt:
**角色**：假如你是一个高性能文本分析工具，你现在所做的任务是**关系抽取**。
**任务定义**：你需要抽取文本当中实体之间的关系，尤其是人物之间的关系，对于分布在多个分句之间的任务可能存在隐式的关系也希望你能把它们抽取或推理出来。直接结构化地按下面的抽取格式给出输入所包含的所有三元组即可。
**抽取格式**：实体对三元组，即(实体1, 关系类型, 实体2)，表示实体1和实体2之间的关系类型。
> **输入文本**：在曹禺的《雷雨》中，周朴园与蘩漪是夫妻但关系紧张，周朴园还与侍萍有旧情，生下周萍和鲁大海。周萍与蘩漪有不伦关系，后爱上了四凤，却不知她是自己同母异父的妹妹。鲁大海是工人领袖，与周朴园对立。周冲是周朴园与蘩漪的儿子，暗恋四凤。侍萍与鲁贵结婚，生下四凤。这些人物关系构成了剧中复杂的家庭与情感纠葛。
```

![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215184448996-136763445.png)
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215184502611-1279965871.png)

```json
[
    ("周朴园", "夫妻", "蘩漪"),
    ("周朴园", "有旧情", "侍萍"),
    ("周朴园", "父亲", "周萍"),
    ("周朴园", "父亲", "鲁大海"),
    ("周朴园", "父亲", "周冲"),
    ("侍萍", "母亲", "周萍"),
    ("侍萍", "母亲", "鲁大海"),
    ("侍萍", "配偶", "鲁贵"),
    ("蘩漪", "母亲", "周冲"),
    ("周萍", "不伦关系", "蘩漪"),
    ("周萍", "恋人", "四凤"),
    ("周萍", "同母异父的妹妹", "四凤"),
    ("鲁大海", "对立", "周朴园"),
    ("周冲", "暗恋", "四凤"),
    ("鲁贵", "父亲", "四凤"),
    ("四凤", "同母异父的哥哥", "周萍"),
    ("鲁贵", "配偶", "侍萍")
]
```

**说明**：  

1. **隐式关系推理**：  
   - 根据"侍萍与鲁贵结婚，生下四凤"，推导出鲁贵是四凤父亲、侍萍是四凤母亲，并补充反向关系（如"同母异父的哥哥"）。  
   - 通过"生下周萍和鲁大海"与"侍萍与鲁贵结婚"的时空错位，推断周萍/鲁大海与四凤的同母异父兄妹关系。  

2. **复合关系处理**：  
   - "关系紧张"作为夫妻关系的属性，未单独列为关系类型  
   - "工人领袖"为实体属性而非关系，未作抽取  

3. **跨句关联**：  
   - 整合"周朴园与侍萍有旧情"和"生下..."两处信息，建立父子/母子关系链  
   - 通过"周冲是周朴园与蘩漪的儿子"补全家族图谱分支

可以看到，deepseek-r1虽然难以把所有的人物逻辑都推出来（真全推出来了那问题就大了），但是它对于复杂语义逻辑的关系推理和隐式关系的把握还是比较准确的。它还知道怎么思考。


## 3.2.4 知识图谱

从实体关系到知识图谱，只是一小步，但是是人类的一大步。知识的表示实际上就可以按照`(实体1，关系，实体2)`这种三元组的方式组织。不同的实体之间存在复杂的关系，就构成了一幅网络图。

> 在计算机科学中，Network、Graph、Image、Figure、Chart都被翻译成了“图”，但是这五个完全是不同的东西！

比如说，一个知识图谱就可以表示成这样的图示：
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215190053221-2071421818.png)

**知识工程**与**专家系统**的概念其实早在七十年代就有了。在机器学习没诞生之前，研究人工智能的老师们都是在做专家系统、做知识库，他们认为那就是当时的人工智能。但是符号主义的东西更多的还是规则，然后对知识的凝练其实还是比较局限在手工阶段而非自动化知识抽取与知识表示的阶段。而现在，在有了自然语言处理以后，知识图谱更像是大模型的知识基座，未来将LLM与KG结合的研究将会越来越多。

知识图谱作为人工智能认知世界的"数字罗盘"，正在重塑人类组织知识的范式。其核心价值不仅在于构建实体关系的网络拓扑，更在于通过语义关联揭示知识的内在逻辑。在技术实现层面，知识图谱的构建犹如在认知宇宙中编织星图——需要自然语言处理技术从非结构化文本中提取实体与关系，运用图计算算法建立多维关联，借助知识表示学习将离散的语义节点转化为高维向量空间中的连续表达。这种从符号逻辑到向量空间的跨越，使得机器不仅能够存储莎士比亚戏剧的人物关系，更能理解哈姆雷特与奥菲莉亚的情感张力如何映射到现代社会的伦理困境。

当前知识图谱的演进呈现出"动态生长"与"认知增强"的双螺旋特征。动态知识图谱通过实时事件流处理技术，使知识网络具备了类似生物体的新陈代谢能力，能够自动吸收新闻事件中的新实体、消解过时信息。认知增强型图谱则突破传统三元组的局限，引入事理图谱、因果图谱等新型结构，在描述"周朴园与鲁大海对立"这类事实的同时，还能推演这种对立对企业管理、家庭伦理产生的连锁影响。这种进化使得知识图谱从静态的知识库升维为具有推理能力的认知引擎。

在应用维度，知识图谱正在解构传统行业的认知边界。医疗领域，融合基因图谱与临床知识图谱的智能系统，能够从海量文献中推演出针对特定基因突变的治疗方案；金融领域，企业关系图谱通过穿透多层股权结构，可自动识别隐蔽的关联交易风险；教育领域，学科知识图谱将碎片化知识点编织成动态认知网络，为每个学习者绘制个性化的知识进化路径。这种跨界融合催生了"图谱+"的创新生态，使知识工程从实验室走向产业变革的前沿。

然而，知识图谱的发展也面临"语义深渊"的挑战。当系统试图建模《雷雨》中周萍与四凤的禁忌之恋时，简单的亲属关系标签难以承载复杂的社会伦理内涵；在表示"工人领袖"这类社会角色时，静态的属性标注无法捕捉其在劳资博弈中的动态立场。这推动着知识表示向多模态、可解释的方向进化——未来的知识图谱或将融合视觉符号、情感向量和社会学理论模型，使机器不仅能罗列人物关系表，还能理解这些关系背后的文化基因与人性光谱。

如果我们在Python中想要联动一个知识图谱，最好的图数据库工具就是neo4j。安装好以后，编写如下代码把前面《雷雨》人物关系输入进来：

```python
from py2neo import Graph, Node, Relationship

# 实体关系数据
data = [
    ("周朴园", "夫妻", "蘩漪"),
    ("周朴园", "有旧情", "侍萍"),
    ("周朴园", "父亲", "周萍"),
    ("周朴园", "父亲", "鲁大海"),
    ("周朴园", "父亲", "周冲"),
    ("侍萍", "母亲", "周萍"),
    ("侍萍", "母亲", "鲁大海"),
    ("侍萍", "配偶", "鲁贵"),
    ("蘩漪", "母亲", "周冲"),
    ("周萍", "不伦关系", "蘩漪"),
    ("周萍", "恋人", "四凤"),
    ("周萍", "同母异父的妹妹", "四凤"),
    ("鲁大海", "对立", "周朴园"),
    ("周冲", "暗恋", "四凤"),
    ("鲁贵", "父亲", "四凤"),
    ("四凤", "同母异父的哥哥", "周萍"),
    ("鲁贵", "配偶", "侍萍")
]

# 连接到 Neo4j 数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "your_password"))

# 创建节点和关系
added_nodes = {}

for item in data:
    subject, relation, object = item
    
    # 创建或获取节点
    if subject not in added_nodes:
        node_sub = Node("Person", name=subject)
        graph.create(node_sub)
        added_nodes[subject] = node_sub
    
    if object not in added_nodes:
        node_obj = Node("Person", name=object)
        graph.create(node_obj)
        added_nodes[object] = node_obj
    
    # 创建关系
    rel = Relationship(added_nodes[subject], relation, added_nodes[object])
    graph.create(rel)

# 打印节点和关系
print("节点:")
for node in added_nodes.values():
    print(node["name"])

print("\n关系:")
for item in data:
    print(f"{item[0]} {item[1]} {item[2]}")
```

![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215194008384-935915051.png)


# 3.3 机器翻译

## 3.3.1 机器翻译

机器翻译（Machine Translation, MT）是指利用计算机技术将一种自然语言（源语言）的文本自动转换为另一种自然语言（目标语言）的文本。这一任务是自然语言处理领域最具挑战性的任务之一，旨在打破语言障碍，促进跨语言交流。机器翻译的发展经历了从基于规则的方法到基于统计的方法，再到如今基于深度学习的神经机器翻译（Neural Machine Translation, NMT）的多个阶段。早期的基于规则的机器翻译依赖于手工编写的语言规则，难以处理语言的多样性和复杂性。基于统计的机器翻译通过分析双语语料库，学习源语言和目标语言之间的统计模型，性能有了显著提升。近年来，深度学习技术的兴起使得神经机器翻译成为主流方法，通过构建深度神经网络模型，能够自动学习源语言和目标语言之间的映射关系，生成更加流畅和准确的翻译结果。

机器翻译起始就是从源语言到目标语言的一个变换。从不同视角看机器翻译，它也有不同的解释。

早期的机器翻译系统更多的是基于规则。比如，给定源语言句子，我们对词性进行标注、逐词翻译后，对比源语言与目标语言的语法句法，按照句法规则进行解析翻译排序，最后得到译文。这种方法对比较规范的书面语还能有一些作用，可以帮助翻译一些常用语。但在面对不常见的语法现象、新的语法变化时，这种人为规则的方式工作量大，显得捉襟见肘。
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215194103830-1021559422.png)
从信号处理的角度，机器学习工作者把机器翻译问题看作一个信号变换的问题。它不过是从源域到变换域了嘛，语义信息是没变的。那么既然我们做了这样一个变换，我们不妨用贝叶斯统计的视角看待统计语言模型下的翻译：即给定源语言句子的条件下生成这一目标语言句子的概率。通过贝叶斯公式变换，分母项因为是给定的所以是个常数。我们要优化的就是两个事情：第一，目标语言的句子生成概率$P(T)$，这是语言模型的部分；第二，是目标句子如果正确，它和源语言句子的对应概率$P(T|S)$，这是翻译模型的部分。由此，拉开了统计机器翻译的大门，HMM等模型成为比较好的机器翻译方案。
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215194117919-264498966.png)
深度学习的到来为机器翻译注入了新的活力。我们开始考虑用统一的神经网络架构进行机器翻译（NMT）工作，其中，Transformer模型的Encoder-Decoder架构获得了重大成功。并且，对一些隐喻等不太常见的语法现象，它居然也能翻译出来。
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215194132371-1658914544.png)

下面的demo是用Transformer的预训练权重进行机器翻译任务的代码：

```python
from transformers import TranslationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# 加载预训练的翻译模型（英文→中文）
model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 创建翻译pipeline
translator = TranslationPipeline(
    model=model,
    tokenizer=tokenizer,
    framework="pt",
)

# 输入文本
input_text = "Hello world! This is a simple translation example using transformers."

# 翻译文本
translation = translator(input_text, max_length=40)

# 输出结果
print(f"原始文本: {input_text}")
print(f"翻译结果: {translation[0]['translation_text']}")
```

机器翻译任务的评价指标主要用于衡量翻译结果的质量，常用的指标包括BLEU（Bilingual Evaluation Understudy）、ROUGE（Recall-Oriented Understudy for Gisting Evaluation）、METEOR（Metric for Evaluation of Translation with Explicit Ordering）和TER（Translation Edit Rate）等。BLEU是最常用的评价指标之一，通过比较翻译结果与参考译文的n-gram重叠情况来计算得分，得分越高表示翻译质量越好。ROUGE主要用于评估信息召回率，常用于抽取式文本摘要和机器翻译任务中。METEOR综合考虑了词形、词义、词序等多种因素，能够更全面地评估翻译结果的质量。TER通过计算将翻译结果编辑为参考译文所需的最小编辑次数来衡量翻译质量，编辑次数越少表示翻译结果越接近参考译文。这些评价指标从不同角度反映了机器翻译系统的性能，为模型的优化和改进提供了重要参考。

**BLEU（Bilingual Evaluation Understudy）**  
BLEU指标通过n-gram精度来衡量翻译质量，其计算公式为：  
$$
BLEU = BP \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)
$$
其中，$BP$为简洁惩罚因子（Brevity Penalty），用于惩罚过短的翻译结果：  
$$
BP = \begin{cases} 
1 & \text{if } c > r \\
e^{(1-r/c)} & \text{if } c \leq r 
\end{cases}
$$
$c$为候选译文长度，$r$为参考译文长度。$p_n$表示n-gram精度，$w_n$为各阶n-gram的权重，通常取均匀权重。

**ROUGE（Recall-Oriented Understudy for Gisting Evaluation）**  
ROUGE系列指标中最常用的是ROUGE-N，其计算公式为：  
$$
ROUGE\text{-}N = \frac{\sum_{S\in \{Ref\}} \sum_{gram_n \in S} Count_{match}(gram_n)}{\sum_{S\in \{Ref\}} \sum_{gram_n \in S} Count(gram_n)}
$$
其中，$Count_{match}(gram_n)$表示候选译文与参考译文共现的n-gram数，$Count(gram_n)$为参考译文中n-gram的总数。

**METEOR（Metric for Evaluation of Translation with Explicit Ordering）**  
METEOR指标的计算较为复杂，其核心公式为：  
$$
METEOR = (1 - \gamma \cdot \text{frag}^{\beta}) \cdot \frac{P \cdot R}{\alpha \cdot P + (1-\alpha) \cdot R}
$$
其中，$P$为准确率，$R$为召回率，$\alpha$为准确率权重系数。$frag$表示分块碎片度，$\beta$和$\gamma$为调节参数。

**TER（Translation Edit Rate）**  
TER通过编辑距离来衡量翻译质量，其计算公式为：  
$$
TER = \frac{E}{L}
$$
其中，$E$为将候选译文编辑为参考译文所需的最少编辑操作次数，$L$为参考译文的长度。编辑操作包括插入、删除、替换和移位。


## 3.3.2 低资源语言面临的困境

全世界的语言有这么多种，但是真正用的人多的恐怕两只手就可以数过来：中文、英文、法语、德语、西班牙语、葡萄牙语、意大利语、俄语、日语、韩语、阿拉伯语等。很多语言是缺少互联网上公开的语料、也缺少用这些语言记载的文献资料的。比如印第安语、爪哇语等。

缺少语料对于模型来说是一个问题，一些模型在大国语言上表现更好，但对于一些用的人比较少的语言就很容易乱回复（因为语言模型对词和概率的把握并不准确，样本量太小了）。这也是个不公平的问题。就很简单的例子，中国国内很多少数民族的语言是缺少文献记载支撑的，因此为了考虑到这些少数民族的文化习惯，必须要对他们的本民族语言能力进行提升，尽可能缩小与汉语之间的gap。

还有一个非常现实的问题在于低资源语言容易遭遇越狱攻击。这一点是我的同学告诉我的。比如，我们打开deepseek，不带长思考和联网回答，问他怎么造原子弹：
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215200000022-1643631185.png)

好，我们换一个低资源语言去问。拿百度翻译把这个句子翻译一下：
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215200104374-1255821522.png)
可以看到，deepseek会调用工具了，并且装傻充愣，说明这种低资源越狱的现象被加强了。
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215200154108-1442349200.png)
如果使用R1长思考，会发生什么？
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215200632243-1189053606.png)
你会发现这个时候回答的就是个稀烂玩意：
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215200652476-689150571.png)

> 其实我在去年12月测试的时候，kimi的低资源语言能力就比较弱，用爪哇语很快就可以攻破防线。但是现在就不行了。应该是被加强过。
> ![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215200748037-424096848.png)

一种提升低资源语言能力的有效方法是增加双语约束，也就是把两种语言的语义进行绑定，用高资源语言的语言能力扶着低资源模型。但即使如此，低资源语言由于数据问题，其语言模型能力还是会弱一些。
![image](https://img2024.cnblogs.com/blog/3349206/202502/3349206-20250215201017319-1518614077.png)
