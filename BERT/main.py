from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
import pickle

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# 读取训练数据
with open('train.txt', 'r') as file:
    values = file.read()
df = values.split('], [')

df[0] = df[0][1:]
df[-1] = df[-1][:-1]

inputs_train = tokenizer(df, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs_train)
    inputs_train = outputs.last_hidden_state
#inputs_train = torch.tensor(last_hidden_states_train)
#print(last_hidden_states.size())


# 读取测试数据
with open('test.txt', 'r') as file:
    values = file.read()
df2 = values.split('], [')
df2[0] = df2[0][1:]
df2[-1] = df2[-1][:-1]
inputs_test = tokenizer(df2, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs_test)
    last_hidden_states_test = outputs.last_hidden_state
inputs_test = torch.tensor(last_hidden_states_test)
#print(last_hidden_states.size())


# 读取训练标签
with open('train_label.txt', 'r') as file:
    values = file.read()
train_labels = values.split(', ')
train_labels[0] = train_labels[0][2:]
train_labels[-1] = train_labels[-1][:-2]
train_labels = torch.tensor(train_labels)


# 训练
hidden_states = inputs_train
hidden_states = hidden_states.long()
hidden_states = hidden_states.squeeze(-1)


labels = torch.tensor(list(map(int, train_labels))).long()
labels = labels.squeeze(-1)


dataset = TensorDataset(inputs_train, labels)
dataloader = DataLoader(dataset, batch_size=32)

# 创建模型和优化器
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
optimizer = AdamW(model.parameters(), lr=1e-5)

# 进行训练
for epoch in range(10):
    for batch in dataloader: #
        batch_hidden_states, batch_labels = batch

        # 将数据输入到模型中，获取输出结果
        outputs = model(batch_hidden_states)
        loss = outputs.loss

        # 使用优化器更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

