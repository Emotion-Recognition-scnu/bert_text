from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 初始化tokenizer和model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


def read_data(file_name):
    with open(file_name, 'r') as file:
        values = file.read()
    df = values.split('], [')
    df[0] = df[0][1:]
    df[-1] = df[-1][:-1]
    return df


# 读取并处理训练数据
df_train = read_data('train.txt')
inputs_train = tokenizer(df_train, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

with torch.no_grad():
    outputs_train = model(**inputs_train)

# 读取并处理测试数据
df_test = read_data('test.txt')
inputs_test = tokenizer(df_test, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

with torch.no_grad():
    outputs_test = model(**inputs_test)


# 处理训练标签
label = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '0', '0', '1', '0', '0', '0', '1', '0', '0',
         '1', '1', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '1', '0', '0', '0',
         '0', '0', '0', '0', '1', '0', '0', '1', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1', '0', '1',
         '1', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '1', '0',
         '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
train_labels = []
for i in label:
    train_labels.append([int(i)])
labels = torch.tensor(train_labels)

# 准备DataLoader
dataset = TensorDataset(inputs_train['input_ids'], inputs_train['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=8)  # 调整batch_size以适应内存


# 定义模型
class MyModel(nn.Module):
    def __init__(self, num_labels):
        super(MyModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits


# 实例化模型和优化器
model = MyModel(num_labels=2)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 训练模型
for epoch in range(10):  # 调整epochs根据需要
    for batch in dataloader:
        batch_input_ids, batch_attention_mask, batch_labels = batch
        # print(batch_input_ids.shape)
        # print(batch_attention_mask.shape)
        # print(batch_labels.shape)
        logits = model(batch_input_ids, batch_attention_mask)
        # print(logits.shape)
        loss = nn.CrossEntropyLoss()(logits, batch_labels.squeeze(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
