from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
import pickle
import torch
from torch import nn
from transformers import DistilBertModel
import torch.nn as nn
import torch.optim as optim


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# 读取训练数据
with open('train.txt', 'r') as file:
    values = file.read()
df = values.split('], [')

df[0] = df[0][1:]
df[-1] = df[-1][:-1]
for i in df:
    if len(i) > 512:
        i = i[:512]
    if len(i) < 512:
        i.append('0'*(512-len(i)))
# print(len(df))

inputs_train = tokenizer(df, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
inputs_ids = inputs_train['input_ids'].long()
print(len(inputs_ids))
attention_mask = inputs_train['attention_mask'].long()
with torch.no_grad():
    outputs = model(**inputs_train)
    last_hidden_states_train = outputs.last_hidden_state


inputs_train = last_hidden_states_train
inputs_train = inputs_train.clamp_min_(0)



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
# print(last_hidden_states.size())


# 读取训练标签

<<<<<<< HEAD
label = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '0', '0', '1', '0', '0', '0', '1', '0', '0',
         '1', '1', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '1', '0', '0', '0',
         '0', '0', '0', '0', '1', '0', '0', '1', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1', '0', '1',
         '1', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '1', '0',
         '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
train_labels = []
for i in label:
    train_labels.append([int(i)])
=======
    # 将DataFrame中的文本转换为列表
    texts = df
    # 对文本进行分词处理
    inputs = [tokenizer(s, padding='max_length', truncation=True, max_length=512, return_tensors='pt') for s in df]

>>>>>>> af37a18b0e71a7655d1459b638476d78801038f0

embedding_layer = torch.nn.Embedding(num_embeddings=model.config.vocab_size, embedding_dim=512)
embedded = embedding_layer(inputs_ids)
embedded = embedded.clamp_min_(0)
embedded = embedded.squeeze(-1)

# 训练
# hidden_states = inputs_train
# hidden_states = inputs_train.to(torch.int64)
# hidden_states = hidden_states.squeeze(-1)
hidden_states = embedded.long()

labels = torch.tensor(train_labels)

# 创建TensorDataset和DataLoader
dataset = TensorDataset(hidden_states,attention_mask,labels)
dataloader = DataLoader(dataset, batch_size=107)  # 你可以根据需要调整batch_size

# 创建模型和优化器

class MyModel(nn.Module):
    def __init__(self, num_labels):
        super(MyModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        logits = self.classifier(last_hidden_state)
        return logits


# 创建模型和优化器
model = MyModel(num_labels=2)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 进行训练
for epoch in range(10):
    for batch in dataloader:
        batch_input_ids, batch_attention_mask, batch_labels = batch
        print(batch_input_ids.shape)
        print(batch_attention_mask.shape)
        print(batch_labels.shape)

        logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        print(logits.shape)


        loss = nn.CrossEntropyLoss()(logits, batch_labels)

        # 使用优化器更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()