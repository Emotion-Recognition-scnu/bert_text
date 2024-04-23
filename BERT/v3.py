import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel

print(torch.__version__)
print(torch.cuda.is_available())


# 初始化tokenizer和model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def read_data(file_name):
    with open(file_name, 'r') as file:
        values = file.read()
    df = values.split('], [')
    df[0] = df[0][1:]
    df[-1] = df[-1][:-1]
    return df


# 读取并处理训练数据
# df_train = read_data('train.txt')

df_train2 = []
df_train = read_data('train.txt')
for str_l in df_train:
    str_e = ''
    for str in str_l:
        str2 = str.replace('\'', '')
        str3 = str2.replace('\"', '')
        str4 = str3.replace(',', '')
        str_e += str3
    df_train2.append(str_e)

inputs_train = tokenizer(df_train2, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

inputs_train = inputs_train.to(device)
with torch.no_grad():
    outputs_train = model(**inputs_train)

# 读取并处理测试数据
df_test = read_data('test.txt')

df_test2 = []
df_train = read_data('test.txt')
for str_l in df_train:
    str_e = ''
    for str in str_l:
        str2 = str.replace('\'', '')
        str3 = str2.replace('\"', '')
        str4 = str3.replace(',', '')
        str_e += str3
    df_test2.append(str_e)

inputs_test = tokenizer(df_test2, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
inputs_test = inputs_test.to(device)
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
labels = torch.tensor(train_labels).cuda()

# 准备DataLoader
dataset = TensorDataset(inputs_train['input_ids'], inputs_train['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=20)  # 调整batch_size以适应内存


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
model = model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 注意：这里的lr是预热后的目标学习率

# 预热步数
warmup_steps = 500

# 预热期的学习率调整函数
lambda1 = lambda epoch: epoch / warmup_steps if epoch < warmup_steps else 1

# 创建预热调度器
scheduler_warmup = LambdaLR(optimizer, lr_lambda=lambda1)

total_epochs = 75
# 创建余弦退火调度器
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_steps)

# 训练模型
for epoch in range(total_epochs):  # 调整epochs根据需要
    for batch in dataloader:
        batch_input_ids, batch_attention_mask, batch_labels = batch

        # Move the batch tensors to the same device as the model
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_input_ids, batch_attention_mask)
        loss = nn.CrossEntropyLoss()(logits, batch_labels.squeeze(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 选择调度器
        if epoch < 15:
            scheduler_warmup.step()
        else:
            scheduler_cosine.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{total_epochs}, Current LR: {current_lr}")


# 测试模型
# 准备测试标签
test_labels = ['0', '0', '0', '1', '1', '1', '0', '0', '0', '1',
               '0', '1', '0', '1', '1', '0', '1', '0', '0', '1',
               '0', '0', '0', '1', '0', '0', '1', '0', '1', '0',
               '0', '0', '0', '0', '0', '0', '0', '1', '1', '0',
               '0', '0', '0', '0', '0', '0', '0']
test_labels = [[int(i)] for i in test_labels]
test_labels = torch.tensor(test_labels).cuda()

# 准备测试数据的DataLoader
test_dataset = TensorDataset(inputs_test['input_ids'], inputs_test['attention_mask'], test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=1)

model.eval()

correct_predictions = 0
p_l = []

# 遍历测试数据
for batch in test_dataloader:
    batch_input_ids, batch_attention_mask, batch_labels = batch

    # 使用模型进行预测
    with torch.no_grad():
        logits = model(batch_input_ids, batch_attention_mask)

    # 获取预测结果中概率最高的类别
    _, predicted = torch.max(logits, dim=1)
    predicted_l = predicted.tolist()
    print(predicted)
    print(predicted_l)
    for i in predicted_l:
        p_l.append(int(i))

    # 更新正确预测的数量
    correct_predictions += (predicted == batch_labels).sum().item()

# 计算准确率
accuracy = correct_predictions / len(test_dataloader.dataset)
c = 0
tp = fp = tn = fn = 0

for i in range(47):
    if p_l[i] == int(test_labels[i]):
        c += 1
    if p_l[i] == 0 and int(test_labels[i]) == 0:
        tp += 1
    if p_l[i] == 1 and int(test_labels[i]) == 0:
        fn += 1
    if p_l[i] == 0 and int(test_labels[i]) == 1:
        fp += 1
    if p_l[i] == 1 and int(test_labels[i]) == 1:
        tn += 1
accuracy = c / len(test_labels)
print(tp, fp, tn, fn)
print(f"Accuracy on test set: {accuracy}")
r = tp / (tp + fn)
print(f"Recall: {tp / (tp + fn)}")
p = tp / (tp + fp)
print(f"F1 Score: {2 * (p * r) / (p + r)}")

# 模型保存
# 定义保存路径
save_path = 'Model-Completed.pth'

# 保存模型和优化器的状态字典
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)