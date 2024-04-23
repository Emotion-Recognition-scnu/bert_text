import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel


# 模型结构定义（按你的原始代码）
class MyModel(nn.Module):
    def __init__(self, num_labels):
        super(MyModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # 假设分类层是接在最后一层hidden state的第一个token (如CLS)
        return logits


# 数据读取和处理
def read_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        values = file.read()
    df = values.split('], [')
    df[0] = df[0][1:]  # 移除首字符 '['
    df[-1] = df[-1][:-1]  # 移除末字符 ']'
    return df


def preprocess_data(data_list, tokenizer):
    processed_texts = []
    for item in data_list:
        item_clean = item.replace("'", "").replace("\"", "").replace(",", "")
        processed_texts.append(item_clean)
    inputs = tokenizer(processed_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return inputs


# 主执行函数
def predict(model_path, file_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化tokenizer和模型
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = MyModel(num_labels=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # 读取并处理数据
    train_data = read_data(file_path)
    inputs_train = preprocess_data(train_data, tokenizer)
    input_ids = inputs_train['input_ids'].to(device)
    attention_mask = inputs_train['attention_mask'].to(device)

    # 使用DistilBert模型进行预测
    with torch.no_grad():
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_labels = torch.argmax(predictions, dim=1)
        print(f'Predicted labels: {predicted_labels.tolist()}')

    return predicted_labels.tolist()
