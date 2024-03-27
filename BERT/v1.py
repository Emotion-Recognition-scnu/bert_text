import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import os

# 如果是第一次使用NLTK，可能需要下载stopwords和punkt资源
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):  # 文本清理函数
    # 转化为小写
    text = text.lower()

    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)

    # 去除停用词和语气词
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]

    # 词形还原
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # 返回处理后的文本
    return ' '.join(stemmed_words)


def read_text(folder_path):
    # 初始化一个空的列表来存储所有文本
    all_texts = []

    # 遍历文件夹内的每个文件
    for file_name in os.listdir(folder_path):
        # 检查文件扩展名是否为.csv
        if file_name.endswith('.csv'):
            # 构造完整的文件路径
            file_path = os.path.join(folder_path, file_name)
            # 读取CSV文件的第一列数据，假设没有标题行
            data = pd.read_csv(file_path, header=0, sep='\t', usecols=[3])
            # 将读取的数据追加到列表中
            all_texts.append(data)

    # 将列表中的所有数据合并成一个DataFrame
    all_texts_df = pd.concat(all_texts, ignore_index=True)

    return all_texts_df


def bert_tokenizer(df):  # bert词向量化
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    #texts = df['cleaned_text'].tolist()
    texts = df
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    return last_hidden_states


def distilbert_tokenizer(df):  # DistilBERT词向量化
    # 初始化分词器和DistilBERT模型
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # 将DataFrame中的文本转换为列表
    texts = df
    # 对文本进行分词处理
    inputs = [tokenizer(s, padding='max_length', truncation=True, max_length=512, return_tensors='pt') for s in df]


    # 使用DistilBERT模型获取词向量
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    # 返回最后一层的隐藏状态
    return last_hidden_states


if __name__ == '__main__':
    # 文件夹路径，替换成您存储CSV文件的文件夹路径
    #folder_path = r"..\Data"

    # 假设您有一个包含文本的DataFrame
    with open ('..\\Data\\TRANSCRIPT_all.txt', 'r') as file:
        values = file.read()
    df = values.split('\n')

    # 输出合并后的数据前几行，确认是否正确
    print(df)

    # 应用预处理函数
    #df['cleaned_text'] = df['value'].apply(preprocess_text)
    #print(df)

    # 应用向量化函数
    # last_hidden_states1 = bert_tokenizer(df)
    # print(last_hidden_states1)

    last_hidden_states2 = bert_tokenizer(df)
    with open('..\\Data\\last_hidden_states2.txt', 'w') as file:
        file.write(str(last_hidden_states2))
    print(last_hidden_states2)
