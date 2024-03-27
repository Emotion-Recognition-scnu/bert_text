import os
import glob
import csv
import re


def remove_pun_num(str):
    str = re.sub(r'[0-9.()\t]', '', str)
    return str


def train_index():
    train_index = []
    with open('train_split_Depression_AVEC2017.csv', 'r') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            train_index.append(row['Participant_ID'])
    return train_index


def test_index():
    test_index = []
    with open('test_split_Depression_AVEC2017.csv', 'r') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            test_index.append(row['participant_ID'])
    return test_index


def train_label():
    train_labell = []
    with open('train_split_Depression_AVEC2017.csv', 'r') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            train_labell.append(row['PHQ8_Binary'])
    with open('train_label.txt', 'w') as file:
        file.write(str(train_labell))
    return train_labell


def test_label():
    test_labell = []
    with open('full_test_split.csv', 'r') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            test_labell.append(row['PHQ_Binary'])
    with open('test_label.txt', 'w') as file:
        file.write(str(test_labell))
    return test_labell


testlabels = test_label()
trainlabels = train_label()
testindex = test_index()
trainindex = train_index()


def read_all():
    cwd = os.getcwd()
    global trainindex
    value1 = []
    test1 = []
    for filename in glob.glob(os.path.join(cwd, '*/*_TRANSCRIPT.csv')):
        values = []
        test = []
        with open(filename, 'r') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                if row['speaker'] == 'Participant':
                    if filename[-18:-15] in trainindex:
                        values.append(remove_pun_num(row['value']))
                    if filename[-18:-15] in testindex:
                        test.append(remove_pun_num(row['value']))
                    else:
                        continue
        if len(test) != 0:
            test1.append(test)
        if len(values) != 0:
            value1.append(values)
    with open('train.txt', 'w') as file:
        file.write(str(value1))
    with open('test.txt', 'w') as file:
        file.write(str(test1))
    print(len(test1))
    print(len(testlabels))
    print(len(value1))
    print(len(trainlabels))


read_all()
