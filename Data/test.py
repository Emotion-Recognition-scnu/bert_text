def read_data(file_name):
    with open(file_name, 'r') as file:
        values = file.read()
    df = values.split('], [')
    df[0] = df[0][1:]
    df[-1] = df[-1][:-1]
    return df


# 读取并处理训练数据
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
print(df_train2[0])



print(len(df_train))