import os
import glob
import csv
import re
def remove_pun_num(str):
    str = re.sub(r'[0-9.()\t]', '', str)
    return str
def read_all():
    cwd = os.getcwd()
    values = []
    for filename in glob.glob(os.path.join(cwd, '*/*_TRANSCRIPT.csv')):
        with open(filename, 'r') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                if row['speaker'] == 'Participant':
                    values.append(remove_pun_num(row['value']))
    with open('TRANSCRIPT_all.txt', 'w') as file:
        for value in values:
            file.write(value + '\n')
    #return values
    print(len(values))
read_all()