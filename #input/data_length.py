import glob
import os
import pandas as pd

file_list_test  = glob.glob('./test/*.csv')
file_list_train = glob.glob('./train/*.csv')

test_length = 0
for f in file_list_test:
    df = pd.read_csv(f, header=4)
    test_length += len(df)

train_length = 0
for f in file_list_train:
    df = pd.read_csv(f, header=4)
    train_length += len(df)

with open('check.txt', 'w') as f:
    f.write(f'test  : {test_length}\n')
    f.write(f'train : {train_length}')