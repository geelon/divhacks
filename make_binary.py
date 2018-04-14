
import pandas as pd

train_data = pd.read_csv('MNIST_train.csv', index_col='ID')
train_data_unlabeled = train_data.drop(' Label', axis=1)

percent_correct = train_data
percent_correct['percent_correct'] = train_data_unlabeled.apply(lambda row: row.sum() / 21,1)

average_accuracy = pd.DataFrame(index=['percent_correct'])
for i in range(10):
    average_accuracy[i] = percent_correct.loc[percent_correct[' Label'] == i]['percent_correct'].mean()


import os
import sys

dic={}
arguments=[]

def readable(path):
    return os.access(path, os.R_OK)

def parse_args(args):
    end_args = False
    var_arg  = None
    for a in args:
        arguments.append(a)

def check_read(files):
    for f in files:
        if not readable(f):
            print("mkdic: " + f + " is not readable.")
            sys.exit(1)

def mkset(filename,threshold):
    k = open(filename,'w')
    threshold = float(threshold)
    for x in range(60000):
        if train_data['percent_correct'][x] > threshold:
            k.write('{},{}\n'.format(x,train_data['percent_correct'][x]))
       
    k.close()
    return 

def main(args):
    parse_args(args)
    #check_read(files)
    mkset(arguments[0],arguments[1])

if __name__ == '__main__':
    main(sys.argv[1:])
