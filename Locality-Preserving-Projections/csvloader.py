# Simple script to load data from CSV files
# (c) Duong Nguyen
# Email: nguyen@sg.cs.titech.ac.jp

import csv
import numpy as np

def readData(filename):
    """ Read data from CSV file."""
    rows = []
    with open(filename, 'rt') as fin:
        reader = csv.reader(fin)
        cnt = 0
        for row in reader:
            if cnt == 0:
                header = row[0].split(';')
            else:
                rows.append(map(float, row[0].strip().split(';')))
            cnt += 1
            
    return np.asarray(rows), header, cnt

def readAll(fnames):
    data = []
    tmp, header, _ = readData(fnames[0])
    data.append(tmp)
    for fname in fnames[1:]:
        tmp, _, _ = readData(fname)
        data.append(tmp)
        
    return data, header 
 
def main():
    return readAll(['winequality-red.csv', 'winequality-white.csv'])

if __name__ == '__main__':
    data, header = main()
    