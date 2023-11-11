'''

@author: Steven (Jiaxun) Tang <jtang@umass.edu>
'''
import numpy as np
import pandas
import os
import matplotlib.pyplot as plt

DATA_PATH = '/media/umass/datasystem/steven/Downloads/SimpleApps'

# Test the std of innerTime. This is to verify the integrity of
print('Inner time')
for folder in os.listdir(DATA_PATH):
    print(folder, end='\t')
    for i in range(1, 9):
        # print(os.path.join(DATA_PATH, folder, '%d.csv' % (i)))
        df = pandas.read_csv(os.path.join(DATA_PATH, folder, '%d.csv' % (i)))
        # print(df.describe(include='all'))
        print(np.std(df['innerTime']), end='\t')
    print()

print('Outer time - Inner time')
for folder in os.listdir(DATA_PATH):
    print(folder, end='\t')
    for i in range(1, 9):
        # print(os.path.join(DATA_PATH, folder, '%d.csv' % (i)))
        df = pandas.read_csv(os.path.join(DATA_PATH, folder, '%d.csv' % (i)))
        # print(df.describe(include='all'))
        print(np.std(df['innerTime'] - df[' outerTime']), end='\t')  # One extra space in outertime
    print()

df = pandas.read_csv(os.path.join(DATA_PATH, 'Prehook', '5.csv'))
plt.scatter(range(len(df['innerTime'])), df['innerTime'])
plt.title('Prehook 5.csv')
plt.show()
print(df.describe())
