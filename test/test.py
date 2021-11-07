from pandas import Series,DataFrame
import  pandas as pd
dictionary = {
    'state': ['one', 'two', 'three', 'four', 'five'],
    'year': ['one', 'two', 'three', 'four', 'five'],
    'pop': ['one', 'two', 'three', 'four', 'five']}
df1 = DataFrame(dictionary)
df1['new_add'] = [7, 4, 5, 8, 2]
print(df1)