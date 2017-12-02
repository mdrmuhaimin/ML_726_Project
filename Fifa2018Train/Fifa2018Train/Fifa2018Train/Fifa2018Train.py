import math
import numpy as np
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

# converting string values columns to several boolean values colunms
def process_columns_for_work_rate_att_work_rate_def(data, columns):
    for c in columns:
        work_rate_columns = {'work_rate_' + c + '_high':[],'work_rate_' + c + '_medium':[],'work_rate_' + c + '_low':[]}
        for e in data['work_rate_' + c]:
            if e == 'High':
                work_rate_columns['work_rate_' + c + '_high'].append(1)
            else:
                work_rate_columns['work_rate_' + c + '_high'].append(0)
            if e == 'Medium':
                work_rate_columns['work_rate_' + c + '_medium'].append(1)
            else:
                work_rate_columns['work_rate_' + c + '_medium'].append(0)
            if e == 'Low':
                work_rate_columns['work_rate_' + c + '_low'].append(1)
            else:
                work_rate_columns['work_rate_' + c + '_low'].append(0)

        data['work_rate_' + c + '_high'] = work_rate_columns['work_rate_' + c + '_high']
        data['work_rate_' + c + '_medium'] = work_rate_columns['work_rate_' + c + '_medium']
        data['work_rate_' + c + '_low'] = work_rate_columns['work_rate_' + c + '_low']

        del data['work_rate_' + c]
            
    return data

# standartizing data except binary columns and adding intersept column
def subtract_mean_and_divide_by_deviation(data):
    column_names = list(data)

    for c in range(len(column_names) - 6):
        data[column_names[c]]-=data[column_names[c]].mean()
        data[column_names[c]]/=data[column_names[c]].std()


    return data


data = pd.read_csv('shuffled_filtered_fifa2018_players.csv') 
#data=data.reindex(np.random.permutation(data.index))
#data.to_csv('shuffled_filtered_fifa2018_players.csv')
del data['ID']


train_data = data.ix[:399,:]
train_overall_column = data.ix[:399,'overall']
train_data_with_added_columns = process_columns_for_work_rate_att_work_rate_def(train_data,['att','def'])

del train_data_with_added_columns['overall']
normalized_train_data = subtract_mean_and_divide_by_deviation(train_data_with_added_columns)
#
test_data = data.ix[400:,:]
test_overall_column = data.ix[400:,'overall']
test_data_with_added_columns = process_columns_for_work_rate_att_work_rate_def(test_data,['att','def'])

del test_data_with_added_columns['overall']
normalized_test_data = subtract_mean_and_divide_by_deviation(test_data_with_added_columns)


X_train = normalized_train_data.as_matrix()
Y_train = train_overall_column.as_matrix()

X_test = test_data_with_added_columns.as_matrix()
Y_test = test_overall_column.as_matrix()

print(X_train)

model = Sequential()

model.add(Dense(units=200, activation='relu', input_shape=(17,), use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'))
#model.add(Dense(units=36, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model.add(Dense(units=1, activation='linear', use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'))


model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.SGD(lr=0.031,momentum=0.1))

model.fit(x=X_train, y=Y_train, batch_size=400, epochs=300)

print(model.evaluate(x=X_test, y=Y_test))

#print(train_data)
#print(train_data_with_added_columns)
#print(train_overall_column)
#print (test_data)
#print(test_overall_column)