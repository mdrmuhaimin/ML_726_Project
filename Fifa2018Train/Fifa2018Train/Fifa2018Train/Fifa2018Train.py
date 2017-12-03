import math
import numpy as np
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

# converting string values columns to several boolean values colunms
def process_columns_for_work_rate_att_work_rate_def_features(data, columns):
    
    for c in columns:
        work_rate_columns = {'work_rate_' + c + '_high':[],'work_rate_' + c + '_medium':[],'work_rate_' + c + '_low':[]}
        for e in data[c + '_WR']:
            if e == 'high':
                work_rate_columns['work_rate_' + c + '_high'].append(1)
            else:
                work_rate_columns['work_rate_' + c + '_high'].append(0)
            if e == 'medium':
                work_rate_columns['work_rate_' + c + '_medium'].append(1)
            else:
                work_rate_columns['work_rate_' + c + '_medium'].append(0)
            if e == 'low':
                work_rate_columns['work_rate_' + c + '_low'].append(1)
            else:
                work_rate_columns['work_rate_' + c + '_low'].append(0)

        data['work_rate_' + c + '_high'] = work_rate_columns['work_rate_' + c + '_high']
        data['work_rate_' + c + '_medium'] = work_rate_columns['work_rate_' + c + '_medium']
        data['work_rate_' + c + '_low'] = work_rate_columns['work_rate_' + c + '_low']

        del data[c + '_WR']
            
    return data

def process_columns_for_work_rate_att_work_rate_def2018(data, columns):
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

def subtract_mean_and_divide_by_deviation(data,n):
    column_names = list(data)

    for c in range(len(column_names) - n):
        data[column_names[c]]-=data[column_names[c]].mean()
        data[column_names[c]]/=data[column_names[c]].std()


    return data

data2018 = pd.read_csv('shuffled_filtered_fifa2018_players.csv')
del data2018['ID']

train_data2018 = data2018.ix[:399,:]
train_overall_column2018 = data2018.ix[:399,'overall']
train_data_with_added_columns2018 = process_columns_for_work_rate_att_work_rate_def2018(train_data2018,['att','def'])

del train_data_with_added_columns2018['overall']
normalized_train_data2018 = subtract_mean_and_divide_by_deviation(train_data_with_added_columns2018,6)
#
test_data2018 = data2018.ix[400:,:]
test_overall_column2018 = data2018.ix[400:,'overall']
test_data_with_added_columns2018 = process_columns_for_work_rate_att_work_rate_def2018(test_data2018,['att','def'])

del test_data_with_added_columns2018['overall']
normalized_test_data2018 = subtract_mean_and_divide_by_deviation(test_data_with_added_columns2018,6)


X_train2018 = normalized_train_data2018.as_matrix()
Y_train2018 = train_overall_column2018.as_matrix()

X_test2018 = test_data_with_added_columns2018.as_matrix()
Y_test2018 = test_overall_column2018.as_matrix()

model2018 = Sequential()

model2018.add(Dense(units=200, activation='relu', input_shape=(17,),use_bias=True, kernel_initializer='random_uniform',bias_initializer='random_uniform'))
model2018.add(Dense(units=1, activation='linear', use_bias=True,kernel_initializer='random_uniform', bias_initializer='random_uniform'))


model2018.compile(loss='mean_absolute_error',optimizer=keras.optimizers.SGD(lr=0.031,momentum=0.1))
model2018.fit(x=X_train2018, y=Y_train2018, batch_size=400, epochs=500)

print('MODEL 2018')
print(model2018.evaluate(x=X_test2018, y=Y_test2018))

###########################################################################################################################################
data_features = pd.read_csv('main_features.csv') 

del data_features['id']

features = ['pac','sho','pas','dri','def','phy']

train_data_features = data_features.ix[:399,:]
train_overall_column_features = data_features.ix[:399,features]
train_data_with_added_columns_features = process_columns_for_work_rate_att_work_rate_def_features(train_data_features,['f16_Att','f16_Def','f17_Att','f17_Def'])

for f in features:
    del train_data_with_added_columns_features[f]

normalized_train_data_features = subtract_mean_and_divide_by_deviation(train_data_with_added_columns_features,12)
#
test_data_features = data_features.ix[400:,:]
test_overall_column_features = data_features.ix[400:,features]
test_data_with_added_columns_features = process_columns_for_work_rate_att_work_rate_def_features(test_data_features,['f16_Att','f16_Def','f17_Att','f17_Def'])

for f in features:
    del test_data_with_added_columns_features[f]
normalized_test_data_features = subtract_mean_and_divide_by_deviation(test_data_with_added_columns_features,12)


X_train_features = normalized_train_data_features.as_matrix()
Y_train_features = train_overall_column_features.as_matrix()

X_test_features = test_data_with_added_columns_features.as_matrix()
Y_test_features = test_overall_column_features.as_matrix()

model_features = Sequential()

model_features.add(Dense(units=500, activation='relu', input_shape=(89,), use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model_features.add(Dense(units=6, activation='linear', use_bias=True, kernel_initializer='random_uniform', bias_initializer='random_uniform'))


model_features.compile(loss='mean_absolute_error', optimizer=keras.optimizers.SGD(lr=0.031,momentum=0.3))
model_features.fit(x=X_train_features, y=Y_train_features, batch_size=400, epochs=1000)

print('MODEL 2016-2017')
print(model_features.evaluate(x=X_test_features, y=Y_test_features))
predicted_features=model_features.predict(X_test_features)
print(predicted_features)


