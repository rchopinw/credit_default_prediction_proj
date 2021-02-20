import json
import pandas as pd
from data_processing_tools import *
from sklearn.preprocessing import StandardScaler


# Loading data
data_raw = []
with open('transactions.txt') as f:
    for line in f.readlines():
        data_raw.append(json.loads(line))
data = pd.DataFrame(data_raw)
print('... Data import completed, with {} observations and {} features ...'.format(data.shape[0], data.shape[1]))

del data_raw
data = data.applymap(lambda x: None if x == '' else x)
# data.to_csv('data_raw.csv', encoding='utf_8_sig', index=False)
# identifying unique values
data.apply(lambda x: len(x.unique()), axis=1)

# converting bool to discrete 0/1 variable
bool_columns = ['cardPresent', 'expirationDateKeyInMatch', 'isFraud']
for c in bool_columns:
    data[c] = data[c].apply(lambda x: 1 if x else 0)

# converting Date-time into year-month-day-hour-minute-second
data['transactionDateTime'] = \
    data['transactionDateTime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'))

for i in ['year', 'month', 'day', 'hour', 'minute', 'second']:
    data['transaction_' + i] = data['transactionDateTime'].apply(lambda x: float(x.__getattribute__(i)))

data['accountOpenDate'] = data['accountOpenDate'].apply(date_transfer)
data['dateOfLastAddressChange'] = data['dateOfLastAddressChange'].apply(date_transfer)
data['currentExpDate'] = data['currentExpDate'].apply(lambda x: datetime.datetime.strptime(x, '%m/%Y'))

# Missing value configurations and early feature selection
all_missing_columns = ['echoBuffer', 'merchantCity', 'merchantState', 'merchantZip',
                       'cardPresent', 'posOnPremises', 'recurringAuthInd']
all_duplicated_columns = ['acqCountry', 'merchantCountryCode']
identifying_columns = ['accountNumber', 'customerId', 'merchantName']
data = data.drop(columns=all_missing_columns + all_duplicated_columns + identifying_columns)
data = data.dropna()

# creating feature: cvv matching
data['CVVMatch'] = [1 if x == y else 0 for x, y in zip(data['cardCVV'], data['enteredCVV'])]
data = data.drop(columns=['cardCVV', 'enteredCVV'])
# creating feature: one-hot coded 'merchantCategoryCode'
data = pd.concat([data, pd.get_dummies(data['merchantCategoryCode'])], axis=1)
data = data.drop(columns=['merchantCategoryCode'])

# creating feature: transactionDateTime - accountOpenDate
data['account_open_length'] = data['transactionDateTime'] - data['accountOpenDate']
data['account_open_length'] = data['account_open_length'].apply(lambda x: x.seconds).astype('float')
data = data.drop(columns=['accountOpenDate'])

# creating feature: transactionDateTime - dateOfLastAddressChange
data['address_change_length'] = data['transactionDateTime'] - data['dateOfLastAddressChange']
data['address_change_length'] = data['address_change_length'].apply(lambda x: x.seconds).astype('float')
data = data.drop(columns=['dateOfLastAddressChange'])

# creating feature: transactionDateTime - currentExpDate
data['exp_length'] = data['currentExpDate'] - data['transactionDateTime']
data['exp_length'] = data['exp_length'].apply(lambda x: x.seconds).astype('float')
data = data.drop(columns=['transactionDateTime'])
data = data.drop(columns=['currentExpDate'])

# creating feature: one-hot coded for 'posEntryMode'
data = pd.concat([data, pd.get_dummies(data['posEntryMode'])], axis=1)
data = data.drop(columns=['posEntryMode'])

# creating feature: one-hot coded for 'posConditionCode'
data = pd.concat([data, pd.get_dummies(data['posConditionCode'])], axis=1)
data = data.drop(columns=['posConditionCode'])

# creating feature: one-hot coded for 'transactionType'
data = pd.concat([data, pd.get_dummies(data['transactionType'])], axis=1)
data = data.drop(columns=['transactionType'])

# Creating feature: one-hot coded for 'creditLimit'
data['creditLimit'] = data['creditLimit'].astype('int').astype(str)
data = pd.concat([data, pd.get_dummies(data['creditLimit'])], axis=1)
data = data.drop(columns=['creditLimit'])

# Creating feature: length of cardLast4Digits (accidentally found that some cards ending with only 3 digits)
data['last_x_digits_length'] = data['cardLast4Digits'].apply(lambda x: float(len(str(x))))
data = data.drop(columns=['cardLast4Digits'])

# Check finalized data shape and columns
print("Finalized training data has {} samples and {} features.".format(data.shape[0], data.shape[1]))
print('\n')
print(data.info())

# Scaling the continuous data
ss_scaler = StandardScaler()
continuous = ['availableMoney', 'transactionAmount', 'currentBalance',
              'account_open_length', 'address_change_length', 'exp_length', 'last_x_digits_length']
for i in continuous:
    data[i] = ss_scaler.fit_transform(data[[i]])

# Save cleaned data to local
data.to_csv('finalized_data.csv', encoding='utf_8_sig', index=False)
