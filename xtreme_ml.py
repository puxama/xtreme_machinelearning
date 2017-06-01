import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np

############################
##Read train/test set
##read Barcelona's Holidays
############################
train_contacts = pd.read_csv('Train/Contacts_Pre_2017.csv')
test_contacts = pd.read_csv('Test/Contacts2017.csv')
feriados_barcelona = pd.read_csv('Train/barcelona_festivos.csv')

###################
##CONTACT.TYPE
#Call - Input                   2556
#Web - Input                    2556
#Internal Management            2332
#Mail - Recieved                2077
#Fax - Input                    1838
#Fax Acknowledgement - Input    1760
#Visit                          1744
#Mail - Input                   1624
#Installation Report - Input     206
#Tweet - Input                    73
##################
###
##GROUP BY CONTACT.TYPE
train_contacts.drop(['END.DATE'], axis=1, inplace = True)
train_contacts = train_contacts.groupby(['START.DATE', 'CONTACT.TYPE'])['Contacts'].sum().reset_index()
train_contacts['ID'] = -1
contacts = pd.concat([train_contacts, test_contacts])

##################
###Dates preprocessing
#################
contacts['START.DATE'] = pd.to_datetime(contacts['START.DATE'])
contacts = contacts.loc[~((contacts['START.DATE'].dt.month == 2) &
                                    (contacts['START.DATE'].dt.day == 29)), ]
contacts['Dayofweek'] = contacts['START.DATE'].dt.dayofweek
contacts['Day'] = contacts['START.DATE'].dt.day
contacts['Month'] = contacts['START.DATE'].dt.month

######
###Join Barcelona's Holidays
######
feriados_barcelona['Fecha'] = pd.to_datetime(feriados_barcelona['Fecha'])
contacts = contacts.merge(feriados_barcelona, how='left', left_on='START.DATE', right_on='Fecha')
contacts.drop(['Fecha'], axis=1, inplace = True)
contacts.fillna(0, inplace=True)


#Hyperparameters xgboost
params = {'objective': 'reg:linear', 'booster': 'gbtree', 'silent': 1,
'max_depth': 10, 'eta': 0.1, 'nthread': 4,
'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 20,
'max_delta_step': 0, 'gamma': 0}

#model for each CONTACT.TYPE, with 50 ensamble for each CONTACT.TYPE
type_contact = contacts['CONTACT.TYPE'].value_counts().index
model_list = []
final_contacts = contacts[0:0].copy()
for contact in type_contact:
    train_set = contacts.loc[(contacts['CONTACT.TYPE']==contact) & (contacts['ID']==-1),]
    test_set = contacts.loc[(contacts['CONTACT.TYPE']==contact) & (contacts['ID']>-1),]
    dtrain = xgb.DMatrix(train_set[predictors], label=train_set['Contacts'])
    dtest = xgb.DMatrix(test_set[predictors])
    watchlist = [(dtrain, 'dtrain')]
    
    num_rounds = 224
    preds = np.zeros(test_set.shape[0])
    for s in np.random.randint(0, 1000000, size=50):
        params['seed'] = s
        clf_xgb_main = xgb.train(dtrain=dtrain, params=params, num_boost_round=num_rounds, evals=watchlist,
                            verbose_eval=False)
        preds += clf_xgb_main.predict(dtest)
    preds = preds/50
    test_set['Contacts'] = preds
    final_contacts = pd.concat([final_contacts, test_set])

#Export results
final_contacts = final_contacts.reset_index(drop=True)
final_contacts[['Contacts', 'ID']].to_csv('contacts.csv', index=False)


######################
####RESOLUTION MODEL##
######################

#Read train/test set of resolution
train_resolution = pd.read_csv('Train/Resolution_Pre_2017.csv')
test_resolution = pd.read_csv('Test/Resolution2017.csv')

#Groupby Category and Subject
train_resolution = train_resolution[['Date', 'Category', 'Subject', 'Resolution']]
train_resolution = train_resolution.groupby(['Date', 'Category', 'Subject'])['Resolution'].sum().reset_index()
train_resolution['ID'] = -1
resolution = pd.concat([train_resolution, test_resolution])

resolution['Cat_Subject'] = resolution['Category']+'_'+resolution['Subject']

#dates preprocessing
resolution['Date'] = pd.to_datetime(resolution['Date'])
resolution = resolution.loc[~((resolution['Date'].dt.month == 2) &
                                    (resolution['Date'].dt.day == 29)), ]
resolution['Dayofweek'] = resolution['Date'].dt.dayofweek
resolution['Day'] = resolution['Date'].dt.day
resolution['Month'] = resolution['Date'].dt.month

######
###Join Barcelona's Holidays
######
feriados_barcelona['Fecha'] = pd.to_datetime(feriados_barcelona['Fecha'])
resolution = resolution.merge(feriados_barcelona, how='left', left_on='Date', right_on='Fecha')
resolution.drop(['Fecha'], axis=1, inplace = True)
resolution.fillna(0, inplace=True)


predictors = [var for var in resolution.columns.values if var not in ['Date', 'Category', 'Subject', 'Resolution', 'ID',
                                                                   'Cat_Subject']]

#50 model for each category-subject combination
type_resolution = resolution['Cat_Subject'].value_counts().index
model_list = []
final_resolution = resolution[0:0].copy()
for contact in type_resolution:
    train_set = resolution.loc[(resolution['Cat_Subject']==contact) & (resolution['ID']==-1),]
    test_set = resolution.loc[(resolution['Cat_Subject']==contact) & (resolution['ID']>-1),]
    dtrain = xgb.DMatrix(train_set[predictors], label=train_set['Resolution'])
    dtest = xgb.DMatrix(test_set[predictors])
    watchlist = [(dtrain, 'dtrain')]
    
    num_rounds = 224
    preds = np.zeros(test_set.shape[0])
    if train_set.shape[0]>0:
        for s in np.random.randint(0, 1000000, size=50):
            params['seed'] = s
            clf_xgb_main = xgb.train(dtrain=dtrain, params=params, num_boost_round=num_rounds, evals=watchlist,
                                verbose_eval=False)
            preds += clf_xgb_main.predict(dtest)
        preds = preds/50
        test_set['Resolution'] = preds
    else:
        test_set['Resolution'] = 0
    final_resolution = pd.concat([final_resolution, test_set])

#Export results
final_resolution = final_resolution.reset_index(drop=True)
final_resolution[['Resolution', 'ID']].to_csv('resolution.csv', index=False)