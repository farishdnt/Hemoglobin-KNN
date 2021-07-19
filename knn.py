import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import requests
import json
import time

df = pd.read_csv('datalatih2.csv', delimiter = ';')

X_train, X_test, y_train, y_test = train_test_split(df['spo2'].values.reshape(-1, 1), df['hb'].values.reshape(-1, 1), random_state=0)

knn = KNeighborsRegressor(n_neighbors=2, weights='distance', metric='euclidean')
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
mean_squared_error(y_test, pred)

while True:
    m = requests.get("https://api.thingspeak.com/channels/1011523/feeds/last.json?api_key=XK7HED8TTLDEK3E4")
    if m.status_code == 200:
        mdata = json.loads(m.text)
        print(mdata)
        if float(mdata['field1']) > 2 and float(mdata['field2']) == 0:
            msensor = float(mdata['field1'])

            #mengolah data dari thinkspeak
            # Predict training data
            predict = knn.predict(np.array([msensor]).reshape(-1, 1))
            print('Predict = ', predict[0][0])
        
            m = requests.get("https://api.thingspeak.com/update?api_key=0FBBVYQHPGF4HOX8&field1=0&field2=%s" %(predict[0][0]))
            if m.status_code == 200:
                print("Update Thingspeak OK")
            else:
                print("Update Thingspeak Failed")
    else:
        print("Request Thingspeak Failed")
    print("Waiting ...")
    time.sleep(0.5)

