# To run the program
# python3 /path/to/file/decision_tree.py

import math
import numpy as np
import csv
import copy

from id3 import Id3Estimator
from id3 import export_graphviz
from array import array

estimator = Id3Estimator()
feaure_names=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
with open('AI/Decision_tree/cleveland.csv') as csvfile:
    reader = csv.DictReader(csvfile,fieldnames=feaure_names)
    list_all_data=[dat for dat in reader if "?" not in dat.values()]

    temp_all_data=copy.deepcopy(list_all_data[:-1])
    list_data=np.array([np.array([v for k,v in data.items()]) for data in temp_all_data if data.pop('num',None)])
    list_target=np.array([np.array([data['num']]) for data in list_all_data[:-1] ])

estimator.fit(list_data, list_target)
export_graphviz(estimator.tree_, 'DecTree.dot', feaure_names)
test_temp_data=copy.deepcopy(list_all_data[-5:-4])
test_data=np.array([np.array([v for k,v in data.items()]) for data in test_temp_data if data.pop('num',None)])

predict_data=estimator.predict(test_data)
actual_outcome=[d['num'] for d in list_all_data[-5:-4]]
print('\n\nTesting input set:\n')
for k,v in list_all_data[-5:-4][0].items():
    if(k!='num'):
        print(f'{k}={v}')

print('\n')
print(f'Actual outcome: {actual_outcome[0]}')
print(f"Predicted outcome: {predict_data[0]}")