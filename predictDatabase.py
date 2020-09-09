'''
python essentiaTest.py path/results.csv path/corpus (wav files)

Read in csv, file, value
extract features from file
write low level features to csv

appends audio features to results and outputs results_features.csv (File, Value, ...)
'''

import essentia
import essentia.standard as es
import numpy as np
import csv
import sys
import os
import pickle
from sklearn.svm import SVR
import numpy as np

data = []
# pass these features ot handle nested arrays
select_features = pickle.load(open('select_features.pkl', 'rb'))
# load the model from disk
loaded_model = pickle.load(open('model.pkl', 'rb'))

passFeatures=['rhythm.bpm_histogram','lowlevel.mfcc.cov','lowlevel.gfcc.cov',\
    'lowlevel.mfcc.icov','lowlevel.gfcc.icov']

list_of_files = {}
# open directory of samples to predict
for (dirpath, dirnames, filenames) in os.walk(sys.argv[1]):
    for filename in filenames:
        if filename.endswith('.wav'):
            path_file = os.sep.join([dirpath, filename])
            row = {'File':filename, 'Features':[]}
            try:
                features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],\
                rhythmStats=[],tonalStats=[])(path_file)
            except (RuntimeError, TypeError, NameError):
                pass
            # log low level features
            for feature in sorted(features.descriptorNames()):
                # only log low level features not in the passFeatures list
                if feature not in passFeatures and not feature.find('lowlevel'):
                    # if feature is list then expand
                    if isinstance(features[feature], (list, tuple, np.ndarray)):
                        index = 0
                        for i in features[feature]:
                            feat = feature+str(index)
                            index = index + 1
                            if feat in select_features:
                                row['Features'].append(i)
                    else:
                        if feature in select_features:
                            row['Features'].append(features[feature])
            # massage the features for the SVR model
            row['Features'] = np.asarray(row['Features'])
            row['Value'] = loaded_model.predict(row['Features'].reshape(1,-1))
            data.append(row)


# write to csv
keys = list(data[0].keys())
print(keys)
with open('results_predicted.csv', 'w', newline='')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys,restval="",extrasaction='ignore')
    dict_writer.writeheader()
    dict_writer.writerows(data)
