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

data = []
# pass these features ot handle nested arrays
passFeatures=['rhythm.bpm_histogram','lowlevel.mfcc.cov','lowlevel.gfcc.cov',\
    'lowlevel.mfcc.icov','lowlevel.gfcc.icov']

with open(sys.argv[1]) as csvfile:
    reader = csv.DictReader(csvfile,delimiter=',')
    header = reader.fieldnames
    for row in reader:
        try:
            print(sys.argv[2]+row['File'])
            features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],\
            rhythmStats=[],tonalStats=[])(sys.argv[2]+row['File'])
        except (RuntimeError, TypeError, NameError):
            pass
        # log low level features
        for feature in sorted(features.descriptorNames()):
            if feature not in passFeatures and !feature.find('lowlevel'):
                if isinstance(features[feature], (list, tuple, np.ndarray)):
                    index = 0
                    for i in features[feature]:
                        feat = feature+str(index)
                        row[feat] = i
                        index = index + 1
                else:
                    row[feature] = features[feature]
        data.append(row)
# write to csv
keys = list(data[0].keys())
print(keys)
with open('results_features.csv', 'w', newline='')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys,restval="",extrasaction='ignore')
    dict_writer.writeheader()
    dict_writer.writerows(data)
