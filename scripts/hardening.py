import argparse
import os
import sys

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

feature_groups = {
    '1a': ['duration'],
    '1b': ['src_bytes'],
    '1c': ['dst_bytes'],
    '1d': ['tot_pkts'],
    '2a': ['duration', 'src_bytes'],
    '2b': ['duration', 'dst_bytes'],
    '2c': ['duration', 'tot_pkts'],
    '2d': ['src_bytes', 'dst_bytes'],
    '2e': ['src_bytes', 'tot_pkts'],
    '2f': ['dst_bytes', 'tot_pkts'],
    '3a': ['duration', 'src_bytes', 'dst_bytes'],
    '3b': ['duration', 'src_bytes', 'tot_pkts'],
    '3c': ['duration', 'dst_bytes', 'tot_pkts'],
    '3d': ['src_bytes', 'dst_bytes', 'tot_pkts'],
    '4a': ['duration', 'src_bytes', 'dst_bytes', 'tot_pkts']
}

increment_steps = {
    'I': {
        'duration': 1, 'src_bytes': 1, 'dst_bytes': 1, 'tot_pkts': 1
    },
    'II': {
        'duration': 2, 'src_bytes': 2, 'dst_bytes': 2, 'tot_pkts': 2
    },
    'III': {
        'duration': 5, 'src_bytes': 8, 'dst_bytes': 8, 'tot_pkts': 5
    },
    'IV': {
        'duration': 10, 'src_bytes': 16, 'dst_bytes': 16, 'tot_pkts': 10
    },
    'V': {
        'duration': 15, 'src_bytes': 64, 'dst_bytes': 64, 'tot_pkts': 15
    },
    'VI': {
        'duration': 30, 'src_bytes': 128, 'dst_bytes': 128, 'tot_pkts': 20
    },
    'VII': {
        'duration': 45, 'src_bytes': 256, 'dst_bytes': 256, 'tot_pkts': 30
    },
    'VIII': {
        'duration': 60, 'src_bytes': 512, 'dst_bytes': 512, 'tot_pkts': 50
    },
    'IX': {
        'duration': 120, 'src_bytes': 1024, 'dst_bytes': 1024, 'tot_pkts': 100
    }
}

parser = argparse.ArgumentParser(description='Harden detectors')
parser.add_argument('-p', '--processed-data', type=str, required=True, help='path to processed data')
parser.add_argument('-g', '--generated-data', type=str, required=True, help='path to generated data')
parser.add_argument('-bn', '--baseline-normal-scores', type=str, required=True, help='path to save the baseline detector scores in normal settings')
parser.add_argument('-ba', '--baseline-adversarial-scores', type=str, required=True, help='path to save the baseline detector scores in adversarial settings')
parser.add_argument('-hn', '--hardened-normal-scores', type=str, required=True, help='path to save the hardened detector scores in normal settings')
parser.add_argument('-ha', '--hardened-adversarial-scores', type=str, required=True, help='path to save the hardened detector scores in adversarial settings')

args = parser.parse_args()

if not os.path.exists(args.processed_data) or not os.path.isfile(args.processed_data):
    sys.exit('Path to processed data does not exist or is not a file')

if not os.path.exists(args.generated_data) or not os.path.isfile(args.generated_data):
    sys.exit('Path to generated data does not exist or is not a file')

orig_data = pd.read_csv(args.processed_data)

orig_data = orig_data.drop(columns=['src_ip_port', 'dst_ip_port'])

ben_orig_data = orig_data[orig_data.label == 0]

mal_orig_data = orig_data[orig_data.label == 1]

data = pd.concat([ben_orig_data, mal_orig_data], ignore_index=True).sample(frac=1)

train, test = train_test_split(data, test_size=0.2)

classifier = RandomForestClassifier(n_estimators=10, n_jobs=-1)
classifier.fit(train.drop(columns=['label']), train.label)
pred = classifier.predict(test.drop(columns=['label']))

scores = pd.DataFrame(data={'F1': f1_score(test.label, pred), 'Precision': precision_score(test.label, pred), 'Recall': recall_score(test.label, pred)}, index=[0])
scores.to_csv(args.baseline_normal_scores, index=False)
print(scores)

groups = []
steps = []
f1_scores = []
precision_scores = []
recall_scores = []

for feature_group in feature_groups.keys():
    for increment_step in increment_steps.keys():
        groups.append(feature_group)
        steps.append(increment_step)
        adv_data = test.copy()
        for altered_feature in feature_groups[feature_group]:
            adv_data.loc[adv_data.label == 1, altered_feature] += increment_steps[increment_step][altered_feature]
        adv_data.tot_bytes = adv_data.src_bytes + adv_data.dst_bytes
        pred = classifier.predict(adv_data.drop(columns=['label']))
        f1_scores.append(f1_score(adv_data.label, pred))
        precision_scores.append(precision_score(adv_data.label, pred))
        recall_scores.append(recall_score(adv_data.label, pred))

scores = pd.DataFrame(data={'Group': groups, 'Step': steps, 'F1': f1_scores, 'Precision': precision_scores, 'Recall': recall_scores})
scores.to_csv(args.baseline_adversarial_scores, index=False)
print(scores)

synt_data = pd.read_csv(args.generated_data)

ben_synt_data = synt_data[synt_data.label == 0]

mal_synt_data = synt_data[synt_data.label == 1]

data = pd.concat([ben_orig_data, ben_orig_data, mal_orig_data, mal_synt_data], ignore_index=True).sample(frac=1)

train, test = train_test_split(data, test_size=0.2)

classifier = RandomForestClassifier(n_estimators=10, n_jobs=-1)
classifier.fit(train.drop(columns=['label']), train.label)
pred = classifier.predict(test.drop(columns=['label']))

scores = pd.DataFrame(data={'F1': f1_score(test.label, pred), 'Precision': precision_score(test.label, pred), 'Recall': recall_score(test.label, pred)}, index=[0])
scores.to_csv(args.hardened_normal_scores, index=False)
print(scores)

groups = []
steps = []
f1_scores = []
precision_scores = []
recall_scores = []

for feature_group in feature_groups.keys():
    for increment_step in increment_steps.keys():
        groups.append(feature_group)
        steps.append(increment_step)
        adv_data = test.copy()
        for altered_feature in feature_groups[feature_group]:
            adv_data.loc[adv_data.label == 1, altered_feature] += increment_steps[increment_step][altered_feature]
        adv_data.tot_bytes = adv_data.src_bytes + adv_data.dst_bytes
        pred = classifier.predict(adv_data.drop(columns=['label']))
        f1_scores.append(f1_score(adv_data.label, pred))
        precision_scores.append(precision_score(adv_data.label, pred))
        recall_scores.append(recall_score(adv_data.label, pred))

scores = pd.DataFrame(data={'Group': groups, 'Step': steps, 'F1': f1_scores, 'Precision': precision_scores, 'Recall': recall_scores})
scores.to_csv(args.hardened_adversarial_scores, index=False)
print(scores)