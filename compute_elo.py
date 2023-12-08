import pathlib
from eval_othello import evaluate
import os
import json


score_file = 'score.jsonl'

cached_result = {}

if os.path.exists(score_file):
    with open(score_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            key = f'{obj["p1"]},{obj["p2"]}'
            cached_result[key] = obj['score']


files = list(pathlib.Path('.').glob('input_*.pt'))


epochs = []
for f in files:
    epoch_id = f.name.split('_')[1].split('.')[0]
    epochs.append(int(epoch_id))

# sort epochs from small to large
epochs.sort(key=lambda x: int(x))
print(epochs)

# run pairwise evaluation
for i in range(len(epochs)):
    for j in range(i+1, len(epochs)):
        p1 = epochs[i]
        p2 = epochs[j]

        if f'{p1},{p2}' in cached_result:
            score = cached_result[f'{p1},{p2}']
        else:
            print(f'{p1},{p2} evaluating')
            file_1 = f'input_{p1}.pt'
            file_2 = f'input_{p2}.pt'
            score = evaluate(file_1, file_2)
            cached_result[f'{p1},{p2}'] = score
            with open(score_file, 'a', encoding='utf-8') as f:
                json.dump({
                    'p1': p1,
                    'p2': p2,
                    'score': score
                }, f)
                f.write('\n')
        print(f'{p1},{p2} evaluated, score: {score}')

        # switch p1 and p2
        p1 = epochs[j]
        p2 = epochs[i]

        if f'{p1},{p2}' in cached_result:
            score = cached_result[f'{p1},{p2}']
        else:
            print(f'{p1},{p2} evaluating')
            file_1 = f'input_{p1}.pt'
            file_2 = f'input_{p2}.pt'
            score = evaluate(file_1, file_2)
            cached_result[f'{p1},{p2}'] = score
            with open(score_file, 'a', encoding='utf-8') as f:
                json.dump({
                    'p1': p1,
                    'p2': p2,
                    'score': score
                }, f)
                f.write('\n')
        print(f'{p1},{p2} evaluated, score: {score}')
      



