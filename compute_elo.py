import pathlib
from eval_othello import evaluate
import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import plotly.express as px
import sys


def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for rd, model_a, model_b, winner in battles[['p1', 'p2', 'score']].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == 1:
            sa = 1
        elif winner == -1:
            sa = 0
        elif winner == 0:
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    return rating


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


score_file = 'score.jsonl'

cached_result = {}

if os.path.exists(score_file):
    with open(score_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            key = f'{obj["p1"]},{obj["p2"]}'
            cached_result[key] = obj['score']


files = list(pathlib.Path('.').glob('input_*.pt'))

print(files)
epochs = []
for f in files:
    epoch_id = f.name.split('_')[1].split('.')[0]
    epochs.append(epoch_id)

# sort epochs from small to large
epochs.sort()
print(epochs)
sys.exit(0)

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
      

objs = []
with open(score_file, 'r', encoding='utf-8') as f:
    for l in f:
        objs.append(json.loads(l))
battles = pd.DataFrame(objs)



BOOTSTRAP_ROUNDS = 2000

np.random.seed(42)
bootstrap_elo_lu = get_bootstrap_result(battles, compute_elo, BOOTSTRAP_ROUNDS)
bootstrap_lu_median = bootstrap_elo_lu.median().reset_index().set_axis(["model", "Elo rating"], axis=1)
bootstrap_lu_median["Elo rating"] = (bootstrap_lu_median["Elo rating"] + 0.5).astype(int)
print(bootstrap_lu_median)

def visualize_bootstrap_scores(df, title):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        rating = df.quantile(.5),
        upper = df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y",
                     error_y_minus="error_y_minus", text="rating_rounded",
                     title=title)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating")
    # save to png
    fig.show()
    return fig

fig = visualize_bootstrap_scores(bootstrap_elo_lu, "Bootstrap of Elo Estimates")
fig
