"""Train AutoGluon (AutoML, automated machine learning) on the VNF data and save the results.

Mirrors the data preparation of vnf_web.ipynb (small flows, 128 MB memory, 20 zero
measurements added, MinMax-scaled throughput -> CPU) so that the saved models can be
loaded in the notebook and compared against the other regressors.

For each VNF this saves
  ml_models/<vnf>/autogluon/            final predictor trained on all data (full stack/ensemble)
  ml_models/<vnf>/autogluon_cv.json     5-fold CV RMSE (same folds as sklearn's cross_val_score)

Usage: python train_autogluon.py [--time-limit SECONDS] [--preset best_quality] [--vnfs nginx squid]
"""
import argparse
import json
import os
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

FEATURE = 'Max. throughput [kB/s]'
TARGET = 'CPU'
SMALL_CMD = {
    'nginx': 'ab -c 1 -t 60 -n 99999999 -e /tngbench_share/ab_dist.csv -s 60 -k -i http://20.0.0.254:8888/',
    'haproxy': 'ab -c 1 -t 60 -n 99999999 -e /tngbench_share/ab_dist.csv -s 60 -k -i http://20.0.0.254:8888/',
    'squid': 'ab -c 1 -t 60 -n 99999999 -e /tngbench_share/ab_dist.csv -s 60 -k -i -X 20.0.0.254:3128 http://40.0.0.254:80/',
}
VNFS = {  # name -> (csv file, function id used in the column names)
    'nginx': ('WEB1', 'lb-nginx'),
    'haproxy': ('WEB2', 'lb-haproxy'),
    'squid': ('WEB3', 'px-squid'),
}


def load_vnf(vnf, mem=128, num_zeros=20):
    """Same processing as vnf_web.ipynb: small flows, fixed memory, plus zero measurements."""
    csv, func = VNFS[vnf]
    df = pd.read_csv(f'vnf_data/csv_experiments_{csv}.csv')
    prefix = f'param__func__de.upb.{func}.0.1__'
    df = df.rename(columns={
        'param__func__mp.input__cmd_start': 'size',
        'metric__mp.input.vdu01.0__ab_transfer_rate_kbyte_per_second': FEATURE,
        prefix + 'cpu_bw': TARGET,
        prefix + 'mem_max': 'Memory',
    })
    df = df[(df['size'] == SMALL_CMD[vnf]) & (df['Memory'] == mem)][[FEATURE, TARGET]]
    zeros = pd.DataFrame({FEATURE: [0] * num_zeros, TARGET: [0] * num_zeros})
    return pd.concat([df, zeros], ignore_index=True)


def fit_predictor(train, path, args):
    from autogluon.tabular import TabularPredictor
    shutil.rmtree(path, ignore_errors=True)
    predictor = TabularPredictor(label=TARGET, problem_type='regression',
                                 eval_metric='root_mean_squared_error', path=path, verbosity=1)
    predictor.fit(train, presets=args.preset, time_limit=args.time_limit,
                  num_cpus=args.num_cpus, num_gpus=0)
    return predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vnfs', nargs='+', default=['nginx', 'squid'], choices=list(VNFS))
    ap.add_argument('--time-limit', type=int, default=300, help='seconds per fit (final + each CV fold)')
    ap.add_argument('--preset', default='best_quality')
    ap.add_argument('--num-cpus', type=int, default=os.cpu_count())
    ap.add_argument('--cv-folds', type=int, default=5, help='0 to skip the CV evaluation')
    args = ap.parse_args()

    for vnf in args.vnfs:
        data = load_vnf(vnf)
        X = data[[FEATURE]].fillna(data[FEATURE].median())
        y = data[TARGET]
        os.makedirs(f'ml_models/{vnf}', exist_ok=True)

        # the notebook scales the feature with MinMaxScaler before every model; do the same
        scaler = joblib.load(f'ml_models/{vnf}/scaler.joblib') if os.path.exists(f'ml_models/{vnf}/scaler.joblib') \
            else MinMaxScaler().fit(X)
        Xs = pd.DataFrame(scaler.transform(X), columns=[FEATURE])

        if args.cv_folds:
            rmse = []
            for k, (tr, va) in enumerate(KFold(args.cv_folds).split(Xs)):  # same folds as cross_val_score
                train = pd.concat([Xs.iloc[tr], y.iloc[tr]], axis=1)
                p = fit_predictor(train, f'/tmp/ag_cv_{vnf}_{k}', args)
                pred = p.predict(Xs.iloc[va])
                rmse.append(float(np.sqrt(np.mean((pred.values - y.iloc[va].values) ** 2))))
                print(f'{vnf} fold {k}: RMSE {rmse[-1]:.4f}')
                shutil.rmtree(f'/tmp/ag_cv_{vnf}_{k}', ignore_errors=True)
            with open(f'ml_models/{vnf}/autogluon_cv.json', 'w') as f:
                json.dump({'rmse': rmse, 'preset': args.preset, 'time_limit': args.time_limit}, f)
            print(f'{vnf} CV RMSE: {np.mean(rmse):.4f} (+/-{np.std(rmse):.4f})')

        predictor = fit_predictor(pd.concat([Xs, y], axis=1), f'ml_models/{vnf}/autogluon', args)
        print(predictor.leaderboard(silent=True).head(10))
        # keep only what is needed for inference, drops training artifacts and shrinks the folder
        predictor.save_space()


if __name__ == '__main__':
    main()
