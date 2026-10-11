"""Train AutoGluon (AutoML, automated machine learning) on the VNF data and save the results.

Mirrors the data selection of vnf_web.ipynb (small flows, 128 MB memory, 20 zero measurements
added; throughput -> CPU) so that the saved models can be loaded in the notebook and compared
against the other regressors. AutoGluon gets the raw, unscaled, unimputed throughput and does
all preprocessing itself.

For each VNF this saves a single artifact, ml_models/<vnf>/AutoGluon.zip, containing
  predictor/   the final predictor trained on all data (stack/ensemble, reduced to what inference needs)
  meta.json    5-fold CV RMSE (same folds as sklearn's cross_val_score), preset, time limit, AutoGluon version
The zip can only be loaded with the AutoGluon version it was created with (see requirements-optional.txt).

Usage: python train_autogluon.py [--time-limit SECONDS] [--preset best_quality] [--vnfs nginx squid]
"""
import argparse
import json
import os
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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
    """Same selection as vnf_web.ipynb: small flows, fixed memory, plus zero measurements."""
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
    predictor = TabularPredictor(label=TARGET, problem_type='regression',
                                 eval_metric='root_mean_squared_error', path=path, verbosity=1)
    predictor.fit(train, presets=args.preset, time_limit=args.time_limit,
                  num_cpus=args.num_cpus, num_gpus=0)
    return predictor


def bundle(predictor, meta, zip_path):
    """Write the predictor (reduced to what inference needs) and meta data into a single zip file."""
    import autogluon.tabular
    meta = {**meta, 'autogluon_version': autogluon.tabular.__version__}
    with tempfile.TemporaryDirectory() as tmp:
        deploy_dir = os.path.join(tmp, 'predictor')
        predictor.clone_for_deployment(deploy_dir)  # only the final model and what it depends on
        with open(os.path.join(tmp, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmp):
                for name in sorted(files):
                    full = os.path.join(root, name)
                    zf.write(full, os.path.relpath(full, tmp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vnfs', nargs='+', default=['nginx', 'squid'], choices=list(VNFS))
    ap.add_argument('--time-limit', type=int, default=120, help='seconds per fit (final + each CV fold)')
    ap.add_argument('--preset', default='best_quality')
    ap.add_argument('--num-cpus', type=int, default=os.cpu_count())
    ap.add_argument('--cv-folds', type=int, default=5, help='0 to skip the CV evaluation')
    args = ap.parse_args()

    for vnf in args.vnfs:
        data = load_vnf(vnf)
        X, y = data[[FEATURE]], data[TARGET]  # raw: AutoGluon handles missing values, scaling etc.
        os.makedirs(f'ml_models/{vnf}', exist_ok=True)
        meta = {'preset': args.preset, 'time_limit': args.time_limit}

        if args.cv_folds:
            rmse = []
            for k, (tr, va) in enumerate(KFold(args.cv_folds).split(X)):  # same folds as cross_val_score
                with tempfile.TemporaryDirectory() as tmp:
                    p = fit_predictor(pd.concat([X.iloc[tr], y.iloc[tr]], axis=1), tmp, args)
                    pred = p.predict(X.iloc[va])
                rmse.append(float(np.sqrt(np.mean((pred.values - y.iloc[va].values) ** 2))))
                print(f'{vnf} fold {k}: RMSE {rmse[-1]:.4f}')
            meta['cv_rmse'] = rmse  # only present if CV was run for this very training run
            print(f'{vnf} CV RMSE: {np.mean(rmse):.4f} (+/-{np.std(rmse):.4f})')

        with tempfile.TemporaryDirectory() as tmp:
            predictor = fit_predictor(data, tmp, args)
            print(predictor.leaderboard(silent=True).head(10))
            bundle(predictor, meta, f'ml_models/{vnf}/AutoGluon.zip')


if __name__ == '__main__':
    main()
