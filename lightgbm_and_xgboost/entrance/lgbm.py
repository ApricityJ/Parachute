import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from lightgbm_and_xgboost.model.lightGBM import LightGBM
from lightgbm_and_xgboost.utils.util import of_json, to_json

warnings.filterwarnings("ignore")


if __name__ == '__main__':

    args = {
        'dataset': 'first_loan',
        'version': '1',
        'objective': 'binary',  # binary, multiclass...
        'metric': 'auc',  # None需在模型中指定
        'num_class': 2,
        'target': 'feature_importance',  # train, predict, feature_importance
        'optimizer': 'optuna',  # hyperopt, optuna...
        'save_experiment': True,
        'data_path': Path("../data"),
        'result_path': Path("../result"),
        'train_file_name': 'train.p',
        'test_file_name': 'predict.p',
        'out_model_name': 'result_model_lgb.p',
        'magic_seed': 29,
        'load_best_params': False,
        'params_file_name': 'best_params_lgb.dict',
    }
    print("-----------------------------")
    print(f"args : {args}")

    if args['objective'] == 'multiclass':
        assert args['num_class'] > 2, 'multiclass objective should have class num > 2.'
    assert args['params_file_name'] != '', 'please name the best params file.'

    if args['target'] == 'train':
        print("--------- begin load train and predict data ---------")
        data_bunch = pickle.load(open(args['data_path'] / args['train_file_name'], 'rb'))
        col_names = data_bunch.col_names
        print(f"columns : {col_names}")
        category_cols = data_bunch.category_cols
        print(f"category columns : {category_cols}")
        X = data_bunch.data
        y = data_bunch.target
        print(f"X train shape : {X.shape}")
        print(f"y train shape : {y.shape}")

        data_bunch = pickle.load(open(args['data_path'] / args['test_file_name'], 'rb'))
        X_predict = data_bunch.data
        id_predict = data_bunch.id
        print(f"X predict shape : {X_predict.shape}")
        print(f"id predict shape : {id_predict.shape}")
        print("--------- done load train and predict data ---------")

        # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=1, stratify=y)

        lightGBM = LightGBM(
            dataset=args['dataset'],
            train_set=[X, y],
            predict_set=[X_predict, id_predict],
            col_names=col_names,
            category_cols=category_cols,
            objective=args['objective'],
            metric=args['metric'],
            num_class=args['num_class'],
            optimizer=args['optimizer'],
            magic_seed=args['magic_seed'],
            out_dir=args['result_path'],
            out_model_name=args['out_model_name'],
            save=args['save_experiment'],
            version=args['version'])

        if args['load_best_params']:
            best_params = of_json(args['result_path'].joinpath(args['params_file_name']))
        else:
            best_params = lightGBM.optimize()
            best_params['num_leaves'] = int(best_params['num_leaves'])
            to_json(best_params, args['result_path'].joinpath(args['params_file_name']))

        print("--------- best params ---------")
        print(best_params)

        if args['objective'] == 'binary':
            lightGBM.train_and_predict_binary(best_params)
        elif args['objective'] == 'multiclass':
            lightGBM.train_and_predict_multiclass(best_params)
        else:
            pass

    elif args['target'] == 'predict':
        assert args['out_model_name'] != '' and args['result_path'] != '', 'please give the model path.'

        print("--------- begin load predict data ---------")
        data_bunch = pickle.load(open(args['data_path'] / args['test_file_name'], 'rb'))
        X_predict = data_bunch.data
        id_predict = data_bunch.id
        print(f"X predict shape : {X_predict.shape}")
        print(f"id predict shape : {id_predict.shape}")
        print("--------- done load predict data ---------")

        data_bunch = pickle.load(open(args['result_path'] / args['out_model_name'], 'rb'))
        model = data_bunch.model
        test_prediction = model.predict(X_predict)
        print(test_prediction)

        if args['objective'] == 'binary':
            test_result = pd.DataFrame({'id': id_predict, 'predicts': test_prediction})
            test_result.to_csv(args['result_path'] / '{}_lgbm_model_{}_test_from_all_data_model.csv'
                               .format(args['dataset'], args['version']), index=False)
        elif args['objective'] == 'multiclass':
            test_prediction = np.argmax(test_prediction, axis=1)
            test_result = pd.DataFrame({'id': id_predict, 'predicts': test_prediction})
            test_result.to_csv(args['result_path'] / '{}_lgbm_model_{}_test_from_all_data_model.csv'
                               .format(args['dataset'], args['version']), index=False)
        else:
            pass

    elif args['target'] == 'feature_importance':
        assert args['out_model_name'] != '' and args['result_path'] != '', 'please give the model path.'

        data_bunch = pickle.load(open(args['result_path'] / args['out_model_name'], 'rb'))
        LightGBM.print_feature_importance(data_bunch)

        train_data_bunch = pickle.load(open(args['data_path'] / args['train_file_name'], 'rb'))
        X = train_data_bunch.data
        LightGBM.shap_feature_importance(data_bunch, X)
    else:
        pass
