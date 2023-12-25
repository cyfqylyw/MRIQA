import os
import torch
import pandas as pd
import numpy as np
from args import args
from main import SimCLR, Encoder3D, Encoder3D_fourlier
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from mri_datasets import get_filename


def generate_features(model_path, data_type):
    """
    use model to generate features
    :param data_type: only one of ['origin_data', 'fourlier_origin_data']
    """
    # load model
    model = torch.load(model_path)
    model_encoder = model.encoder

    file_lst = list(pd.read_csv('label.csv')['file_list'])

    if not os.path.exists('./datasets/' + data_type +  '_features/'):
        os.mkdir('./datasets/' + data_type +  '_features/')

    # generate features with batch=50
    print('Generating features with batch size 50:')
    for idx in range(len(file_lst)):
        filename = get_filename(file_path=file_lst[idx])
        data = np.load('./datasets/' + data_type + '/' + filename)
        data = data.reshape((data.shape[0], data.shape[-3], data.shape[-2], data.shape[-1]))
        data = np.expand_dims(data, axis=0)
        all_data = data if idx%50==0 else np.concatenate((all_data, data), axis=0)

        if idx % 50 == 49 or idx == len(file_lst) - 1:
            y = model_encoder(torch.tensor(all_data.astype(np.float32)).to(torch.device('cuda:0')))
            np.save('./datasets/' + data_type +  '_features/' + str(idx+1) + '.npy', y.detach().cpu().numpy())
            print('\t', idx+1, all_data.shape, y.shape)

def merge_features(model_path, data_type):
    # merge all the features
    file_lst = list(pd.read_csv('label.csv')['file_list'])
    all_data = None
    for idx in range(len(file_lst)):
        if idx % 50 == 49 or idx == len(file_lst) - 1:
            data = np.load('./datasets/' + data_type +  '_features/' + str(idx+1) + '.npy')
            all_data = data if idx==49 else np.vstack((all_data, data))
    
    # save the overall features
    model_short_name = '.'.join(model_path.split('/')[-1].split('.')[:-1])
    np.save('./datasets/' + data_type +  '_features/all_data_' + model_short_name + '.npy', all_data)


def generate_feature_label_df(all_features, labels, test_size=0.2):
    df1 = pd.DataFrame(all_features, columns=['f' + str(x) for x in range(all_features.shape[1])])
    df2 = pd.DataFrame(labels, columns=['label'])
    df = pd.concat([df1, df2], axis=1)
    df_train, df_test = train_test_split(df, test_size=test_size)
    return df_train, df_test


if __name__ == '__main__':
    # use model to generate features with corresponding data type
    mode = args.mode
    batch_size = args.batch_size
    proj_dim = args.projection_dim

    if mode == 'combined':
        model_path_1 = 'models/model_augmentation_b' + str(batch_size) + '_p' + str(proj_dim) + '_e100_lr0.001.pt'
        data_type_1 = 'origin_data' if 'augmentation' in model_path_1 else 'fourlier_origin_data'
        model_short_name_1 = '.'.join(model_path_1.split('/')[-1].split('.')[:-1])
        print('datasets/' + data_type_1 +  '_features/all_data_' + model_short_name_1 + '.npy')
        all_features_1 = np.load('datasets/' + data_type_1 +  '_features/all_data_' + model_short_name_1 + '.npy')
    
        model_path_2 = 'models/model_fourlier_b' + str(batch_size) + '_p' + str(proj_dim) + '_e100_lr0.001.pt'
        data_type_2 = 'origin_data' if 'augmentation' in model_path_2 else 'fourlier_origin_data'
        model_short_name_2 = '.'.join(model_path_2.split('/')[-1].split('.')[:-1])
        all_features_2 = np.load('datasets/' + data_type_2 +  '_features/all_data_' + model_short_name_2 + '.npy')
    
        labels = list(pd.read_csv('label.csv')['label'])
        df1 = pd.DataFrame(all_features_1, columns=['a' + str(x) for x in range(all_features_1.shape[1])])
        df2 = pd.DataFrame(all_features_2, columns=['f' + str(x) for x in range(all_features_2.shape[1])])
        df3 = pd.DataFrame(labels, columns=['label'])
        df = pd.concat([df1, df2, df3], axis=1)
        
    else:
        model_path = 'models/model_' + mode + '_b' + str(batch_size) + '_p' + str(proj_dim) + '_e100_lr0.001.pt'
        # model_path = 'models/model_augmentation_b32_p128_e100_lr0.001.pt'
        print(model_path)
    
        data_type = 'origin_data' if 'augmentation' in model_path else 'fourlier_origin_data'
        generate_features(model_path=model_path, data_type=data_type)
        merge_features(model_path=model_path, data_type=data_type)
    
        # Load data
        model_short_name = '.'.join(model_path.split('/')[-1].split('.')[:-1])
        all_features = np.load('./datasets/' + data_type +  '_features/all_data_' + model_short_name + '.npy')
        labels = list(pd.read_csv('label.csv')['label'])

        df1 = pd.DataFrame(all_features, columns=['f' + str(x) for x in range(all_features.shape[1])])
        df2 = pd.DataFrame(labels, columns=['label'])
        df = pd.concat([df1, df2], axis=1)


    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
    train_data = TabularDataset(df_train)
    test_data = TabularDataset(df_test)

    models = {
        'RF': {},   # RandomForest
        'XGB': {},  # XGBoost
        'KNN': {}   # K-Nearest Neighbors
    }
    predictor = TabularPredictor(label='label').fit(train_data=train_data, hyperparameters=models)
    predictions = predictor.predict(test_data)

    metric_lst = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovo_macro']
    mec_df = predictor.leaderboard(test_data, extra_metrics=metric_lst, silent=True)

    acc, pre, rec, f1, auc = mec_df[metric_lst].iloc[0,:]
    print("acc, pre, rec, f1, auc =", [round(x, 6) for x in [acc, f1, pre, rec, auc]])  
