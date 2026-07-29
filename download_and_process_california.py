import os
import numpy as np
import pandas as pd
from urllib import request
import zipfile
import json
from generate_mask import generate_mask

DATA_DIR = '/kaggle/working/DiffPuter_AttentionV1/datasets'

NAME_URL_DICT_UCI = {
    'adult': 'https://archive.ics.uci.edu/static/public/2/adult.zip',
    'default': 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip',
    'magic': 'https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip',
    'shoppers': 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip',
    'news': 'https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip',
    'gesture': 'https://archive.ics.uci.edu/static/public/302/gesture+phase+segmentation.zip',
    'letter': 'https://archive.ics.uci.edu/static/public/59/letter+recognition.zip',
    'bean': 'https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip'
}

def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)

def download_from_uci(name):
    print(f'Start processing dataset {name} from UCI.')
    save_dir = f'{DATA_DIR}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, f'{save_dir}/{name}.zip')
        print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
        
        unzip_file(f'{save_dir}/{name}.zip', save_dir)
        print(f'Finish unzipping {name}.')
    else:
        print(f'{name} already downloaded.')

# =====================================================================
# PROSES CALIFORNIA: MENGAMBIL DATA DARI KAGGLE & MENGHAPUS OCEAN_PROXIMITY
# =====================================================================
def process_california():
    print('Start processing dataset california from Kaggle source.')
    save_dir = f'{DATA_DIR}/california'
    os.makedirs(save_dir, exist_ok=True)
    
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    raw_csv_path = f'{save_dir}/housing.csv'
    save_path = f'{save_dir}/data.csv'
    
    # Download file dari Kaggle mirror jika belum ada
    if not os.path.exists(raw_csv_path):
        print(f'Downloading housing.csv from Kaggle source...')
        request.urlretrieve(url, raw_csv_path)
    
    df = pd.read_csv(raw_csv_path)
    
    # EKSPLISIT: Buang kolom 'ocean_proximity' agar tersisa 9 kolom numerik murni (sesuai Paper)
    if 'ocean_proximity' in df.columns:
        df = df.drop(columns=['ocean_proximity'])
        
    df.to_csv(save_path, index=False)
    print(f'California data processed with 9 numerical columns. Saved to {save_path}')
# =====================================================================

def process_adult():
    path = f'{DATA_DIR}/adult/adult.data'
    save_path = f'{DATA_DIR}/adult/data.csv'
    data_df = pd.read_csv(path, header=None)
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)

def process_gesture():
    file_names = ['a1_va3', 'a2_va3', 'a3_va3', 'b1_va3', 'b1_va3', 'c1_va3', 'c3_va3']
    datas = []
    for name in file_names:
        df = pd.read_csv(f'{DATA_DIR}/gesture/{name}.csv')
        data = df.to_numpy()
        datas.append(data)
    data = np.concatenate(datas, axis=0)
    data_df = pd.DataFrame(data)
    data_df.to_csv(f'{DATA_DIR}/gesture/data.csv', index=False)

def process_letter():
    dataname = 'letter'
    path = f'{DATA_DIR}/{dataname}/{dataname}-recognition.data'
    save_path = f'{DATA_DIR}/{dataname}/data.csv'
    df = pd.read_csv(path, header=None)
    cols = df.columns.tolist()
    cols = cols[1:] + cols[:1]
    df = df[cols]
    df.to_csv(save_path, index=False, header=True)

def process_news():
    path = f'{DATA_DIR}/news/OnlineNewsPopularity/OnlineNewsPopularity.csv'
    save_path = f'{DATA_DIR}/news/data.csv'
    data_df = pd.read_csv(path)
    data_df = data_df.drop('url', axis=1)
    columns = np.array(data_df.columns.tolist())
    cat_columns1 = columns[list(range(12,18))]
    cat_columns2 = columns[list(range(30,38))]
    cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis=1)
    cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis=1)
    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)
    data_df['data_channel'] = cat_col1
    data_df['weekday'] = cat_col2
    data_df.to_csv(f'{save_path}', index=False)

def process_shoppers():
    path = f'{DATA_DIR}/shoppers/online_shoppers_intention.csv'
    save_path = f'{DATA_DIR}/shoppers/data.csv'
    data_df = pd.read_csv(path)
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)

def process_default():
    path = f'{DATA_DIR}/default/default of credit card clients.xls'
    save_path = f'{DATA_DIR}/default/data.csv'
    data_df = pd.read_excel(path, sheet_name='Data', header=1)
    data_df = data_df.drop('ID', axis=1)
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)

def process_magic():
    path = f'{DATA_DIR}/magic/magic04.data'
    save_path = f'{DATA_DIR}/magic/data.csv'
    data_df = pd.read_csv(path, header=None)
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)

def process_bean():
    path = f'{DATA_DIR}/bean/DryBeanDataset/Dry_Bean_Dataset.xlsx'
    save_path = f'{DATA_DIR}/bean/data.csv'
    data_df = pd.read_excel(path, sheet_name='Dry_Beans_Dataset', header=1)
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(save_path, index=False)

# =====================================================================
# PEMISAHAN DATA TRAIN & TEST (DISESUAIKAN UNTUK 9 KOLOM NUMERIK)
# =====================================================================
def train_test_split(dataname, ratio=0.7, mask_prob=0.3):
    data_dir = f'{DATA_DIR}/{dataname}'
    path = f'{DATA_DIR}/{dataname}/data.csv'
    info_path = f'{DATA_DIR}/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    cat_idx = info['cat_col_idx']
    num_idx = info['num_col_idx']

    data_df = pd.read_csv(path)
    total_num = data_df.shape[0]

    if len(cat_idx) == 0:
        # Mengambil kolom numerik sesuai num_idx
        data_values = data_df.iloc[:, num_idx].values.astype(np.float32)

        nan_idx = np.isnan(data_values).nonzero()[0]

        keep_idx = list(set(np.arange(data_values.shape[0])) - set(list(nan_idx)))
        keep_idx = np.array(keep_idx)
    else:
        keep_idx = np.arange(total_num)

    num_train = int(keep_idx.shape[0] * ratio)
    num_test = total_num - num_train
    seed = 1234

    np.random.seed(seed)
    np.random.shuffle(keep_idx)

    train_idx = keep_idx[:num_train]
    test_idx = keep_idx[-num_test:]

    train_df = data_df.loc[train_idx]
    test_df = data_df.loc[test_idx]        

    train_path = f'{data_dir}/train.csv'
    test_path = f'{data_dir}/test.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f'Spliting Training and Testing data for {dataname} is done.')
    print(f'Training data shape: {train_df.shape}, Testing data shape: {test_df.shape}')


if __name__ == '__main__':

    # 1. Download & Process UCI datasets
    for name in NAME_URL_DICT_UCI.keys():
        download_from_uci(name)

    for name in NAME_URL_DICT_UCI.keys():
        eval(f'process_{name}()')
        train_test_split(name, ratio=0.7, mask_prob=0.3)
        for mask_type in ['MCAR', 'MAR', 'MNAR_logistic_T2']:
            for mask_p in [0.3]:
                generate_mask(dataname=name, mask_type=mask_type, mask_num=10, p=mask_p)
    
    # 2. Download & Process California Housing (Sesuai Paper)
    process_california()
    train_test_split('california', ratio=0.7, mask_prob=0.3)
    for mask_type in ['MCAR', 'MAR', 'MNAR_logistic_T2']:
        for mask_p in [0.3]:
            generate_mask(dataname='california', mask_type=mask_type, mask_num=10, p=mask_p)
