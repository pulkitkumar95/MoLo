import pandas as pd
import os
import shutil
import glob
from tqdm import tqdm 

def transfer(row, new_base_path):
    old_base_path = '/fs/nexus-scratch/pulkit/feats_sans_tskip_sans_feat/kinetics'
    
    features_to_transfer = 'cotrack_fps_10_grid_16'
    new_vid_base_path = os.path.join(new_base_path, 'videos')
    feat_vid_name = row['feat_base_name']
    old_feat_path = os.path.join(old_base_path, features_to_transfer, feat_vid_name)
    new_feat_path = os.path.join(new_base_path, features_to_transfer, feat_vid_name)

    base_new_vid_name = feat_vid_name.replace('pkl', 'mp4')
    base_vid_path = os.path.dirname(base_new_vid_name)
    vid_dir_path = os.path.join(new_vid_base_path, base_vid_path)
    vid_path = os.path.join(new_vid_base_path, base_new_vid_name)


    

    if not os.path.exists(old_feat_path):
        breakpoint()
        return vid_path, False
    os.makedirs(os.path.dirname(new_feat_path), exist_ok=True)
    os.makedirs(vid_dir_path, exist_ok=True)
    shutil.copy(old_feat_path, new_feat_path)
    shutil.copy(row['video_path'], vid_path)
    return vid_path, True




def transfer_kinetics():
    original_df = pd.read_csv('/fs/nexus-scratch/pulkit/feats_sans_tskip_sans_feat/kinetics/kinetics400_with_few_shot_missing.csv')
    new_base_path = '/fs/nexus-scratch/pulkit/feats_sans_tskip_sans_feat/kinetics_molo/'
    csv_dir_path = os.path.join(new_base_path, 'data_csvs')
    os.makedirs(csv_dir_path, exist_ok=True)
    original_df.rename(columns={'id': 'og_id', 'label_id': 'og_label_id', 'split': 'og_split'}, inplace=True)
    dataset = 'kinetics100'
    molo_config_path = f'/fs/cfar-projects/actionloc/bounce_back/MoLo/configs/projects/MoLo/{dataset}'
    molo_txt_files = glob.glob(f'{molo_config_path}/*.txt')
    all_vids_to_be_processed = []
    few_shot_dfs = []
    for file in molo_txt_files:
        split = file.split('/')[-1].split('_')[0]
        df = pd.read_csv(file, header=None)
        df.columns = ['og_name']
        df['few_shot_vid_name'] = df['og_name'].apply(lambda x: x.split('/')[-1].split('.')[0])
        df['vid_id'] = df['few_shot_vid_name'].apply(lambda x: x[:11])
        df['split'] = split
        df['label_id'] = df['og_name'].apply(lambda x: int(x.split('//')[0].replace(split, '')))
        few_shot_dfs.append(df)
    few_shot_df = pd.concat(few_shot_dfs).reset_index(drop=True)
    few_shot_vid_ids = set(few_shot_df['vid_id'].tolist())
    original_vid_ids = set(original_df['vid_id'].tolist())
    missing_vid_ids = few_shot_vid_ids - original_vid_ids
    missing_few_shot_df = few_shot_df[few_shot_df['vid_id'].isin(missing_vid_ids)]
    few_shot_df = few_shot_df.merge(original_df, on='vid_id')
    video_paths = []
    feat_exists = []
    
    # for i, row in tqdm(few_shot_df.iterrows()):
    #     video_path, feat_exist = transfer(row, new_base_path)
    #     feat_exists.append(feat_exist)
    #     video_paths.append(video_path)
    few_shot_df['vid_base_path'] = few_shot_df['feat_base_name'].apply(lambda x: os.path.join('videos',x.replace('pkl', 'mp4')))
    # few_shot_df['vid_path_check'] = few_shot_df['vid_base_path'].apply(lambda x: os.path.exists(os.path.join(new_base_path, 'videos', x)))
    # breakpoint()
    # print((~few_shot_df['feat_exists']).sum())
    few_shot_df.to_csv(os.path.join(new_base_path, f'data_csvs/{dataset}.csv'))

def fix_missing_vids():
    original_df = pd.read_csv('/fs/nexus-scratch/pulkit/feats_sans_tskip_sans_feat/kinetics/kinetics400_all.csv')
    label_dict = {}
    for label_name , label_df in original_df.groupby('label_name'):
        label_id = label_df['label_id'].iloc[0]
        some_id = label_df['id'].iloc[0]
        assert label_id == some_id
        label_dict[label_name] = label_id
    main_ds_path  = '/fs/vulcan-datasets/kinetics/'
    dataset = 'kinetics100'
    missing_df = pd.read_csv(f'{dataset}_missing.csv')
    missing_df['path'] = missing_df['og_name'].apply(lambda x: x.split('//')[1])
    missing_df['path'] = missing_df['path'].apply(lambda x: x.replace('_256/', '_256_new/'))
    missing_df['base_dir'] = missing_df['path'].apply(lambda x: os.path.dirname(x))
    missing_df['video_path'] = main_ds_path + missing_df['base_dir'] + '/' + missing_df['vid_id'] + '.mp4'
    missing_df['vid_exists'] = missing_df['video_path'].apply(lambda x: os.path.exists(x))
    missing_df['label_name'] = missing_df['og_name'].apply(lambda x: x.split('//')[1].split('/')[-2].replace('_', ' '))
    missing_df['label_id'] = missing_df['label_name'].apply(lambda x: label_dict[x])
    missing_df['id'] = missing_df['label_id']
    missing_df['vid_name'] = missing_df['video_path'].apply(lambda x: os.path.basename(x))
    missing_df['split'] =  missing_df['video_path'].apply(lambda x: x.split('/')[-3].split('_')[0])
    missing_df['feat_name'] = missing_df['vid_name'].apply(lambda x: x.replace('mp4', 'pkl'))
    missing_df['feat_base_name'] = missing_df['split'] + '/' + missing_df['label_name'] + '/' + missing_df['feat_name']
    missing_df = missing_df[original_df.columns]
    final_df = pd.concat([original_df, missing_df]).reset_index(drop=True)
    final_df.to_csv('/fs/nexus-scratch/pulkit/feats_sans_tskip_sans_feat/kinetics/kinetics400_with_few_shot_missing.csv')
    missing_df.to_csv('../process_datasets/missing_kinetics100.csv')

if __name__ == '__main__':
    transfer_kinetics()
