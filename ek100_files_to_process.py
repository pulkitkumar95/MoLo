import pickle as pkl
import pandas as pd
import os
import glob


if __name__ == '__main__':
    main_anno_base_path = '/fs/cfar-projects/actionloc/bounce_back/process_datasets/epic-kitchens-100-annotations'
    splits_to_process = ['train','validation']
    path_annotations_pickle = [os.path.join(main_anno_base_path, f'EPIC_100_{split}.pkl') for split in splits_to_process]
    few_shot_anno_paths = glob.glob('configs/projects/hyrsmplusplus/epic_kitchens/*.txt')
    few_shot_df = []
    for anno_path in few_shot_anno_paths:
        anno_split = anno_path.split('/')[-1].split('_')[0]

        df = pd.read_csv(anno_path, header=None)
        df.columns = ['few_shot_file_name']
        df['split'] = anno_split
        df['label_id'] = df['few_shot_file_name'].apply(lambda x: int(x.split('//')[0].replace(anno_split, '')))
        df['narration_id'] = df['few_shot_file_name'].apply(lambda x: x.split('/')[-1].split('.')[0])
        few_shot_df.append(df)
    few_shot_df = pd.concat(few_shot_df).reset_index(drop=True)

    anno_dfs = []
    for file in path_annotations_pickle:
        anno_df = pd.read_pickle(file)
        anno_df.reset_index(inplace=True)
        anno_dfs.append(anno_df)
    anno_df = pd.concat(anno_dfs).reset_index(drop=True)
    few_shot_anno_df = few_shot_df.merge(anno_df, on='narration_id')
    few_shot_anno_df.set_index('narration_id', inplace=True)
    for split in few_shot_anno_df['split'].unique():
        split_df = few_shot_anno_df[few_shot_anno_df['split'] == split]
        dump_path = os.path.join(main_anno_base_path, f'EPIC_100_{split}_few_shot.pkl')
        split_df.to_pickle(dump_path)
