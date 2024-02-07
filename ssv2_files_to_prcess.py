import pandas as pd
import os
import glob

if __name__ == '__main__':
    vids_processed = glob.glob('/fs/vulcan-projects/VidGram/ssv2_mp4/*')
    processed_df = pd.DataFrame(vids_processed, columns=['vid'])
    processed_df['id'] = processed_df['vid'].apply(lambda x: x.split('/')[-1].split('.')[0])
    dataset = 'ssv2_full'
    molo_config_path = f'/fs/cfar-projects/actionloc/bounce_back/MoLo/configs/projects/MoLo/{dataset}'
    molo_txt_files = glob.glob(f'{molo_config_path}/*.txt')
    all_vids_to_be_processed = []
    for file in molo_txt_files:
        df = pd.read_csv(file, header=None)
        df.columns = ['vid']
        df['id'] = df['vid'].apply(lambda x: x.split('/')[-1])
        all_vids_to_be_processed.extend(df['id'].tolist())
    procesed = set(processed_df['id'].tolist())
    to_process = set(all_vids_to_be_processed)
    to_process = to_process - procesed
    to_process = list(to_process)
    print(f'Number of videos to be processed: {len(to_process)}')
    df = pd.DataFrame(to_process, columns=['id'])
    df.to_csv(f'../process_datasets/molo_{dataset}.csv')
    