import os
import glob
import pandas as pd

if __name__ == '__main__':
    base_dir = '/fs/cfar-projects/actionloc/bounce_back/MoLo/configs/projects/MoLo/ssv2_full'
    ds_name = os.path.basename(base_dir)
    final_json_path = '/fs/cfar-projects/actionloc/bounce_back/process_datasets/molo_few_shot/{}'.format(ds_name)
    os.makedirs(final_json_path, exist_ok=True)
    txt_files = glob.glob(os.path.join(base_dir, '*.txt'))
    splits = ['train', 'val', 'test']
    for split in splits:
        txt_file  = os.path.join(base_dir, '{}_few_shot.txt'.format(split))
        json_file_name = '{}_few_shot.json'.format(split)
        df = pd.read_csv(txt_file, names=['path'])
        df['id'] = df['path'].apply(lambda x: x.split('/')[-1])
        df['label'] = df['path'].apply(lambda x: int(x.split('/')[-2].replace(split, '')))
        df.drop(columns=['path'], inplace=True)
        json_data = df.to_json(orient='records')
        json_file = os.path.join(final_json_path, json_file_name)
        with open(json_file, 'w') as f:
            f.write(json_data)
    

    