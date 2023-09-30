import json
import os
from argparse import ArgumentParser



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/'
    )
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        default=None
    )
    args = parser.parse_args()
    train_data_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset == 'dbpedia':
        train_data_file = os.path.join(train_data_dir, 'train_subset.jsonl')
    else:
        train_data_file = os.path.join(train_data_dir, 'train.jsonl')

    
    class_idx_cnter = {}
    f1 = open(os.path.join(train_data_dir, 'train_idx.jsonl'), 'w') if args.dataset != 'dbpedia' else open(os.path.join(train_data_dir, 'train_subset_idx.jsonl'), 'w')
    with open(train_data_file, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            example = json.loads(line.strip())
            if example['label'] not in class_idx_cnter:
                class_idx_cnter[example['label']] = 0
            example['idx'] = class_idx_cnter[example['label']]
            
            json.dump(example, f1)
            f1.write('\n')
            
            class_idx_cnter[example['label']] += 1
            
    f.close()
    f1.close()
    
