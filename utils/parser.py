import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="For ICL baselines & select_k.")
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--llm_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--multi_gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--local_rank",
        type=int
    )
    parser.add_argument(
        "--n_shot_list",
        type=str,
        default='[4]',
    )
    parser.add_argument(
        "--seed_sample_list",
        type=str,
        default='[1, 2, 3, 4, 5]'
    )
    
    
    
    parser.add_argument(
        "--rtrv_model_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_store",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--store_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dis_type",
        type=str,
        default='iicscore',
    )
    


    parser.add_argument(
        "--additional_str",
        type=str,
        default=''
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    
    args = parser.parse_args()
    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.store_dir is not None:
        os.makedirs(args.store_dir, exist_ok=True)
    
    if args.n_shot_list is not None:
        args.n_shot_list = eval(args.n_shot_list)
    if args.seed_sample_list is not None: 
        args.seed_sample_list = eval(args.seed_sample_list)
    

    return args