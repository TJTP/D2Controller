import logging
from tqdm import tqdm
import csv
import os
import pickle

import torch

from utils.k_dataset import *
from utils.template import *
from utils.parser import *
from utils.distribution_store import *
from utils.llm_utils import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To suppress warnings about parallelism in tokenizers
logger = logging.getLogger('logger')
logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", 
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
logger.setLevel(logging.INFO)

# ==========================================================
if __name__ == "__main__":
    # parse args
    args = parse_args()
    logger.info(args)

    '''============ configure (retrieval) model ============'''
    model_path = args.rtrv_model_dir if 'opt-30b' in args.llm_dir else args.llm_dir
    assert model_path is not None
    model_name, model, tokenizer, max_context_len = configure_load_model(model_path, gpu_id=args.gpu_id)
    
    '''============ load data ============'''
    train_data = configure_load_dataset(args, mode='train')
    train_data.divide_examples_by_cls()

    '''============ compute & save prob distribution of train examples by class ============'''
    logger.info('<%s> Computing probability distributions of train examples...'%(args.dataset))
    train_data.init_store(dim_size=50272 if model_name == 'opt-2.7b' else 50257)

    logger.info('The retrieval model: %s'%(model_name))
    for cls in train_data.data_by_cls.keys():
        n_example = len(train_data.data_by_cls[cls])
        for idx in tqdm(iterable=range(n_example), total=n_example, desc='Class-%s'%(cls), leave=True):
            distri_prompt = make_prompt(train_data.data_by_cls[cls][idx], args.dataset, mode='distribution')
            distribution = llm_gen(model, distri_prompt, tokenizer, max_context_len) # no multi cards when computing store
            train_data.store_one_distribution(cls, idx, torch.softmax(distribution, dim=-1))
    
    
    '''============ [Optional] configure multi cards training ============'''
    if args.multi_gpu:
        model_name, model, tokenizer, max_context_len = configure_load_model(args.llm_dir, use_multi_gpu=True)
        assert model_name == 'opt-30b' # only 30b model needs to use mutli cards
    else:
        if 'opt-30b' in args.llm_dir:
            model_name, model, tokenizer, max_context_len = configure_load_model(args.llm_dir, gpu_id=args.gpu_id)

    '''============ sample examples for each shot & get candidate examples for eval set by class'''
    '''      & get distributions for sampled examples of each shot & compute class centers ============'''

    train_data.sample_each_shot_by_seed_cls(args, logger)
    
    '''============ evaluate on each shot ============'''
    label2id, label2verb, id2verb = train_data.label2id, train_data.label2verb, train_data.id2verb

    n_seed = len(args.seed_sample_list)
    logger.info('**** Dataset: <%s> ****'%(args.dataset))
    logger.info('The running model: %s'%(model_name))
    logger.info('Selecting eval examples by [%s]'%(args.dis_type))
    
    nshot_acc_list = []
    for n_shot in tqdm(iterable=args.n_shot_list, desc='Eval each shot', leave=True):
        logger.info('Evaluating on %d-shot..'%(n_shot))
        eval_examples = []

        for seed in args.seed_sample_list:
            if args.dis_type != 'random':
                # select nearest examples for current shot and seed
                nearest_idx_each_cls_by_seed = train_data.find_nearest_instance_each_cls_by_seed(args, n_shot, seed)
            
            for cls in train_data.data_by_cls.keys():
                if args.dis_type != 'random':
                    new_idx = nearest_idx_each_cls_by_seed[cls]
                    old_idx = train_data.new_old_ids_each_cls[cls][new_idx]
                    eval_examples.append(train_data.data_by_cls[cls][old_idx])
                else:
                    # ==========randomly sampled n eval samples===============
                    random.seed(seed)
                    sampled_idx = random.randint(0, len(train_data.new_old_ids_each_cls[cls].keys())-1)
                    old_idx = train_data.new_old_ids_each_cls[cls][sampled_idx]
                    eval_examples.append(train_data.data_by_cls[cls][old_idx]) 
        
        total_acc = 0.

        for seed in args.seed_sample_list:
            # merge train examples of all classes together & shuffle them
            train_examples = []

            for cls in train_data.data_by_cls.keys():
                train_examples.extend(train_data.train_examples_each_seed_cls_by_shot[n_shot][seed][cls])

            random.seed(seed)
            random.shuffle(train_examples)

            prompt_prefix = make_prompt(train_examples, args.dataset, mode='train', label2verb=label2verb)
            gt_labels, seed_pred_labels = [], []
            for ins in eval_examples:
                prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
                distribution = llm_gen(model, prompt, tokenizer, max_context_len)
                pred_label = parse_response(args, distribution, tokenizer, id2verb)
                gt_labels.append(label2id[ins['label']])
                seed_pred_labels.append(pred_label)

            seed_acc = sum([1 if gt_labels[i] == seed_pred_labels[i] else 0 for i in range(len(gt_labels))]) / len(gt_labels)
            total_acc += seed_acc

        total_acc /= n_seed
        nshot_acc_list.append((n_shot, total_acc))
    
    nshot_acc_list = sorted(nshot_acc_list, key=lambda x:x[1], reverse=True)
    logger.info('Sorted shot list: %s'%(nshot_acc_list))
    chosen_k = nshot_acc_list[0][0]
    
    logger.info('The chosen number of shots: %d, the accuracy: %f'%(chosen_k, nshot_acc_list[0][1]))
    
    # save results
    save_results_file = os.path.join(args.output_dir, 'results_chosenk_%s_%s.csv'%(model_name, args.dis_type))
    csv_exists = os.path.isfile(save_results_file)
    with open(save_results_file, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['dataset', 'llm', 'chosen_shot', 'nshot_acc', 'test_seeds'])
        csvwriter.writerow([args.dataset,
                            model_name,
                            chosen_k,
                            nshot_acc_list,
                            args.seed_sample_list,
                            ])