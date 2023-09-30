import logging
from tqdm import tqdm
import csv
import os

from utils.k_dataset import *
from utils.template import *
from utils.parser import *
from utils.llm_utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To suppress warnings about parallelism in tokenizers
logger = logging.getLogger('logger')
logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", 
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    '''============ configure model ============'''
    model_name, model, tokenizer, max_context_len = configure_load_model(args.llm_dir, use_multi_gpu=args.multi_gpu, gpu_id=args.gpu_id)
    
    logger.info('Model: %s, Dataset: %s'%(model_name, args.dataset))

    '''============ load data ============'''
    train_data = configure_load_dataset(args, mode='train')
    train_data.divide_examples_by_cls()

    label2verb, label2id, id2verb = train_data.label2verb, train_data.label2id, train_data.id2verb

    dev_data = configure_load_dataset(args, mode='dev')
    
    extra_logging_str = ''

    for n_shot in args.n_shot_list:
        for seed_sp in args.seed_sample_list:
            logger.info('<%s>  Sampling train examples of %d-shot on seed: %d'%(args.dataset, n_shot, seed_sp))
            train_data.sample_icl_train_examples(seed_sp, n_shot)
            train_data.shuffle_icl_train_examples()

            prompt_prefix = make_prompt(train_data.data, args.dataset, mode='train', label2verb=label2verb)
                

            dev_labels, dev_pred = [], []
            for ins in tqdm(iterable=dev_data.data, desc='Eval on %d dev examples'%(dev_data.__len__()), leave=True):
                dev_labels.append(label2id[ins['label']])
                prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
                gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
                dev_pred.append(parse_response(args,gen_logits, tokenizer, id2verb))

            dev_correct = [1 if dev_labels[i] == dev_pred[i] else 0 for i in range(len(dev_labels))]
            acc = sum(dev_correct) / len(dev_labels)
            logger.info(f"Acc: {acc}")
    
            # logging
            save_results_file = os.path.join(args.output_dir, 'results_icl_%s%s%s.csv'%(model_name, extra_logging_str, args.additional_str))
            csv_exists = os.path.isfile(save_results_file)
            with open(save_results_file, 'a+', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                if not csv_exists:
                    csvwriter.writerow(['dataset', 'llm', 'n_shot', 'accuracy', 'seed_sampler'])
                csvwriter.writerow([args.dataset,
                                    model_name,
                                    n_shot,
                                    acc,
                                    '<%d>'%(seed_sp)
                                    ])