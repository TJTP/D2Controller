import torch
import os
import deepspeed
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

def configure_load_model(model_dir, use_multi_gpu=False, gpu_id=None):
    # set model name
    if 'gpt2-medium' in model_dir:
        model_name = 'gpt2-medium'
    elif 'gpt2-large' in model_dir:
        model_name = 'gpt2-large'
    elif 'gpt2-xl' in model_dir:
        model_name = 'gpt2-xl'
    elif 'Cerebras-GPT-2.7B' in model_dir:
        model_name = 'Cerebras-GPT-2.7B'
    elif 'Cerebras-GPT-6.7B' in model_dir:
        model_name = 'Cerebras-GPT-6.7B'
    elif 'opt-13b' in model_dir:
        model_name = 'opt-13b'
    elif 'opt-30b' in model_dir:
        model_name = 'opt-30b'
    else:
        raise ValueError
    
    # set maximum context length
    if 'gpt2' in model_dir:
        max_context_len = 1024
    else: 
        max_context_len = 2048
    
    
    model_config = AutoConfig.from_pretrained(model_dir)
    model_config.pad_token_id = model_config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast= False if model_name == 'opt-13b' else True)
    tokenizer.padding_side = "left" # set pad token ids for batched inference cus gpt2 does not have one
    tokenizer.pad_token = tokenizer.eos_token
    if not use_multi_gpu:
        device = torch.device("cuda:%s"%(gpu_id) if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.to(device)
        model.eval()

    else:
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        world_size = int(os.getenv('WORLD_SIZE', '1'))
        model_hidden_size = model_config.hidden_size
        train_batch_size = 1 * world_size
        ds_config = {
            "fp16": {
                "enabled": True
            },
            "bf16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": model_hidden_size * model_hidden_size,
                "stage3_prefetch_bucket_size":
                0.9 * model_hidden_size * model_hidden_size,
                "stage3_param_persistence_threshold": 10 * model_hidden_size
            },
            "steps_per_print": 2000,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }

        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
        ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
        ds_engine.module.eval()

        model = ds_engine.module

    return model_name, model, tokenizer, max_context_len

def llm_gen(model, prompt, tokenizer, max_context_len):
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt", padding=True).to(device=model.device)
    if inputs['input_ids'].shape[1] > max_context_len:
        inputs['input_ids'] = inputs['input_ids'][:, -max_context_len:]
        inputs['attention_mask'] = inputs['attention_mask'][:, -max_context_len:]
    with torch.no_grad():
        logits = model.forward(input_ids=inputs['input_ids'],
                               attention_mask=inputs['attention_mask'],
                               return_dict=True).logits.detach().cpu()
    # the output prob is shifted by -1, so we should use the output at the last input token position
    # gen_logits.shape = [1, 50257]
    gen_logits = logits[:, -1, :]

    return gen_logits

def choose_label_probs(args, gen_logits, tokenizer, id2verb):
    if args.multi_gpu:
        gen_prob = torch.softmax(gen_logits.float(), dim=-1)
    else:
        gen_prob = torch.softmax(gen_logits, dim=-1)
    prob_per_cls = []
    for label_verb in id2verb:
        label_verb_token_id = tokenizer.encode(' ' + label_verb)[-1] # note the space before label word
        prob_per_cls.append(gen_prob[:, label_verb_token_id])
    return torch.cat(prob_per_cls, dim=0) # [n_class, ]

def parse_response(args, gen_logits, tokenizer, id2verb):
    prob_per_cls = choose_label_probs(args, gen_logits, tokenizer, id2verb)
    pred = torch.argmax(prob_per_cls).tolist()
    
    return pred