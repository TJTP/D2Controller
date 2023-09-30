def make_prompt(examples, dataset_name, mode, label2verb=None):
    if dataset_name == 'sst2':
        template_func = template_sst2
    elif dataset_name == 'subj':
        template_func = template_subj
    elif dataset_name == 'agnews':
        template_func = template_agnews
    elif dataset_name == 'cb':
        template_func = template_cb
    elif dataset_name == 'cr':
        template_func = template_cr
    elif dataset_name == 'dbpedia':
        template_func = template_dbpedia
    elif dataset_name == 'mpqa':
        template_func = template_mpqa
    elif dataset_name == 'mr':
        template_func = template_mr
    elif dataset_name == 'rte':
        template_func = template_rte
    elif dataset_name == 'sst5':
        template_func = template_sst5
    
    if mode == 'inference':
        return template_func(examples, None, mode)
    if mode == 'calibration':
        return template_func(examples, None, mode)
    elif mode == 'distribution':
        return template_func(examples, None, mode)
    elif mode == 'train':
        prompt = ''
        for ins in examples:
            prompt += template_func(ins, label2verb[ins['label']], mode)
            prompt += '\n'
        return prompt
    else:
        raise ValueError

def template_sst2(ins, label, mode):
    if mode == 'train':
        return f"Review: {ins['sentence']}\nSentiment: {label}\n"
    elif mode == 'calibration':
        return f"Review: {ins}\nSentiment:"
    elif mode == 'distribution':
        return ins['sentence']
    elif mode == 'inference':
        return f"Review: {ins['sentence']}\nSentiment:"
    else:
        raise ValueError


def template_subj(ins, label, mode):
    if mode == 'train':
        return f"Input: {ins['sentence']}\nType: {label}\n"
    elif mode == 'calibration':
        return f"Input: {ins}\nType:"
    elif mode == 'distribution':
        return ins['sentence']
    elif mode == 'inference':
        return f"Input: {ins['sentence']}\nType:"
    else:
        raise ValueError


def template_agnews(ins, label, mode):
    if mode == 'train':
        return f"input: {ins['sentence']}\ntype: {label}\n"
    elif mode == 'calibration':
        return f"input: {ins}\ntype:"
    elif mode == 'distribution':
        return ins['sentence']
    elif mode == 'inference':
        return f"input: {ins['sentence']}\ntype:"
    else:
        raise ValueError


def template_cb(ins, label, mode):
    if mode == 'train':
        return f"premise: {ins['premise']}\nhypothesis: {ins['hypothesis']}\nprediction: {label}\n"
    elif mode == 'calibration':
        return f"premise: {ins}\nhypothesis: {ins}\nprediction:"
    elif mode == 'distribution':
        return ins['premise'] + ins['hypothesis']
    elif mode == 'inference':
        return f"premise: {ins['premise']}\nhypothesis: {ins['hypothesis']}\nprediction:"
    else:
        raise ValueError


def template_cr(ins, label, mode):
    if mode == 'train':
        return f"Review: {ins['sentence']}\nSentiment: {label}\n"
    elif mode == 'calibration':
        return f"Review: {ins}\nSentiment:"
    elif mode == 'distribution':
        return ins['sentence']
    elif mode == 'inference':
        return f"Review: {ins['sentence']}\nSentiment:"
    else:
        raise ValueError


def template_dbpedia(ins, label, mode):
    if mode == 'train':
        return f"input: {ins['sentence']}\ntype: {label}\n"
    elif mode == 'calibration':
        return f"input: {ins}\ntype:"
    elif mode == 'distribution':
        return ins['sentence']
    elif mode == 'inference':
        return f"input: {ins['sentence']}\ntype:"
    else:
        raise ValueError


def template_mpqa(ins, label, mode):
    if mode == 'train':
        return f"Review: {ins['sentence']}\nSentiment: {label}\n"
    elif mode == 'calibration':
        return f"Review: {ins}\nSentiment:"
    elif mode == 'distribution':
        return ins['sentence']
    elif 'inference':
        return f"Review: {ins['sentence']}\nSentiment:"
    else:
        raise ValueError


def template_mr(ins, label, mode):
    if mode == 'train':
        return f"Review: {ins['sentence']}\nSentiment: {label}\n"
    elif mode == 'calibration':
        return f"Review: {ins}\nSentiment:"
    elif mode == 'distribution':
        return ins['sentence']
    elif mode == 'inference':
        return f"Review: {ins['sentence']}\nSentiment:"
    else:
        raise ValueError


def template_rte(ins, label, mode):
    if mode == 'train':
        return f"premise: {ins['sentence_1']}\nhypothesis: {ins['sentence_2']}\nprediction: {label}\n"
    elif mode == 'calibration':
        return f"premise: {ins}\nhypothesis: {ins}\nprediction:"
    elif mode == 'distribution':
        return ins['sentence_1'] + ins['sentence_2']
    elif mode == 'inference':
        return f"premise: {ins['sentence_1']}\nhypothesis: {ins['sentence_2']}\nprediction:"
    else:
        raise ValueError


def template_sst5(ins, label, mode):
    if mode == 'train':
        return f"Review: {ins['sentence']}\nSentiment: {label}\n"
    elif mode == 'calibration':
        return f"Review: {ins}\nSentiment:"
    elif mode == 'distribution':
        return ins['sentence']
    elif mode == 'inference':
        return f"Review: {ins['sentence']}\nSentiment:"
    else:
        raise ValueError