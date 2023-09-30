import json
import random
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from utils.distribution_store import *
from utils.template import *

def configure_load_dataset(args, mode):
    # prepare dataset
    if args.dataset == 'sst2':
        AutoDataset = SST2Dataset
    elif args.dataset == 'subj':
        AutoDataset = SUBJDataset
    elif args.dataset == 'agnews':
        AutoDataset = AGNEWSDataset
    elif args.dataset == 'cb':
        AutoDataset = CBDataset
    elif args.dataset == 'cr':
        AutoDataset = CRDataset
    elif args.dataset == 'dbpedia':
        AutoDataset = DBPEDIADataset
    elif args.dataset == 'mpqa':
        AutoDataset = MPQADataset
    elif args.dataset == 'mr':
        AutoDataset = MRDataset
    elif args.dataset == 'rte':
        AutoDataset = RTEDataset
    elif args.dataset == 'sst5':
        AutoDataset = SST5Dataset
    else:
        raise ValueError
    
    data = AutoDataset(os.path.join(args.data_dir, args.dataset), mode=mode)

    return data

class BaseDataset:
    def __init__(self, data_dir, mode):
        """data key: sentence, label[0/1]"""
        super().__init__()

        if mode == 'dev': mode = 'dev_subsample'
        elif mode == 'train': mode += '_idx'
        elif mode == 'train_subset_idx': pass
        else: raise ValueError
        self.data_file = os.path.join(data_dir, mode + '.jsonl')

        self.data = []
        with open(self.data_file, 'r') as f:
            read_lines = f.readlines()
            for line in read_lines:
                instance = json.loads(line.strip())
                self.data.append(instance)
        
        # customize your own label map in inheritance
        self.id2label = {0: 'negative', 1: 'positive'}
        self.label2id = {'negative': 0, 'positive': 1}

    def __len__(self):
        return len(self.data)
    
    "================== for select k & icl =================="
    def divide_examples_by_cls(self):
        self.data_by_cls = defaultdict(list) # {cls1: [...............],
                                             #  cls2: [...............]}
        for i in range(self.__len__()):
            label = self.data[i]['label']
            cls= self.label2id[label]
            self.data_by_cls[cls].append(self.data[i])

        assert len(self.data_by_cls.keys()) == len(self.label2id.keys())
        self.n_class = len(self.data_by_cls.keys())
    
    "================== for icl =================="
    def sample_icl_train_examples(self, seed, n_shot):
        random.seed(seed)
        train_examples = []
        
        for cls in self.data_by_cls.keys():
            train_samples_by_cls = random.sample(self.data_by_cls[cls], min(n_shot, len(self.data_by_cls[cls])))
            train_examples.extend(train_samples_by_cls)

        self.data = train_examples
    
    def shuffle_icl_train_examples(self):
        random.shuffle(self.data)

    "================== for select k =================="
    def init_store(self, dim_size=50257):
        # init distribution store for representations
        self.distribution_store = DistributionStore(self.data_by_cls, self.n_class, dim_size)
        
    def store_one_distribution(self, cls, idx, distribution):
        self.distribution_store.store_one_distribution(cls, idx, distribution)
    
    def sample_each_shot_by_seed_cls(self, args, logger):   
        self.train_examples_each_seed_cls_by_shot = defaultdict(dict) # {shot1: {seed1: {cls1: [...], cls2: [...]}}, {seed2: {cls1: [...], cls2: [...]}}
                                                                      #  shot2: {seed1: {cls1: [...], cls2: [...]}}, {seed2: {cls1: [...], cls2: [...]}},
                                                                      #  shot3: {seed1: {cls1: [...], cls2: [...]}}, {seed2: {cls1: [...], cls2: [...]}}}
        
        excluded_indices_each_cls = defaultdict(set) # {cls1: {...............},
                                                     #  cls2: {...............}}
        
        logger.info('Sampling train examples for each shot on seeds: %s...'%(args.seed_sample_list))

        for n_shot in args.n_shot_list:
            seed_cls_centers = {} # {seed1: {cls1: [...], cls2: [...]},
                                  #  seed2: {cls1: [...], cls2: [...]},
                                  #  seed3: {cls1: [...], cls2: [...]}}

            for seed in tqdm(iterable=args.seed_sample_list, desc='%d-shot'%(n_shot), leave=True):
                random.seed(seed)
                cls_centers = {} #  {cls1: ..., cls2: ...}
                self.train_examples_each_seed_cls_by_shot[n_shot][seed] = {}

                for cls in self.data_by_cls.keys():
                    # set train samples
                    train_samples_by_cls = random.sample(self.data_by_cls[cls], min(n_shot, len(self.data_by_cls[cls])))
                    self.train_examples_each_seed_cls_by_shot[n_shot][seed][cls] = train_samples_by_cls

                    chosen_indices = []
                    if len(self.data_by_cls[cls]) > 16 or len(args.seed_sample_list) <=5: # for CB dataset
                        for ins in train_samples_by_cls:
                            chosen_indices.append(ins['idx'])
                    # record the sampled indices 
                    excluded_indices_each_cls[cls].update(set(chosen_indices)) 
                    # set class center
                    cls_centers[cls] = self.compute_cls_center_by_shot(cls, n_shot, chosen_indices=np.array(chosen_indices))     
                seed_cls_centers[seed] = cls_centers
            self.set_cls_centers_by_shot(n_shot, seed_cls_centers)

        logger.info('Construting candidate set for choosing eval examples...')
        self.new_old_ids_each_cls = dict()
        for cls in self.data_by_cls.keys():
            # remap indices of remaining examples 
            remain_indices = set(range(len(self.data_by_cls[cls]))) - excluded_indices_each_cls[cls]
            remain_indices = sorted(list(remain_indices))
            new_indices = list(range(len(remain_indices)))
            self.new_old_ids_each_cls[cls] = dict(zip(new_indices, remain_indices))

            # delete distributions of chosen examples
            self.filter_out_chosen_distributions(cls, np.array(remain_indices))
    
    def compute_cls_center_by_shot(self, cls, n_shot, chosen_indices):
        return self.distribution_store.compute_cls_center_by_shot(cls, n_shot, chosen_indices)
    
    def set_cls_centers_by_shot(self, n_shot, seed_cls_centers):
        self.distribution_store.set_cls_centers_by_shot(n_shot, seed_cls_centers)
    
    def filter_out_chosen_distributions(self, cls, remain_indices):
        self.distribution_store.filter_out_chosen_distributions(cls, remain_indices)
    
    def find_nearest_instance_each_cls_by_seed(self, args, n_shot, seed):
        return self.distribution_store.find_nearest_instance_each_cls_by_seed(args, n_shot, seed)
    
class SST2Dataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']


class SUBJDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        # subj only has test set
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'subjective', '1': 'objective'}
        self.id2verb = ['subjective', 'objective']


class AGNEWSDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'1': 0, '2': 1, '3': 2, '4': 3}
        self.label2verb = {'1': 'world', '2': 'sports', '3': 'business', '4': 'technology'}
        self.id2verb = ['world', 'sports', 'business', 'technology']


class CBDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
        self.label2verb = {'contradiction': 'false', 'entailment': 'true', 'neutral': 'neither'}
        self.id2verb = ['false', 'true', 'neither']


class CRDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']


class DBPEDIADataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        if mode == 'train':
            mode = 'train_subset_idx'  # this is an exception case
        super().__init__(data_dir, mode)
        self.label2id = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
                         '6': 5, '7': 6, '8': 7, '9': 8, '10': 9,
                         '11': 10, '12': 11, '13': 12, '14': 13}
        self.label2verb = {'1': 'company', '2': 'school', '3': 'artist', '4': 'athlete', '5': 'politics',
                           '6': 'transportation', '7': 'building', '8': 'nature', '9': 'village', '10': 'animal',
                           '11': 'plant', '12': 'album', '13': 'film', '14': 'book'}
        self.id2verb = ['company', 'school', 'artist', 'athlete', 'politics',
                        'transportation', 'building', 'nature', 'village', 'animal',
                        'plant', 'album', 'film', 'book']


class MPQADataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']


class MRDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']


class RTEDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'not_entailment': 0, 'entailment': 1}
        self.label2verb = {'not_entailment': 'false', 'entailment': 'true'}
        self.id2verb = ['false', 'true']


class SST5Dataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        mode
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.label2id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        self.label2verb = {'0': 'terrible', '1': 'bad', '2': 'okay', '3': 'good', '4': 'great'}
        self.id2verb = ['terrible', 'bad', 'okay', 'good', 'great']