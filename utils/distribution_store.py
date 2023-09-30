import torch 
from torch import nn 
from collections import defaultdict

class DistributionStore(nn.Module):
    def __init__(self, data_by_cls, n_class, dim_size=50257):
        super(DistributionStore, self).__init__()

        self.n_class = n_class
        self.dim_size = dim_size

        self.cls_keys = list(data_by_cls.keys())

        for cls in self.cls_keys:
            setattr(self, 'distri_reps_cls_%d'%(cls), torch.randn(len(data_by_cls[cls]), 
                                                                         self.dim_size))
        
        self.cls_centers_by_shot = defaultdict(dict) # {shot1: {seed1: {cls1: ..., cls2, ...},
                                                     #          seed2: {cls1: ..., cls2, ...}}}
    
    def store_one_distribution(self, cls, idx, distribution):
        getattr(self, 'distri_reps_cls_%d'%(cls))[idx] = distribution
    
    def compute_cls_center_by_shot(self, cls, n_shot, chosen_indices):
        train_example_distributions = getattr(self, 'distri_reps_cls_%d'%(cls))[chosen_indices, :] # [n_shot, dim_size]
        cls_center = torch.sum(train_example_distributions, dim=0, keepdim=True) / n_shot # [1, dim_size]
        return cls_center

    def set_cls_centers_by_shot(self, n_shot, seed_cls_centers):
        self.cls_centers_by_shot[n_shot] = seed_cls_centers
    
    def filter_out_chosen_distributions(self, cls, remain_indices):
        filtered_distributions = getattr(self, 'distri_reps_cls_%d'%(cls))[remain_indices, :]
        setattr(self, 'distri_reps_cls_%d'%(cls), filtered_distributions)
    
    def find_nearest_instance_each_cls_by_seed(self, args, n_shot, seed):
        def kl_divergence(distri_reps, cls_center):
            distance = torch.mean(
                    distri_reps[:, None, :] * (distri_reps[:, None, :].log() - cls_center.log()), dim=2
                ).transpose(1,0) # [1, len(data_by_cls[cls])]
            
            return distance

        def euclidean_distance(distri_reps, cls_center):
            distance = torch.sum((distri_reps - cls_center)**2, dim=1, keepdim=True).sqrt().transpose(1, 0) # [1, len(data_by_cls[cls])]
            return distance

        def neg_cos_similarity(distri_reps, cls_center):
            distance = -torch.cosine_similarity(distri_reps, cls_center, dim=1).unsqueeze(dim=0) # [1, len(data_by_cls[cls])]
            return distance

        idx_dict = {}
        cls_centers = self.cls_centers_by_shot[n_shot][seed]
        
        for cls0 in self.cls_keys:
            cls0_center = cls_centers[cls0]
            
            if args.dis_type[:8] == 'iicscore':
                if args.dis_type[-3:] == '-eu':
                    intra_distance = euclidean_distance(getattr(self, 'distri_reps_cls_%d'%(cls0)), cls0_center)
                elif args.dis_type[-4:] == '-cos':
                    intra_distance = neg_cos_similarity(getattr(self, 'distri_reps_cls_%d'%(cls0)), cls0_center)
                else:
                    intra_distance = kl_divergence(getattr(self, 'distri_reps_cls_%d'%(cls0)), cls0_center)
                
                norm_factor = 0
                inter_cls_distance = 0.
                
                for cls1 in self.cls_keys:
                    n_cls1_instance = getattr(self, 'distri_reps_cls_%d'%cls1).shape[0]
                    norm_factor += n_cls1_instance
                    if cls1 == cls0: continue
                    
                    cls1_center = cls_centers[cls1]

                    if args.dis_type[-3:] == '-eu':
                        inter_distance = euclidean_distance(getattr(self, 'distri_reps_cls_%d'%cls0), cls1_center)
                    elif args.dis_type[-4:] == '-cos':
                        inter_distance = neg_cos_similarity(getattr(self, 'distri_reps_cls_%d'%cls0), cls1_center)
                    else:
                        inter_distance = kl_divergence(getattr(self, 'distri_reps_cls_%d'%cls0), cls1_center)
                    
                    
                    inter_cls_distance +=  n_cls1_instance * inter_distance
                    
                distance = -intra_distance + inter_cls_distance / norm_factor # [1, len(data_by_cls[cls])]
                
                idx_dict[cls0] = torch.argmax(distance, dim=1).tolist()[0]
                
            elif args.dis_type[:9] == 'intra-dis':
                if args.dis_type[-3:] == '-eu':
                    distance = euclidean_distance(getattr(self, 'distri_reps_cls_%d'%cls0), cls0_center)
                elif args.dis_type[-4:] == '-cos':
                    distance = neg_cos_similarity(getattr(self, 'distri_reps_cls_%d'%cls0), cls0_center)
                else:
                    distance = kl_divergence(getattr(self, 'distri_reps_cls_%d'%cls0), cls0_center)

                idx_dict[cls0] = torch.argmin(distance, dim=1).tolist()[0]
            
            else:
                raise ValueError


        return idx_dict
    
    