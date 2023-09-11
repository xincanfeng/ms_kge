#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import json
import model
import logging

from collections import defaultdict

class TrainDataset(Dataset):
    def __init__(self, args, triples, nentity, nrelation, negative_sample_size, mode):
        self.args = args
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.freq_count = self.count_frequency(triples, args.count_start)
        self.score_count = self.ave_query_score_for_significance_test(triples,triples,args)
        if args.mbs_default or args.mbs_freq or args.mbs_uniq:
            # Cannot allow to combine different subsampling methods
            assert((args.mbs_freq^args.mbs_uniq)^args.mbs_default)
            # Cannot allow to do model-based subsampling without any trained model
            assert(args.subsampling_model != None)
            self.mbs_count = self.count_submodel_freq(triples,triples,args)
        
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
        
        if self.args.mbs_default or self.args.mbs_freq or self.args.mbs_uniq:
            # Cannot allow to combine different subsampling methods
            assert((self.args.mbs_freq^self.args.mbs_uniq)^self.args.mbs_default)
            # Cannot allow to do model-based subsampling without any trained model
            assert(self.args.subsampling_model != None)
            return positive_sample, negative_sample, self.mbs_count[(head, relation, tail)], self.mbs_count[(head, relation)], self.mbs_count[(tail, -relation-1)], self.freq_count[(head, relation)], self.freq_count[(tail, -relation-1)], self.score_count[(head, relation)], self.score_count[(tail, -relation-1)], self.mode
        else:
            return positive_sample, negative_sample, self.freq_count[(head, relation)], self.freq_count[(tail, -relation-1)], self.score_count[(head, relation)], self.score_count[(tail, -relation-1)], self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        if len(data[0]) == 5:
            hr_freq = torch.Tensor([_[2] for _ in data])
            tr_freq = torch.Tensor([_[3] for _ in data])
            batch_type = data[0][4]
            return positive_sample, negative_sample, hr_freq, tr_freq, batch_type
        elif len(data[0]) == 8:
            mbs_triple_freq = torch.Tensor([_[2] for _ in data])
            mbs_hr_freq = torch.Tensor([_[3] for _ in data])
            mbs_tr_freq = torch.Tensor([_[4] for _ in data])
            cnt_hr_freq = torch.Tensor([_[5] for _ in data])
            cnt_tr_freq = torch.Tensor([_[6] for _ in data])
            batch_type = data[0][7]
            return positive_sample, negative_sample, mbs_triple_freq, mbs_hr_freq, mbs_tr_freq, cnt_hr_freq, cnt_tr_freq, batch_type
        else:
            assert(False)
    
    @staticmethod
    def count_frequency(triples, start):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def count_submodel_freq(triples, all_true_triples, args):
        '''
        Count submodel-based frequencies
        '''
        # Restore submodel from checkpoint directory
        with open(os.path.join(args.subsampling_model, 'config.json'), 'r') as fjson:
            argparse_dict = json.load(fjson)
        submodel = model.KGEModel(
            model_name=argparse_dict['model'],
            nentity=argparse_dict['nentity'],
            nrelation=argparse_dict['nrelation'],
            hidden_dim=argparse_dict['hidden_dim'],
            gamma=argparse_dict['gamma'],
            double_entity_embedding=argparse_dict['double_entity_embedding'],
            double_relation_embedding=argparse_dict['double_relation_embedding']
        )
        logging.info('Loading checkpoint %s...' % args.subsampling_model)
        checkpoint = torch.load(os.path.join(args.subsampling_model, 'checkpoint'))
        submodel.load_state_dict(checkpoint['model_state_dict'])
        submodel.eval()
        submodel = submodel.cuda()
        
        mbs_dataset = DataLoader(
            SubModelDataset(
                triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
                'head-batch'
            ), 
            batch_size=args.submodel_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=SubModelDataset.collate_fn
        )

        with torch.no_grad():
            count = {}
            for positive_sample, mode in mbs_dataset:
                if args.cuda:
                    positive_sample = positive_sample.cuda()

                scores = torch.exp(submodel(positive_sample)).squeeze(-1)
                
                for (head, relation, tail), score in zip(positive_sample, scores):
                    head, relation, tail, score = [e.item() for e in [head, relation, tail, score]]
                    count[(head, relation, tail)] = score
                    if (head, relation) in count:
                        count[(head, relation)] += score
                    else:
                        count[(head, relation)] = score
                    if (tail, -relation-1) in count:
                        count[(tail, -relation-1)] += score
                    else:
                        count[(tail, -relation-1)] = score
        
        return count
    
    def dump_freqs(self):
        file_cbs_dump = os.path.join(self.args.save_path, 'file_cbs_dump.tsv')
        print(file_cbs_dump)
        with open(file_cbs_dump, "w") as f_cnt_out:
            for key in self.freq_count.keys():
                if len(key) == 2:
                    f_cnt_out.write("\t".join([str(e) for e in [key[0],key[1],self.freq_count[key]]]) + "\n")
        if self.args.mbs_default or self.args.mbs_freq or self.args.mbs_uniq:
            file_mbs_dump = os.path.join(self.args.save_path, 'file_mbs_dump.tsv')
            print(file_mbs_dump)
            with open(file_mbs_dump, "w") as f_mbs_out:
                for key in self.mbs_count.keys():
                    if len(key) == 2:
                        f_mbs_out.write("\t".join([str(e) for e in [key[0],key[1],self.mbs_count[key]]]) + "\n")

    @staticmethod
    def ave_query_score_for_significance_test(triples, all_true_triples, args):
        '''
        To run the significance test, we need to consider the average scores for each query. 
        We need to run this for CBS model, MBS model, and MIX model, get their query scores and run significance test using "wilcoxon_score.py"
        '''
        # Restore model from checkpoint directory
        with open(os.path.join(args.save_path, 'config.json'), 'r') as fjson:
            argparse_dict = json.load(fjson)
        final_model = model.KGEModel(
            model_name=argparse_dict['model'],
            nentity=argparse_dict['nentity'],
            nrelation=argparse_dict['nrelation'],
            hidden_dim=argparse_dict['hidden_dim'],
            gamma=argparse_dict['gamma'],
            double_entity_embedding=argparse_dict['double_entity_embedding'],
            double_relation_embedding=argparse_dict['double_relation_embedding']
        )
        logging.info('Loading checkpoint %s...' % args.save_path)
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
        final_model.load_state_dict(checkpoint['model_state_dict'])
        final_model.eval()
        final_model = final_model.cuda()
        
        dataset = DataLoader(
            SubModelDataset(
                triples, 
                all_true_triples, 
                args.nentity, 
                args.nrelation, 
                'head-batch'
            ), 
            batch_size=args.batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=SubModelDataset.collate_fn
        )
    
        with torch.no_grad():
            count = defaultdict(list)  # 改用defaultdict来存储多个得分
            for positive_sample, mode in dataset:
                if args.cuda:
                    positive_sample = positive_sample.cuda()

                scores = torch.exp(final_model(positive_sample)).squeeze(-1)
                
                for (head, relation, tail), score in zip(positive_sample, scores):
                    head, relation, tail, score = [e.item() for e in [head, relation, tail, score]]
                    count[(head, relation, tail)].append(score)
                    count[(head, relation)].append(score)
                    count[(tail, -relation-1)].append(score)

            # 计算每个查询的平均得分
            for key in count:
                count[key] = sum(count[key]) / len(count[key])
            
        return count
    
    def dump_scores(self):
        file_dump_score = os.path.join(self.args.save_path, 'file_dump_score.tsv')
        print(file_dump_score)
        with open(file_dump_score, "w") as f_out:
            for key in self.score_count.keys():
                if len(key) == 2:
                    f_out.write("\t".join([str(e) for e in [key[0],key[1],self.score_count[key]]]) + "\n")
 
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail
    
class SubModelDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        mode = data[0][1]
        return positive_sample, mode

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
