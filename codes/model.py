#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset
       
class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False, modulus_weight=1.0, phase_weight=0.5):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.pi = 3.14159262358979323846
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        if model_name == 'HAKE':
            self.entity_dim = 2*hidden_dim
            self.relation_dim = 3*hidden_dim
        else:
            self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
            self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
 
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        if model_name == 'HAKE':
            nn.init.ones_(
                tensor=self.relation_embedding[:, hidden_dim:2 * hidden_dim]
            )
            nn.init.zeros_(
                tensor=self.relation_embedding[:, 2 * hidden_dim:3 * hidden_dim]
            )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        if model_name == 'HAKE':
            self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
            self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'HAKE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'HAKE' : self.HAKE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-self.pi, self.pi]

        phase_relation = relation/(self.embedding_range.item()/self.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        
        #Make phases of entities and relations uniformly distributed in [-self.pi, self.pi]

        phase_head = head/(self.embedding_range.item()/self.pi)
        phase_relation = relation/(self.embedding_range.item()/self.pi)
        phase_tail = tail/(self.embedding_range.item()/self.pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    def HAKE(self, head, relation, tail, mode):
        
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(relation, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        if mode == 'head-batch':
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        score = self.gamma.item() - (phase_score + r_score)
        return score
    
    @staticmethod
    def train_step(model, optimizer, scaler, train_iterator, args):
        '''
        A single train step. Apply back-propagation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, hr_freq, tr_freq, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            hr_freq = hr_freq.cuda()
            tr_freq = tr_freq.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        positive_score = model(positive_sample)
            
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
        
        batch_size = positive_score.shape[0]
        
        if mode == 'head-batch':
            query_weight = tr_freq
        if mode == 'tail-batch':
            query_weight = hr_freq
            
        triple_weight = hr_freq + tr_freq
        
        triple_weight = model._norm_inv(triple_weight, 0.5)
        query_weight = model._norm_inv(query_weight, 0.5)
        
        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                    * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            if args.sum_ns_loss:
                negative_score = F.logsigmoid(-negative_score).sum(dim = 1)
            else:
                negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        if args.cnt_freq:

            positive_sample_loss = - (triple_weight * positive_score).sum()
            negative_sample_loss = - (query_weight * negative_score).sum()
        
        elif args.cnt_uniq:
        
            positive_sample_loss = - (query_weight * positive_score).sum()
            negative_sample_loss = - (query_weight * negative_score).sum()
        
        elif args.cnt_default:
        
            positive_sample_loss = - (triple_weight * positive_score).sum()
            negative_sample_loss = - (triple_weight * negative_score).sum()
        
        else:
        
            positive_sample_loss = - (positive_score).sum() / batch_size
            negative_sample_loss = - (negative_score).sum() / batch_size

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def train_step_ms(model, optimizer, scaler, train_iterator, args):
        '''
        A single train step. Apply back-propagation and return the loss
        Note: This method applies model-based subsampling
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, mbs_triple_freq, mbs_hr_freq, mbs_tr_freq, cnt_hr_freq, cnt_tr_freq, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            mbs_triple_freq = mbs_triple_freq.cuda()
            mbs_hr_freq = mbs_hr_freq.cuda()
            mbs_tr_freq = mbs_tr_freq.cuda()
            cnt_hr_freq = cnt_hr_freq.cuda()
            cnt_tr_freq = cnt_tr_freq.cuda()
        
        with torch.cuda.amp.autocast():

            negative_score = model((positive_sample, negative_sample), mode=mode)

            if args.negative_adversarial_sampling:
                #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                                * F.logsigmoid(-negative_score)).sum(dim = 1)
            else:
                negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

            positive_score = model(positive_sample)
            positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)
            
            # Cannot allow to combine different subsampling methods
            assert((args.mbs_freq^args.mbs_uniq)^args.mbs_default)
            
            if mode == 'head-batch':
                cnt_query_freq = cnt_tr_freq
                mbs_query_freq = mbs_tr_freq
            if mode == 'tail-batch':
                cnt_query_freq = cnt_hr_freq
                mbs_query_freq = mbs_hr_freq
            cnt_triple_freq = cnt_hr_freq + cnt_tr_freq

            mbs_triple_weight = model._norm_inv(mbs_triple_freq, args.subsampling_model_temperature)
            mbs_query_weight = model._norm_inv(mbs_query_freq, args.subsampling_model_temperature)
                
            if args.mbs_ratio < 1.0:
                cnt_query_weight = model._norm_inv(cnt_query_freq, 0.5)
                cnt_triple_weight = model._norm_inv(cnt_triple_freq, 0.5)
                mixed_triple_weight = model._norm_wsum(mbs_triple_weight, cnt_triple_weight, args.mbs_ratio).squeeze(-1)
                mixed_query_weight = model._norm_wsum(mbs_query_weight, cnt_query_weight, args.mbs_ratio).squeeze(-1)
            else:
                mixed_triple_weight = mbs_triple_weight.squeeze(-1)
                mixed_query_weight = mbs_query_weight.squeeze(-1)
            
            if args.mbs_freq:
                
                positive_sample_loss = - (mixed_triple_weight * positive_score).sum()
                negative_sample_loss = - (mixed_query_weight * negative_score).sum()
            
            elif args.mbs_uniq:
                
                positive_sample_loss = - (mixed_query_weight * positive_score).sum()
                negative_sample_loss = - (mixed_query_weight * negative_score).sum()
            
            elif args.mbs_default:
           
                positive_sample_loss = - (mixed_triple_weight * positive_score).sum()
                negative_sample_loss = - (mixed_triple_weight * negative_score).sum()
            
            else:
                assert(False)

            loss = (positive_sample_loss + negative_sample_loss)/2
        
            if args.regularization != 0.0:
                #Use L3 regularization for ComplEx and DistMult
                regularization = args.regularization * (
                    model.entity_embedding.norm(p = 3)**3 + 
                    model.relation_embedding.norm(p = 3).norm(p = 3)**3
                )
                loss = loss + regularization
                regularization_log = {'regularization': regularization.item()}
            else:
                regularization_log = {}
            
        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def _norm_inv(w, temp):
        w = torch.pow(1 / w, temp)
        w = w / w.sum()
        return w

    @staticmethod
    def _norm_wsum(a, b, r):
        assert(a.shape == b.shape)
        assert(r >= 0.0 and 1.0 >= r)
        return r * a + (1.0 - r) * b

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
    
