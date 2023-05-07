import logging
import os
import time

import dgl
import pandas as pd
import torch
import torch.nn.functional as F
# from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import shutil
from helper.eval import *
from loss.UnsupervisedContrastiveLoss import UspConLoss
from loss.SupervisedContrastiveLoss import SupConLoss
from model.Locator import Locator
from model.TypeClassifier import TypeClassifier
from model.MainModel import MainModel
from model.backbone.TAGClassify import *
from loss.AutomaticWeightedLoss import AutomaticWeightedLoss
from helper.aug import *


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class TVDiag(object):

    def __init__(self, args, device, train_dl, test_dl):
        self.args = args
        self.device = device
        self.train_loader = train_dl
        self.test_loader = test_dl
        self.epochs = self.args['epochs']
        self.eval_period = self.args['eval_period']
        self.log_every_n_steps = self.args['log_every_n_steps']
        
        log_dir = os.path.join(self.args['log_dir'], self.args['dataset_name'])

        self.model = MainModel(args).to(device)

        # Liebel L, Körner M. Auxiliary tasks in multi-task learning[J]. arXiv preprint arXiv:1805.06334, 2018.
        self.awl = AutomaticWeightedLoss(3)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        
        t = self.args['temperature']
        self.supConLoss = SupConLoss(t)
        self.uspConLoss = UspConLoss(t)

        # alias tensorboard='python3 -m tensorboard.main'
        # tensorboard --logdir=logs
        self.writer = SummaryWriter(log_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        
        logging.info(self.model)

    def train(self):

        logging.info(f"Start training for {self.epochs} epochs.")
        logging.info(f"Training with gpu: {self.device}")
        logging.info(f"Status of task-oriented learning: {self.args['TO']}")
        logging.info(f"Status of cross-modal association: {self.args['CM']}")
        logging.info(f"Status of dynamic_weight: {self.args['dynamic_weight']}")
        if self.args['aug']:
            logging.info(f"Augmente the data: {self.args['aug_method']}")
        logging.info(f"lr: {self.args['lr']}, weight_decay: {self.args['weight_decay']}")
        
        n_test=0
        
        best_avg, best_f1 = 0, 0
        best_data = {}
        
        for epoch_counter in range(self.epochs):
            n_iter = 0
            start_time = time.time()
            self.model.train()
            epoch_loss = 0
            for batch_graphs, batch_labels in self.train_loader:
                instance_labels = batch_labels[:, 0].to(self.device)
                type_labels = batch_labels[:, 1].to(self.device)

                self.opt.zero_grad()
                
                # aug
                if self.args['aug']:
                    p = self.args['aug_percent']
                    graph_list = dgl.unbatch(batch_graphs)
                    if self.args['aug_method'] == 'node_drop':
                        aug_graph_list = aug_drop_node_list(graph_list, instance_labels, p)
                    elif self.args['aug_method'] == 'random_walk':
                        aug_graph_list = aug_random_walk_list(graph_list, instance_labels, p)
                    batch_graphs = dgl.batch(aug_graph_list + graph_list)
                    instance_labels = torch.cat((instance_labels, instance_labels), dim=0)
                    type_labels = torch.cat((type_labels, type_labels), dim=0)

                (metric_embs, trace_embs, log_embs), root_logit, type_logit = self.model(batch_graphs)

                # Task-oriented learning
                l1, l2 = 0, 0
                if self.args['TO']:
                    l1 = self.supConLoss(metric_embs, instance_labels) + \
                        self.supConLoss(trace_embs, instance_labels) + \
                                   self.supConLoss(log_embs, type_labels)

                # Cross-modal association
                if self.args['CM']:
                    l2 = self.uspConLoss(metric_embs, log_embs) + \
                        self.uspConLoss(metric_embs, trace_embs)
                
                guide_weight = self.args['guide_weight']
                contrastive_loss = guide_weight * (l1 + l2)
                
                RCA_loss = F.cross_entropy(root_logit, instance_labels)
                TC_loss = F.cross_entropy(type_logit, type_labels)

                if self.args['dynamic_weight']:
                    total_loss = self.awl(RCA_loss, TC_loss, contrastive_loss)
                else:
                    total_loss = contrastive_loss + RCA_loss + TC_loss

                logging.debug("con_loss: {:.3f}, RCA_loss: {:.3f}, TC_loss: {:.3f}"
                      .format(contrastive_loss, RCA_loss, TC_loss))

                total_loss.backward()

                self.opt.step()
                epoch_loss += total_loss.detach().item()

                n_iter += 1
                
            # early break
            mean_epoch_loss = epoch_loss / n_iter

            end_time = time.time()
            time_per_batch = (end_time - start_time)
            logging.info("Epoch {} done. Loss: {:.3f}, Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                         .format(epoch_counter, mean_epoch_loss, time_per_batch,
                                 self.train_loader.batch_size / time_per_batch))

            top1, top3, top5 = accuracy(root_logit, instance_labels, topk=(1, 3, 5))
            pre = precision(type_logit, type_labels, k=5)
            rec = recall(type_logit, type_labels, k=5)
            f1 = f1score(type_logit, type_labels, k=5)
            self.writer.add_scalar('loss', epoch_loss / n_iter, global_step=epoch_counter)
            self.writer.add_scalar('train/top1', top1, global_step=epoch_counter)
            self.writer.add_scalar('train/top3', top3, global_step=epoch_counter)
            self.writer.add_scalar('train/top5', top5, global_step=epoch_counter)
            self.writer.add_scalar('train/precision', pre, global_step=epoch_counter)
            self.writer.add_scalar('train/recall', rec, global_step=epoch_counter)
            self.writer.add_scalar('train/f1-score', f1, global_step=epoch_counter)

            # evaluate
            if epoch_counter % self.eval_period == 0:
                n_test += 1
                self.model.eval()
                root_pred = torch.FloatTensor([])
                type_pred = torch.FloatTensor([])
                root_truth = torch.FloatTensor([])
                type_truth = torch.FloatTensor([])
                for batch_graphs, batch_labels in self.test_loader:
                    with torch.no_grad():
                        batch_graphs = batch_graphs.to(self.device)

                        instance_labels = batch_labels[:, 0]
                        type_labels = batch_labels[:, 1]

                        (metric_embs, trace_embs, log_embs), root_logit, type_logit = self.model(batch_graphs)

                        root_pred = torch.cat((root_pred, root_logit.cpu()), dim=0)
                        type_pred = torch.cat((type_pred, type_logit.cpu()), dim=0)
                        root_truth = torch.cat((root_truth, instance_labels), dim=0)
                        type_truth = torch.cat((type_truth, type_labels), dim=0)

                top1, top2, top3, top4, top5 = accuracy(root_pred, root_truth, topk=(1, 2, 3, 4, 5))
                avg_5 = np.mean([top1, top2, top3, top4, top5])
                pre = precision(type_pred, type_truth, k=5)
                rec = recall(type_pred, type_truth, k=5)
                f1 = f1score(type_pred, type_truth, k=5)

                logging.info("Validation Results - Epoch: {}".format(epoch_counter))
                logging.info("[Root localization] top1: {:.3%}, top2: {:.3%}, top3: {:.3%}, top4: {:.3%}, top5: {:.3%}, avg@5: {:.3f}".format(top1, top2, top3, top4, top5, avg_5))
                logging.info("[Failure type classification] precision: {:.3%}, recall: {:.3%}, f1-score: {:.3%}".format(pre, rec, f1))

                self.writer.add_scalar('test/top1', top1, global_step=n_test)
                self.writer.add_scalar('test/top3', top3, global_step=n_test)
                self.writer.add_scalar('test/top5', top5, global_step=n_test)
                self.writer.add_scalar('test/precision', pre, global_step=n_test)
                self.writer.add_scalar('test/recall', rec, global_step=n_test)
                self.writer.add_scalar('test/f1-score', f1, global_step=n_test)
                
                if (avg_5 + f1) > (best_avg + best_f1):
                    best_avg = avg_5
                    best_f1 = f1
                    best_data['top1'] = top1
                    best_data['top2'] = top2
                    best_data['top3'] = top3
                    best_data['top4'] = top4
                    best_data['top5'] = top5
                    best_data['avg_5'] = avg_5
                    best_data['precision'] = pre
                    best_data['recall'] = rec
                    best_data['f1-score'] = f1
                    state = {
                        'epoch': self.epochs,
                        'model': self.model.state_dict(),
                        'opt': self.opt.state_dict(),
                    }
                    torch.save(state, os.path.join(self.writer.log_dir, 'model_best.pth.tar'))                    
        logging.info("Training has finished.")
        for key in best_data.keys():
            logging.info('{}: {}'.format(key, best_data[key]))
            print('{}: {}'.format(key, best_data[key]))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
