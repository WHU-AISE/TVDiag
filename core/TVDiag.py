import os
import time

import dgl
import torch
import torch.nn.functional as F
# from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import shutil
from core.ita import cal_task_affinity
from core.loss.UnsupervisedContrastiveLoss import UspConLoss
from core.loss.SupervisedContrastiveLoss import SupConLoss
from core.loss.AutomaticWeightedLoss import AutomaticWeightedLoss
from core.model.MainModel import MainModel

from helper.eval import *
from helper.early_stop import EarlyStopping

class TVDiag(object):

    def __init__(self, args, logger, device):
        self.args = args
        self.device = device
        self.logger = logger

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_period = args.eval_period
        self.log_every_n_steps = args.log_step
        self.tau = args.temperature
        log_dir = f"logs/{args.dataset}"
        os.makedirs(log_dir, exist_ok=True)

        self.patience = args.patience

        # alias tensorboard='python3 -m tensorboard.main'
        # tensorboard --logdir=logs
        self.writer = SummaryWriter(log_dir)
        self.printParams()

    def printParams(self):
        self.logger.info(f"Training with: {self.device}")
        self.logger.info(f"batch size: {self.args.batch_size}")
        self.logger.info(f"Status of task-oriented learning: {self.args.TO}")
        self.logger.info(f"Status of cross-modal association: {self.args.CM}")
        self.logger.info(f"guide weight: {self.args.guide_weight}")
        self.logger.info(f"temperature: {self.args.temperature}")
        self.logger.info(f"Status of dynamic_weight: {self.args.dynamic_weight}")
        self.logger.info(f"aug percent: {self.args.aug_percent}")
        self.logger.info(f"lr: {self.args.lr}, weight_decay: {self.args.weight_decay}")

    def train(self, train_dl):

        model = MainModel(self.args).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        awl = AutomaticWeightedLoss(3)
        supConLoss = SupConLoss(self.tau, self.device).to(self.device)
        uspConLoss = UspConLoss(self.tau, self.device).to(self.device)

        self.logger.info(model)
        self.logger.info(f"Start training for {self.epochs} epochs.")
        
        # Overhead
        train_times = []

        # Inter-Task Affinity (RCL -> FTI, FTI -> RCL)
        Z_r2fs, Z_f2rs = [], []
        
        # early stop configuration
        # RCLEarlyStop = EarlyStopping(patience=self.patience, min_delta=1e-4)
        # FTIEarlyStop = EarlyStopping(patience=self.patience, min_delta=1e-4)
        earlyStop = EarlyStopping(patience=self.patience)

        for epoch in range(self.epochs):
            n_iter = 0
            start_time = time.time()
            model.train()
            epoch_loss, epoch_con_l, epoch_rcl_l, epoch_fti_l = 0, 0, 0, 0

            for batch_graphs, batch_labels in train_dl:
                batch_graphs = batch_graphs.to(self.device)
                instance_labels = batch_labels[:, 0].to(self.device)
                type_labels = batch_labels[:, 1].to(self.device)

                opt.zero_grad()

                (f_m, f_t, f_l), root_logit, type_logit = model(batch_graphs)

                # Task-oriented learning
                l_to, l_cm = torch.tensor(0), torch.tensor(0)
                if self.args.TO:
                    l_to = supConLoss(f_m, instance_labels) + \
                        supConLoss(f_t, instance_labels) + \
                                   supConLoss(f_l, type_labels)

                # Cross-modal association
                if self.args.CM:
                    l_cm = uspConLoss(f_m, f_t) + \
                        uspConLoss(f_m, f_l)
                
                # Failure Diagnosis
                gamma = self.args.guide_weight
                l_con = gamma * (l_to + l_cm)

                l_rcl = F.cross_entropy(root_logit, instance_labels)
                l_fti = F.cross_entropy(type_logit, type_labels)
                if self.args.dynamic_weight:
                    total_loss = awl(l_rcl, l_fti, l_con)
                else:
                    total_loss = l_con + l_rcl + l_fti

                self.logger.debug("con_loss: {:.3f}, RCA_loss: {:.3f}, TC_loss: {:.3f}"
                      .format(l_con, l_rcl, l_fti))

                
                # Calculate Inter-Task Affinity
                if epoch == 0:
                    Z_r2f, Z_f2r = cal_task_affinity(model=model, 
                                    optimizer=opt, 
                                    batch_graphs=batch_graphs,
                                    instance_labels=instance_labels,
                                    type_labels=type_labels)
                    Z_r2fs.append(Z_r2f)
                    Z_f2rs.append(Z_f2r)

                total_loss.backward()
                opt.step()
                epoch_loss += total_loss.detach().item()
                epoch_con_l += l_con.detach().item()
                epoch_rcl_l += l_rcl.detach().item()
                epoch_fti_l += l_fti.detach().item()
                epoch_loss += total_loss.detach().item()
                n_iter += 1
                
            mean_epoch_loss = epoch_loss / n_iter
            mean_con_loss = epoch_con_l / n_iter
            mean_rcl_loss = epoch_rcl_l / n_iter
            mean_fti_loss = epoch_fti_l / n_iter
            end_time = time.time()
            time_per_epoch = (end_time - start_time)
            train_times.append(time_per_epoch)
            self.logger.info("Epoch {} done. Loss: {:.3f}, Time per epoch: {:.3f}[s]"
                         .format(epoch, mean_epoch_loss, time_per_epoch))

            top1, top3, top5 = HR(root_logit, instance_labels, topk=(1, 3, 5))
            NDCG_3 = NDCG(root_logit.cpu(), instance_labels.cpu(), topk=(3,))[0]
            pre = precision(type_logit, type_labels, k=5)
            rec = recall(type_logit, type_labels, k=5)
            f1 = f1score(type_logit, type_labels, k=5)
            self.writer.add_scalar('loss/mean total loss', mean_epoch_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean con loss', mean_con_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean RCL loss', mean_rcl_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean FTI loss', mean_fti_loss, global_step=epoch)
            self.writer.add_scalar('train/top1', top1, global_step=epoch)
            self.writer.add_scalar('train/top3', top3, global_step=epoch)
            self.writer.add_scalar('train/top5', top5, global_step=epoch)
            self.writer.add_scalar('train/NDCG_3', NDCG_3, global_step=epoch)
            self.writer.add_scalar('train/precision', pre, global_step=epoch)
            self.writer.add_scalar('train/recall', rec, global_step=epoch)
            self.writer.add_scalar('train/f1-score', f1, global_step=epoch)

            # early break
            # rcl_stop = RCLEarlyStop.should_stop(mean_rcl_loss, epoch)
            # fti_stop = FTIEarlyStop.should_stop(mean_fti_loss, epoch)
            stop = earlyStop.should_stop(mean_epoch_loss, epoch)
            if stop:
                self.logger.info(f"Early stop at epoch {epoch} due to lack of improvement.")
                break

            
        state = {
            'epoch': self.epochs,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
        }
        torch.save(state, os.path.join(self.writer.log_dir, 'TVDiag.pt'))

                 
        self.logger.info("Training has finished.")
        # calculate the training time for raw data
        self.logger.debug(f"The training time is {np.sum(train_times)}[s]")
        self.logger.debug(f"The training time per epoch is {np.mean(train_times)}[s]")
        self.logger.debug(f"The affinity of RCL -> FTI is {np.mean(Z_r2f)}")
        self.logger.debug(f"The affinity of FTI -> RCL is {np.mean(Z_f2r)}")
        self.logger.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")


    def evaluate(self, test_data):
        checkpoint = torch.load(os.path.join(self.writer.log_dir, 'TVDiag.pt'))
        model = MainModel(self.args).to(self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        root_logits, type_logits = [], []
        roots, types = [], []
        inference_times = []
        for data in test_data:
            graph = data[0].to(self.device)
            root = data[1][0]
            failure_type = data[1][1]
            roots.append(root)
            types.append(failure_type)
        
            start_time = time.time()
            with torch.no_grad():
                _, root_logit, type_logit = model(graph)
                root_logits.append(root_logit.flatten())
                type_logits.append(type_logit.flatten())
            end_time = time.time()
            inference_times.append(end_time - start_time)
        root_logits = torch.vstack(root_logits).cpu()
        type_logits = torch.vstack(type_logits).cpu()
        roots = torch.tensor(roots)
        types = torch.tensor(types)

        top1, top2, top3, top4, top5 = HR(root_logits, roots, topk=(1, 2, 3, 4, 5))
        NDCG_3 = NDCG(root_logits, roots, topk=(3,))[0]
        avg_5 = np.mean([top1, top2, top3, top4, top5])
        avg_3 = np.mean([top1, top2, top3])
        pre = precision(type_logits, types, k=5)
        rec = recall(type_logits, types, k=5)
        f1 = f1score(type_logits, types, k=5)

        self.logger.info("[Root localization] top1: {:.3%}, top2: {:.3%}, top3: {:.3%}, top4: {:.3%}, top5: {:.3%},avg@3: {:.3f}, avg@5: {:.3f}, NDCG@3: {:.3f}".format(top1, top2, top3, top4, top5,avg_3, avg_5, NDCG_3))
        self.logger.info("[Failure type classification] precision: {:.3%}, recall: {:.3%}, f1-score: {:.3%}".format(pre, rec, f1))
        self.logger.info(f"The average test time is {np.mean(inference_times)}[s]")
        self.logger.info(f"The total test time is {np.sum(inference_times)}[s]")