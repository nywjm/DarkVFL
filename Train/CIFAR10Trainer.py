import copy
from random import sample
import torch
torch.autograd.set_detect_anomaly(True)  
from dataset.utils import split_vfl
from attack.attack import attack_LFBA, get_near_index, calculate_update_magnitude, calculate_update_distribution, \
    align_benign_consistency, attack_rsa
from utils.utils import *
import time
import numpy as np
import math
class Trainer:
    def __init__(self, device, model_list, extractor_list, extractor, optimizer_list, criterion, train_loader,
                 test_loader,
                 test_asr_loader, trigger_dimensions,
                 logger, args=None, checkpoint=None):
        self.device = device
        self.model_list = model_list
        self.extractor_list = extractor_list
        self.extractor = extractor
        self.optimizer_list = optimizer_list
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_asr_loader = test_asr_loader
        self.logger = logger
        self.args = args
        self.checkpoint = checkpoint
        self.trigger_dimensions = trigger_dimensions

    def adjust_learning_rate(self, epoch):
        lr = self.args.lr * (0.1) ** (epoch // 20)
        for opt in self.optimizer_list:
            for param_group in opt.param_groups:
                param_group['lr'] = lr
    def train(self):
        start_time_train = time.time()
        if self.args.attack:
            self.logger.info("=> Start Training with {}...".format(self.args.attack))
            if self.args.pretrain_stage:
                self.logger.info("=> Pretrain...")
        else:
            self.logger.info("=> Start Training Baseline...")
        epoch_loss_list = []
        model_list = self.model_list
        model_list = [model.train() for model in model_list]
        best_acc = 0
        best_trade_off = 0
        best_epoch = 0
        asr_for_best_epoch = 0
        target_for_best_epoch = 0
        no_change = 0
        total_time_GPC = 0
        total_time_HS = 0
        self.select_his = torch.zeros(self.train_loader.dataset.data.shape[0])
        self.historical_grad_abs = None
        self.frozen_cold_indices = None  
        if self.checkpoint:
            best_acc = self.checkpoint['best_acc']
        # train and update
        for ep in range(self.args.start_epoch, self.args.epoch):
            batch_loss_list = []
            total = 0
            correct = 0
            is_attack_active = (self.args.attack == 'LFBA' and (
                        self.args.attacker_stop_epoch is None or ep < self.args.attacker_stop_epoch))
            if ep >= 1 and self.args.attack == 'LFBA' and is_attack_active:
                self.train_features, self.train_labels, self.train_indexes = self.grad_vec_epoch, self.target_epoch, self.indexes_epoch
                self.train_features, self.train_labels, self.train_indexes = self.train_features.cpu(), self.train_labels.cpu(), self.train_indexes.cpu()
                self.num_poisons = int(self.args.poison_rate * len(self.train_loader.dataset.data))
                self.num_select = int(self.num_poisons * self.args.select_rate)
                if ep == 1:
                    start_time = time.time()
                    self.anchor_idx_t = torch.nonzero(self.train_indexes == self.args.anchor_idx).squeeze()
                    anchor_label_val = self.train_labels[self.anchor_idx_t]
                    if anchor_label_val.dim() > 0:
                        anchor_label_val = anchor_label_val[0]  
                    same_class_mask = (self.train_labels == anchor_label_val)
                    same_class_indices = torch.nonzero(same_class_mask).squeeze()
                    if same_class_indices.dim() == 0:
                        same_class_indices = same_class_indices.unsqueeze(0)
                    if len(same_class_indices) > 0:
                        multi_anchor_feature = torch.mean(self.train_features[same_class_indices], dim=0)
                        self.indexes = get_near_index(multi_anchor_feature, self.train_features, self.num_poisons)
                    else:
                        self.indexes = get_near_index(self.train_features[self.anchor_idx_t], self.train_features,
                                                      self.num_poisons)
                    end_time = time.time()
                    print("The poison set construction time: {}".format((end_time - start_time)))
                    total_time_GPC += (end_time - start_time)
                    self.poison_indexes = self.train_indexes[self.indexes]
                    self.consistent_rate = float(
                        (self.train_labels[self.indexes] == int(self.train_labels[self.anchor_idx_t])).sum() / len(
                            self.indexes))

                # For replace poisoning
                self.indexes = np.isin(self.train_indexes.numpy(), torch.tensor(self.poison_indexes).numpy())
                temp = np.array(range(len(self.train_indexes)))
                self.indexes = temp[self.indexes]
                self.l2_norm_features = torch.norm(self.train_features[self.indexes], p=2, dim=1)
                start_time = time.time()
                self.poison_features, self.select_indexes = self.l2_norm_features.topk(self.num_select, dim=0,
                                                                                       largest=True,
                                                                                       sorted=True)
                end_time = time.time()
                print("The hard-sample selection time: {}".format((end_time - start_time)))
                total_time_HS += (end_time - start_time)
                num_of_replace = int(len(self.poison_indexes) * self.args.select_rate)
                replace_all_list = list(
                    set(self.train_indexes.numpy()).difference(set(torch.tensor(self.poison_indexes).numpy())))
                replace_indexes_others = sample(replace_all_list, num_of_replace)
                random_indexes_target = sample(list(self.poison_indexes), num_of_replace)
                selected_indexes_target = self.train_indexes[self.indexes[self.select_indexes]]

                if self.args.poison_all:
                    if self.args.random_select:
                        self.poison_indexes_t = sample(list(self.poison_indexes), self.num_select)
                        self.indexes = np.isin(self.train_indexes.numpy(), torch.tensor(self.poison_indexes_t).numpy())
                    self.poisoning_labels = np.array(self.train_labels)[self.indexes]
                    self.anchor_label = int(self.train_labels[self.train_indexes == self.args.anchor_idx])
                    self.args.target_label = self.anchor_label
                    self.logger.info('Target label:{}'.format(self.anchor_label))
                    self.clean_data_p = copy.deepcopy(self.train_loader.dataset.data_p)
                    if self.args.random_select:
                        self.train_loader.dataset.data = attack_LFBA(self.args, self.logger, [],
                                                                     [], self.train_indexes,
                                                                     self.poison_indexes_t,
                                                                     self.clean_data_p,
                                                                     self.train_loader.dataset.targets,
                                                                     self.trigger_dimensions,
                                                                     self.args.poison_rate, 'train')
                    else:
                        self.train_loader.dataset.data = attack_LFBA(self.args, self.logger, [],
                                                                     [], self.train_indexes,
                                                                     self.poison_indexes,
                                                                     self.clean_data_p,
                                                                     self.train_loader.dataset.targets,
                                                                     self.trigger_dimensions,
                                                                     self.args.poison_rate, 'train')
                else:
                    if self.args.random_select:
                        replace_indexes_target = random_indexes_target
                    else:
                        replace_indexes_target = selected_indexes_target
                    self.poisoning_labels = np.array(self.train_labels)[self.indexes]
                    self.anchor_label = int(self.train_labels[self.train_indexes == self.args.anchor_idx])
                    self.clean_data_p = copy.deepcopy(self.train_loader.dataset.data_p)
                    self.train_loader.dataset.data = attack_LFBA(self.args, self.logger, replace_indexes_others,
                                                                 replace_indexes_target, self.train_indexes,
                                                                 self.poison_indexes,
                                                                 self.clean_data_p,
                                                                 self.train_loader.dataset.targets,
                                                                 self.trigger_dimensions,
                                                                 self.args.poison_rate, 'train')
                    self.args.target_label = self.anchor_label
                    self.logger.info('Target label:{}'.format(self.anchor_label))
            elif self.args.attacker_stop_epoch is not None and ep >= self.args.attacker_stop_epoch:
                if hasattr(self.train_loader.dataset, 'data_p'):
                    self.train_loader.dataset.data = copy.deepcopy(self.train_loader.dataset.data_p)
                    if hasattr(self.train_loader.dataset, 'targets_p'):
                        self.train_loader.dataset.targets = copy.deepcopy(self.train_loader.dataset.targets_p)
                if ep == self.args.attacker_stop_epoch:
                    self.logger.info(
                        f"Epoch {ep + 1}: attack stop,attacker_stop_epoch:{self.args.attacker_stop_epoch}。")
                if self.args.purify_rate > 0:
                   self.train_loader.dataset.data = attack_rsa(
                        self.args, self.logger,
                        self.train_loader.dataset.data,
                        self.trigger_dimensions,
                        self.args.purify_rate, 'train'
                    )
            elif self.args.attack == 'rsa' or self.args.attack == 'lra' or self.args.attack is None:
                pass

            self.logger.info("=> Start Training for Injecting Backdoor...")

            self.grad_vec_epoch = []
            self.indexes_epoch = []
            self.target_epoch = []
            for step, (x_n, x_p, y, index) in enumerate(self.train_loader):
                x = x_n
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                x_split_list = split_vfl(x, self.args)
                local_output_list = []
                global_input_list = []
                for i in range(self.args.client_num):
                    local_output = model_list[i + 1](x_split_list[i])
                    local_output.retain_grad()

                    def make_hook(c_idx):
                        def hook(grad):
                            modified_grad = grad.clone()

                            if self.args.attack == 'LFBA' and is_attack_active and c_idx == self.args.attack_client_num:
                                batch_mean_abs = torch.mean(torch.abs(modified_grad.detach()), dim=0)

                                if self.historical_grad_abs is None:
                                    self.historical_grad_abs = batch_mean_abs
                                else:
                                    self.historical_grad_abs = 0.7 * self.historical_grad_abs + 0.3 * batch_mean_abs

                                k_percent = 0.30
                                k_num = max(1, int(self.historical_grad_abs.shape[0] * k_percent))
                                _, cold_indices = torch.topk(self.historical_grad_abs, k_num, largest=False)

                                iba_mask = torch.ones_like(modified_grad) * 0.9
                                iba_mask[:, cold_indices] = 2.0

                                modified_grad = modified_grad * iba_mask

                            return modified_grad

                        return hook

                    local_output.register_hook(make_hook(i))
                    local_output_list.append(local_output)
                global_output = model_list[0](local_output_list)
                loss = self.criterion(global_output, y)
                #
                # #Stealth Loss
                # loss_main = self.criterion(global_output, y)
                # 
                # loss_stealth = 0.0
                # if self.args.attack == 'LFBA':
                #     attack_idx = self.args.attack_client_num
                #     benign_idx = 1 - attack_idx  
                #
                #     mal_embed = local_output_list[attack_idx]
                #     
                #     benign_embed = local_output_list[benign_idx].detach()
                #
                #     mal_norm = torch.norm(mal_embed, p=2, dim=1)
                #     benign_norm = torch.norm(benign_embed, p=2, dim=1)
                #
                #     loss_stealth = 0.005 * torch.nn.functional.smooth_l1_loss(mal_norm, benign_norm)
                #
                # loss = loss_main + loss_stealth
                for opt in self.optimizer_list:
                    opt.zero_grad()

                loss.backward()
                
                # if self.args.attack == 'LFBA':
                #     #defense
                #     for client_idx in range(self.args.client_num):
                #         client_grad = local_output_list[client_idx].grad
                #         if client_grad is None:
                #             self.logger.warning(f"client{client_idx}'s gradient is None")
                #             continue
                #         # 1：gradient_compression
                #         # client_grad = grad_compression(client_grad, compress_rate=self.args.compress_rate)
                #         # 2：gaussian_noise_to_gradient
                #         # client_grad = add_gaussian_noise_to_grad(client_grad, noise_std=self.args.noise_std)
                #         
                #         local_output_list[client_idx].grad = client_grad

                if self.args.attack == 'LFBA':
                    attack_client_idx = self.args.attack_client_num
                    grad = local_output_list[attack_client_idx].grad
                    if grad is not None:
                        self.grad_vec_epoch.append(grad.to(self.device))
                    else:
                        self.logger.warning(
                            f"Epoch {ep + 1}, Step {step + 1}: client{attack_client_idx}'s gradient is None")
                        
                        zero_grad = torch.zeros_like(local_output_list[attack_client_idx], device=self.device)
                        self.grad_vec_epoch.append(zero_grad)
                    self.indexes_epoch.append(index)
                    self.target_epoch.append(y)

                for opt in self.optimizer_list:
                    opt.step()
                
                batch_loss_list.append(loss.item())

                # calculate the training accuracy
                _, predicted = global_output.max(1)
                
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                # train_acc
                train_acc = correct / total
                current_loss = sum(batch_loss_list) / len(batch_loss_list)

                if step % self.args.print_steps == 0:
                    self.logger.info(
                        'Epoch: {}, {}/{}: train loss: {:.4f}, train main task accuracy: {:.4f}'.format(ep + 1,
                                                                                                        step + 1,
                                                                                                        len(self.train_loader),
                                                                                                        current_loss,
                                                                                                        train_acc))
            if self.args.attack == 'LFBA':
                self.grad_vec_epoch = torch.cat(self.grad_vec_epoch)
                self.indexes_epoch = torch.cat(self.indexes_epoch)
                self.target_epoch = torch.cat(self.target_epoch)

            epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
            epoch_loss_list.append(epoch_loss)
            self.adjust_learning_rate(ep + 1)
            test_acc, test_poison_accuracy, test_target, test_asr = self.test(ep)
            test_trade_off = (test_acc + test_asr) / 2
            if test_trade_off > best_trade_off:
                # best accuracy
                best_acc = test_acc
                best_trade_off = test_trade_off
                poison_acc_for_best_epoch = test_poison_accuracy
                asr_for_best_epoch = test_asr
                target_for_best_epoch = test_target
                no_change = 0
                best_epoch = ep
                # save model
                self.logger.info("=> Save best model...")
                state = {
                    'epoch': ep + 1,
                    'best_acc': best_acc,
                    'test_trade_off': test_trade_off,
                    'test_target': target_for_best_epoch,
                    'poison_acc': poison_acc_for_best_epoch,
                    'asr': asr_for_best_epoch,
                    'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
                    'optimizer': [self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))],
                }
                filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar'.format(ep + 1))
                torch.save(state, filename)
            else:
                if ep > self.args.pretrain_stage:
                    no_change += 1
            self.logger.info(
                '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, best trade off accuracy: {:.4f}, main task accuracy: {:.4f}, test target accuracy: {:.4f}, test asr: {:.4f}'.format(
                    ep + 1,
                    no_change,
                    best_epoch + 1, best_trade_off, best_acc, target_for_best_epoch, asr_for_best_epoch))
            if no_change == self.args.early_stop:
                end_time_train = time.time()
                print("The total training time: {}".format((end_time_train - start_time_train)))
                print("The average training time of each epoch: {}".format(
                    ((end_time_train - start_time_train)) / (ep + 1)))
                print("The poison set construction time: {}".format(total_time_GPC))
                print("The average hard-sample selection time: {}".format(total_time_HS / (ep + 1)))
                print("The total hard-sample selection time: {}".format(total_time_HS))
                return

    def test(self, ep):
        self.logger.info("=> Test ASR...")
        model_list = self.model_list
        model_list = [model.eval() for model in model_list]
        # test main task accuracy
        batch_loss_list = []
        total = 0
        correct = 0
        total_target = 0
        correct_target = 0
        for step, (x, x_p, y, index) in enumerate(self.test_loader):
            x = x.to(self.device).float()
            y = y.to(self.device).long()
            # split data for vfl
            x_split_list = split_vfl(x, self.args)
            local_output_list = []
            global_input_list = []
            # get the local model outputs
            for i in range(self.args.client_num):
                local_output_list.append(model_list[i + 1](x_split_list[i]))
            # get the global model inputs, recording the gradients
            for i in range(self.args.client_num):
                global_input_t = local_output_list[i].clone()
                global_input_t.requires_grad_(True)
                global_input_list.append(global_input_t)

            # global_output = model_list[0](local_output_list)
            global_output = model_list[0](global_input_list)
            # global model backward
            loss = self.criterion(global_output, y)
            batch_loss_list.append(loss.item())

            # calculate the testing accuracy
            _, predicted = global_output.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            total_target += (y == self.args.target_label).float().sum()
            correct_target += predicted.eq(y)[y == self.args.target_label].float().sum().item()

        # test poison accuracy and asr
        total_poison = 0
        correct_poison = 0
        total_asr = 0
        correct_asr = 0
        for step, (x, x_p, y, index) in enumerate(self.test_asr_loader):
            x = x.to(self.device).float()
            y = y.to(self.device).long()
            y_attack_target = torch.ones(size=y.shape).to(self.device).long()
            y_attack_target *= self.args.target_label
            # split data for vfl
            x_split_list = split_vfl(x, self.args)
            local_output_list = []
            global_input_list = []
            # get the local model outputs
            for i in range(self.args.client_num):
                local_output_list.append(model_list[i + 1](x_split_list[i]))
            # get the global model inputs, recording the gradients
            for i in range(self.args.client_num):
                global_input_t = local_output_list[i].clone()
                global_input_t.requires_grad_(True)
                global_input_list.append(global_input_t)

            # global_output = model_list[0](local_output_list)
            global_output = model_list[0](global_input_list)
            # calculate the poison accuracy
            _, predicted = global_output.max(1)
            total_poison += y.size(0)
            correct_poison += predicted.eq(y).sum().item()
            # calculate the ASR
            
            total_asr += (y != self.args.target_label).float().sum()
            correct_asr += (predicted[y != self.args.target_label] == self.args.target_label).float().sum()

        # main task accuracy, poison_acc and asr
        test_acc = correct / total
        test_poison_accuracy = correct_poison / total_poison
        test_asr = correct_asr / total_asr
        test_target = correct_target / total_target
        epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        test_trade_off = (test_acc + test_asr) / 2
        # main task accuracy on target set
        self.logger.info(
            '=> Test Epoch: {}, main task samples: {}, attack samples: {}, test loss: {:.4f}, test trade off: {:.4f}, test main task '
            'accuracy: {:.4f}, test target accuracy: {:.4f}, test asr: {:.4f}'.format(ep + 1,
                                                                                      len(self.test_loader.dataset),
                                                                                      len(self.test_asr_loader.dataset),
                                                                                      epoch_loss,
                                                                                      test_trade_off, test_acc,
                                                                                      test_target, test_asr))

        return test_acc, test_poison_accuracy, test_target, test_asr
    

    def detect_anomalous_client(self, client_grads, benign_stats=None, threshold=3.0):
        
        client_features = []
        
        for grad in client_grads:
            if grad is None:
                client_features.append(None)
                continue
            mag = calculate_update_magnitude(grad)
            dist = calculate_update_distribution(grad)
            client_features.append({**mag, **dist})

        if benign_stats is None:
            valid_features = [f for f in client_features if f is not None]
            if len(valid_features) == 0:
                return []
            
            feature_keys = valid_features[0].keys()
            baseline = {}
            for k in feature_keys:
                vals = [f[k] for f in valid_features]
                baseline[k] = {"mean": np.mean(vals), "std": np.std(vals)}
        else:
            baseline = {
                k: {"mean": benign_stats[k], "std": np.std([benign_stats[k]])}
                for k in benign_stats.keys()
            }

        anomalous_clients = []
        for idx, feat in enumerate(client_features):
            if feat is None:
                continue
            is_anomalous = False
            for k in feature_keys:
                z_score = abs(feat[k] - baseline[k]["mean"]) / (baseline[k]["std"] + 1e-8)
                if z_score > threshold:
                    is_anomalous = True
                    break
            if is_anomalous:
                anomalous_clients.append(idx)

        return anomalous_clients

    def robust_aggregate(self, client_params, agg_type="median", clip_ratio=0.1):
        
        aggregated_params = []
        
        for param_layer in zip(*client_params):
            # param_layer: (client0_param, client1_param, ...)
            param_tensor = torch.stack(param_layer)  # [client_num, *param_shape]

            if agg_type == "median":
                
                agg_param = torch.median(param_tensor, dim=0)[0]
            elif agg_type == "trimmed_mean":
                
                clip_num = int(len(param_layer) * clip_ratio)
                if clip_num > 0:
                    
                    param_norms = torch.norm(param_tensor.reshape(len(param_layer), -1), p=2, dim=1)
                    sorted_idx = torch.argsort(param_norms)
                    
                    keep_idx = sorted_idx[clip_num:-clip_num]
                    param_tensor = param_tensor[keep_idx]
                agg_param = torch.mean(param_tensor, dim=0)
            else:
                
                agg_param = torch.mean(param_tensor, dim=0)

            aggregated_params.append(agg_param)
        return aggregated_params

def grad_compression(grad, compress_rate=0.6):
    if grad is None:
        return grad
    
    clip_num = math.floor(compress_rate * grad.shape[-1])  
    
    for i in range(grad.shape[0]):
        
        _, pos = torch.topk(torch.abs(grad[i].clone().detach()), k=clip_num, largest=True, sorted=False)
        
        grad_i = torch.zeros_like(grad[i])
        grad_i[pos] = grad[i][pos]
        grad[i] = grad_i
    return grad

def add_gaussian_noise_to_grad(grad, noise_std=0.01):
    
    if grad is None:
        return grad
    
    gaussian_noise = np.random.normal(0, noise_std, grad.shape)
    gaussian_noise = torch.tensor(gaussian_noise).float().to(grad.device)
    
    grad_noise = grad + gaussian_noise
    return grad_noise