import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed, pil_loader
import copy
import os, math
import pickle
from PIL import Image
from itertools import chain
import shutil
from torchvision import transforms
import ipdb
import copy
import logging
import datetime
import random, time

class IndexedDataset(Dataset):
    def __init__(self, dataset, indices, transform):
        
        self.images = dataset.images[indices]
        self.labels = dataset.labels[indices]
        self.transform = transform

    def __getitem__(self, idx):

        image = self.transform(Image.fromarray(self.images[idx]))  # 通过索引映射到原始数据集的索引
        label = self.labels[idx] 
        return idx, image, label

    def __len__(self):

        return len(self.images)

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)

def label_distribution(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets, transform, args, logger):
        if args['dataset'] == 'tiny_imagenet':
            self.use_path = True
        else:
            self.use_path = False

        datasets = [*datasets[0], *datasets[1]] 

        self.labels = np.concatenate([ds.labels for ds in datasets]) 
        self.images = np.concatenate([ds.images for ds in datasets]) 
        self.transform = transform
        
        LD = label_distribution(self.labels)
        logger.info("Label Distribution: %s" % str(LD))
        print(LD)
    def __getitem__(self, idx):

        if self.use_path:
            image = self.transform(pil_loader(self.images[idx]))
        else:
            image = self.transform(Image.fromarray(self.images[idx]))  # 通过索引映射到原始数据集的索引
        label = self.labels[idx] 
        return idx, image, label
    
    def __len__(self):
        return len(self.labels)
    


@torch.no_grad()
def lewis_weights(A: torch.Tensor, p: float = 2.0, 
                  max_iter: int = 50, tol: float = 1e-6,
                  ridge: float = 1e-10, dtype=torch.float64,
                  normalize: str = None):

    A = A.to(dtype)
    n, d = A.shape


    # lev_i = a_i^T (A^T A)^(-1) a_i
    ATA = A.T @ A
    I = torch.eye(d, dtype=dtype, device=A.device)
    ATA_inv = torch.linalg.pinv(ATA + ridge * I)
    q = (A @ ATA_inv * A).sum(dim=1)  # (n,)
    w = torch.clamp(q, min=1e-16).clone()

    if p == 2.0:

        if normalize == 'sum_to_rank':
            r = torch.linalg.matrix_rank(A)
            s = w.sum()
            if s > 0:
                w = w * (r / s)
        return w


    exp_val = 1.0 - 2.0 / p  # W^{1 - 2/p}
    for _ in range(max_iter):

        s = torch.clamp(w, min=1e-32) ** exp_val        # (n,)
        # A^T diag(s) A  = A^T (s[:,None] * A)
        ATWA = A.T @ (s.unsqueeze(1) * A)
        ATWA_inv = torch.linalg.pinv(ATWA + ridge * I)

        # q_i = a_i^T (ATWA)^(-1) a_i
        q = (A @ ATWA_inv * A).sum(dim=1)              # (n,)
        w_new = torch.clamp(q, min=1e-32) ** (p / 2.0)

        diff = torch.max(torch.abs(w_new - w) / (torch.abs(w) + 1e-12)).item()
        w = w_new
        if diff < tol:
            break

    if normalize == 'sum_to_rank':
        r = torch.linalg.matrix_rank(A)
        s = w.sum()
        if s > 0:
            w = w * (r / s)

    return w


def select_by_weights(weights, select_num, selected_idx=None, repeat_entry=None):

    if isinstance(weights, torch.Tensor):
        w_np = weights.detach().cpu().numpy()
    else:
        w_np = np.asarray(weights)
    w_np = np.clip(w_np, a_min=0, a_max=None)
    if not np.any(w_np > 0):
        prob = np.ones_like(w_np) / len(w_np)
    else:
        prob = w_np / w_np.sum()

    num_selected = 0
    selected_idx = [] if selected_idx is None else copy(selected_idx)
    repeat_entry = dict() if repeat_entry is None else copy(repeat_entry)

    while num_selected < select_num:
        sid = np.random.choice(a=len(prob), replace=True, p=prob)
        already_exist_flag = sid in selected_idx
        selected_idx.append(sid)
        if not already_exist_flag:
            num_selected += 1
        else:
            if sid in repeat_entry:
                repeat_entry[sid] += 1
            else:
                repeat_entry[sid] = 2
    return selected_idx, repeat_entry



def fuse_weights_max(list_of_weights):

    Ws = [w if isinstance(w, torch.Tensor) else torch.tensor(w) for w in list_of_weights]
    W = torch.stack(Ws, dim=0)  # (m, n)
    return torch.max(W, dim=0).values



def select_client_weights(local_weights, target_client_id, k):

    local_weights = local_weights.tolist()
    if isinstance(local_weights, list):
        total_clients = len(local_weights)
        all_ids = list(range(total_clients))
    elif isinstance(local_weights, dict):
        all_ids = list(local_weights.keys())
    else:
        raise TypeError("local_weights 必须是 list 或 dict")


    other_ids = [cid for cid in all_ids if cid != target_client_id]
    random.seed(time.time())
    random.shuffle(other_ids)


    k = min(k, len(other_ids))
    random_ids = random.sample(other_ids, k)
    print("random_ids:", target_client_id, random_ids)

    selected_list = []
    if isinstance(local_weights, list):
        selected_list.append(local_weights[target_client_id])
        selected_list.extend([local_weights[i] for i in random_ids])
    else:  # dict
        selected_list.append(local_weights[target_client_id])
        selected_list.extend([local_weights[i] for i in random_ids])

    return selected_list




class CCCC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False) 
        self.class_order = torch.tensor(args["class_order"], device=args["gpu"])
        self.args = args
        self.mem_size = args['mem_size']
        self.r = args['r']
        self.ltc = args['ltc']
        self.transform, self.normalizer = self._get_norm_and_transform(self.args["dataset"])
        self.selected_data_indices = [[] for _ in range(args['num_users'])]
        self.retained_ds_all = [[] for _ in range(args['num_users'])]
        self.local_models = [[] for _ in range(args['num_users'])]
        
    def _get_norm_and_transform(self, dataset):

        if dataset == "cifar100":
            data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
        elif dataset == "cifar10":
            data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])

        elif dataset == "imagenet100":
            data_normalize = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            train_transform = transforms.Compose([
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
        
        elif dataset == "tiny_imagenet":
            data_normalize = dict(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
            train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
        return train_transform, Normalizer(**dict(data_normalize))

    def _get_client_dataset(self, client_idx):

        self.train_dataset, _ = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        
        client_indices = self.user_groups[client_idx]

        client_dataset = DatasetSplit(self.train_dataset, client_indices)
        return client_dataset

    def after_task(self):

        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        test_acc = self._compute_accuracy(self._old_network, self.test_loader)
        self.logger.info("After Task: %d,  Test ACC: %s" % (self._cur_task, str(test_acc))) 
        
        print("After Test Acc: %s" % test_acc)


    def _select_data_for_retention(self, local_weights):
        all_client_features = []

        for client_idx in range(self.args["num_users"]):
            try:
                local_weights_select = select_client_weights(local_weights, target_client_id=client_idx, k=self.args['k_value'])
                client_features = self._extract_client_features(client_idx, local_weights_select)
                # print(client_features)
                if len(client_features[0]) > 0:
                    all_client_features.append(client_features)
            except ValueError as e:
                print(f"Error extracting features for client {client_idx}: {e}")
                continue
            if len(client_features[0]) <= self.args['mem_size']:
                selected_idx = list(range(len(client_features[0])))
                self.selected_data_indices[client_idx] = selected_idx
            else:
                p = self.args['p_value']
                weights_list = []
                for feat in client_features:
                    w = lewis_weights(feat, p=p, max_iter=50, tol=1e-6, ridge=1e-8, dtype=torch.float64,
                                      normalize=None)
                    weights_list.append(w)
                
                final_weights = fuse_weights_max(weights_list)
                
                selected_idx = None
                repeat_entry = None
                selected_idx, repeat_entry = select_by_weights(final_weights, select_num=self.args['mem_size'],
                                                              selected_idx=selected_idx, repeat_entry=repeat_entry)
                # ipdb.set_trace()
                self.selected_data_indices[client_idx] = selected_idx

    
    def _extract_client_features(self, client_idx, local_weights):

        if not hasattr(self, "_known_classes") or not hasattr(self, "_total_classes"):
            raise ValueError("_known_classes or _total_classes is not defined.")
        

        client_dataset = self._get_client_dataset(client_idx)
        local_train_loader = DataLoader(
            client_dataset,
            batch_size=self.args["local_bs"],
            shuffle=False,
            num_workers=self.args["num_worker"],
            pin_memory=True,
            multiprocessing_context=self.args["mulc"],
            persistent_workers=True,
            drop_last=False
        )
        features = [[] for _ in range(self.args['k_value']+1)]

        with torch.no_grad():
            for batch_idx, (_, images, labels) in enumerate(local_train_loader):
                images = images.cuda()
                for c_idx, local_weight in enumerate(local_weights):
                    local_model = copy.deepcopy(self._network)
                    local_model.load_state_dict(local_weight)
                    local_model.eval()
                    output_list = local_model(images)
                    feature = output_list["att"]
                    # print(c_idx)
                    features[c_idx].append(feature.cpu())
        for f_id, feature in enumerate(features):      
            features[f_id] = torch.cat(feature, dim=0)   
        return features

    def _get_retained_dataset(self, client_idx):

        client_dataset = self._get_client_dataset(client_idx)
        retained_dataset = IndexedDataset(client_dataset, self.selected_data_indices[client_idx], self.transform)

        return retained_dataset         
    
    
    def incremental_train(self, data_manager, logger):
        self.logger = logger

        setup_seed(self.seed)
        self.data_manager = data_manager
        self._cur_task += 1

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
                
        self._network.update_fc(self._total_classes)
        self._network.cuda()
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset, _ = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )          

        test_dataset, _ = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"], multiprocessing_context=self.args["mulc"], persistent_workers=True)

        if self._cur_task > 0:
            old_test_dataset, _ = self.data_manager.get_dataset(
                np.arange(0, self._known_classes), source="test", mode="test"
            )
            self.old_loader = DataLoader(
                old_test_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"], multiprocessing_context=self.args["mulc"], persistent_workers=True
            )
            new_dataset, _ = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="test", mode="test"
            )
            self.new_loader = DataLoader(
                new_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"], multiprocessing_context=self.args["mulc"], persistent_workers=True
            )

        self._fl_train(train_dataset, self.test_loader)

    def _fl_train(self, train_dataset, test_loader):

        self._network.cuda()
#         ipdb.set_trace()
        self.best_model = None
        self.lowest_loss = np.inf

        prog_bar = tqdm(range(self.args["com_round"]))
        optimizer = torch.optim.SGD(self._network.parameters(), lr=self.args['local_lr'], momentum=0.9, weight_decay=self.args['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args["com_round"], eta_min=1e-3)

        user_groups, _ = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        self.user_groups = user_groups
        
        for _, com in enumerate(prog_bar):
            local_weights = np.array([copy.deepcopy(self._network.state_dict()) for i in range(self.args["num_users"])])
            local_models = {}
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            # idxs_users = range(self.args["num_users"])
            loss_weight = []
            local_p_list = []
            local_p_label_list = []
            for idx in idxs_users:
                local_train_ds_i = DatasetSplit(train_dataset, self.user_groups[idx])
                if self._cur_task > 0: 
                    print('xxx', local_train_ds_i.labels.shape)
                    local_train_ds_i = CustomConcatDataset([[local_train_ds_i], self.retained_ds_all[idx]], self.transform, self.args, self.logger)      
                    print('####', local_train_ds_i.labels.shape)    
                local_train_loader = DataLoader(local_train_ds_i, batch_size=self.args["local_bs"], shuffle=True, drop_last=False, num_workers=self.args["num_worker"], pin_memory=True, multiprocessing_context=self.args["mulc"], persistent_workers=True)

                if self._cur_task == 0:
                    w, total_loss = self._local_update(copy.deepcopy(self._network), local_train_loader, scheduler.get_last_lr()[0])
                else:
                    w, total_loss = self._local_finetune(self._old_network, copy.deepcopy(self._network), local_train_loader, self._cur_task, idx, scheduler.get_last_lr()[0])

                local_weights[idx] = copy.deepcopy(w)
                loss_weight.append(total_loss)
                if com == self.args["com_round"] - 1:
                    local_models[idx] = copy.deepcopy(w)
                del local_train_loader, w
                torch.cuda.empty_cache()
            
            scheduler.step()
            sum_loss = sum(loss_weight)
            if sum_loss < self.lowest_loss:
                self.lowest_loss = sum_loss
                self.best_model = copy.deepcopy(self._network.state_dict())

            global_weights = average_weights(local_weights[idxs_users])
            self._network.load_state_dict(global_weights)
            
            if self._cur_task <= self.tasks - 1 and com == self.args["com_round"] - 1:
                save_path = "CCCC_model_"+self.args["dataset"]+"_users_"+str(self.args["num_users"])+"_beta_05_localep_2_task_"+str(self._cur_task)+"_memsize_"+str(self.args["mem_size"])+"_round_"+str(com+1)+".pth"
                torch.save(self._network.state_dict(), save_path)
                print(f"Saved final model for last task to {save_path}")

            
            if com % 1 == 0 and com < self.args["com_round"]:
                if self._cur_task > 0:
                    scale = self.args['scale']      
                else:
                    scale = False
                test_acc = self._compute_accuracy(self._network, test_loader, scale=scale, old_classes=self._known_classes)
                if self._cur_task > 0:
                    test_old_acc = self._compute_accuracy(copy.deepcopy(self._network), self.old_loader)
                    test_new_acc = self._compute_accuracy(copy.deepcopy(self._network), self.new_loader)
                    print("Task {}, Test_accy {:.2f} O {} N {}".format(self._cur_task, test_acc, test_old_acc, test_new_acc))
                info = ("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(self._cur_task, com + 1, self.args["com_round"], test_acc))
                self.logger.info(info)
                prog_bar.set_description(info)  
                
        self._select_data_for_retention(local_weights)
        for idx in range(self.args["num_users"]):
            local_retained_ds = self._get_retained_dataset(idx)
            self.retained_ds_all[idx].append(local_retained_ds)           
        
        del self.best_model
        torch.cuda.empty_cache()
        # torch.save(local_models, f"BBBB_{self.args['dataset']}_local_models_task{self._cur_task}_round{self.args['com_round']}_{datetime.datetime.now().strftime('%Y-%m-%d-%H%M-%S')}.pth")
    
    
    def _local_update(self, model, train_data_loader, lr):

        model.train()
        total_loss = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args['weight_decay'])
#         ipdb.set_trace()
        for it in range(self.args["local_ep"]):
            epoch_loss_collector = []
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):           
                images, labels = images.cuda(), labels.cuda()
                output_list = model(images)
                output = output_list["logits"]
                loss_ce = F.cross_entropy(output, labels)
                loss = loss_ce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss_collector.append(loss.item())
                if it == 0:
                    total_loss += loss.detach()
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            self.logger.info('Epoch: %d Loss: %f' % (it, epoch_loss))
        
        return model.state_dict(), total_loss

    def _local_finetune(self, teacher, model, train_data_loader, task_id, client_id, lr):

        model.train()
        teacher.eval()
        total_loss = 0
        class_temperature_dict = {}

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args['weight_decay'])

        for it in range(self.args["local_ep"]):
            epoch_lossce_collector = []
            epoch_losstcl_collector = []
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
#                 ipdb.set_trace()
                images, labels = images.cuda(), labels.cuda()
                output_list = model(images)
                output = output_list["logits"]
                loss_ce = F.cross_entropy(output, labels)

                loss = loss_ce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_lossce_collector.append(loss_ce.item())
                # epoch_losstcl_collector.append(loss_tcl.item())

                if it == 0:
                    total_loss += loss.detach()
            epoch_lossce = sum(epoch_lossce_collector) / len(epoch_lossce_collector)
            # epoch_losstcl = sum(epoch_losstcl_collector) / len(epoch_losstcl_collector)

            self.logger.info('Epoch: %d Loss CE: %f' % (it, epoch_lossce))    

        return model.state_dict(), total_loss




def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval() 