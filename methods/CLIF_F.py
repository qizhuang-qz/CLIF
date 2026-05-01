import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.dmc import partition_data, DatasetSplit, average_weights, setup_seed, pil_loader
import copy, wandb
import os, math
import pickle
from PIL import Image
from itertools import chain
import shutil
from torchvision import transforms
import ipdb
import copy
import random
import logging
import time

class IndexedDataset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.indices = indices
        self.images  = dataset.images[indices]
        self.labels  = dataset.labels[indices]
        self.transform = transform

    def __getitem__(self, idx):
        image = self.transform(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label

    def __len__(self):
        return len(self.indices)

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



def select_client_weights(local_weights, k):

    local_weights = local_weights.tolist()
    if isinstance(local_weights, list):
        total_clients = len(local_weights)
        all_ids = list(range(total_clients))
    elif isinstance(local_weights, dict):
        all_ids = list(local_weights.keys())
    else:
        raise TypeError("local_weights 必须是 list 或 dict")

    random.seed(time.time())
    random.shuffle(all_ids)

    random_ids = random.sample(all_ids, k)
    print("random_ids:", random_ids)

    selected_list = []
    if isinstance(local_weights, list):
        selected_list.extend([local_weights[i] for i in random_ids])
    else:  # dict
        selected_list.extend([local_weights[i] for i in random_ids])

    return selected_list




class CCCC_plus(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.class_order = torch.tensor(args["class_order"], device=args["gpu"])
        self.args = args
        self.r = args['r']
        self.ltc = args['ltc']
        self.transform, self.normalizer = self._get_norm_and_transform(self.args["dataset"])
        self.selected_data_indices = []
        self.retained_ds_all = [[] for _ in range(args['num_users'])]
        
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
            data_normalize = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=63 / 255),
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

        num_clients = self.args["num_users"]
        qbs = self.args["mem_size"]
        p   = self.args["p_value"]
        k   = self.args["k_value"]

        all_feats = []
        all_idx   = []
        valid     = []
        sizes     = []
        local_weights_select = select_client_weights(local_weights, k=k)
        for cid in range(num_clients):
            try:
                feat_list, idx_list = self._extract_client_features(cid, local_weights_select)
                if len(feat_list) == 0 or feat_list[0].numel() == 0:
                    all_feats.append([]); all_idx.append([]); valid.append(False); sizes.append(0)
                    continue
                all_feats.append(feat_list)
                all_idx.append(idx_list)
                valid.append(True)
                sizes.append(feat_list[0].shape[0])
            except ValueError as e:
                print(f"[Global-MV] Extract features error @client {cid}: {e}")
                all_feats.append([]); all_idx.append([]); valid.append(False); sizes.append(0)
    
        if not any(valid):
            print("[Global-MV] No client features. Skip selection.")

            self.selected_data_indices = [[] for _ in range(num_clients)]
            return
    
        first_valid = next(i for i, ok in enumerate(valid) if ok)
        num_views = len(all_feats[first_valid])

        per_client_view_weights = [ [] for _ in range(num_clients) ]

        for v in range(num_views):
            concat_v = []
            slices = []   # [client] -> (start, end)
            run = 0
            for cid in range(num_clients):
                if not valid[cid]:
                    slices.append((run, run))
                    continue
                Fcv = all_feats[cid][v]      # (n_i, d_v)
                n_i = Fcv.shape[0]
                concat_v.append(Fcv)
                slices.append((run, run + n_i))
                run += n_i
    
            if run == 0:

                for cid in range(num_clients):
                    if valid[cid]:
                        per_client_view_weights[cid].append(torch.zeros(sizes[cid], dtype=torch.float64))
                continue
    
            A_v = torch.cat(concat_v, dim=0)          # (sum n_i, d_v)
            w_v = lewis_weights(A_v, p=p, max_iter=50, tol=1e-6, ridge=1e-8,
                                dtype=torch.float64, normalize=None).reshape(-1)  # (sum n_i,)
    

            for cid in range(num_clients):
                if not valid[cid]:
                    continue
                s, e = slices[cid]
                per_client_view_weights[cid].append(w_v[s:e])

        if not isinstance(self.selected_data_indices, list) or len(self.selected_data_indices) != num_clients:
            self.selected_data_indices = [[] for _ in range(num_clients)]
    
        for cid in range(num_clients):
            if not valid[cid] or sizes[cid] == 0:
                self.selected_data_indices[cid] = []
                continue

            if sizes[cid] <= qbs:
                idx0 = all_idx[cid][0] if len(all_idx[cid]) > 0 else list(range(sizes[cid]))
                self.selected_data_indices[cid] = idx0
                continue
    
            fused_w = fuse_weights_max(per_client_view_weights[cid])   # (n_i,)
            fused_w = torch.clamp(fused_w, min=0)
            if torch.all(fused_w == 0):
                fused_w = torch.ones_like(fused_w, dtype=torch.float64)
    
            sel_local, _ = select_by_weights(fused_w, select_num=qbs, selected_idx=None, repeat_entry=None)

            idx0 = all_idx[cid][0] if len(all_idx[cid]) > 0 else list(range(sizes[cid]))
            if len(idx0) != sizes[cid]:
                idx0 = list(range(sizes[cid]))
            picked = [idx0[i] for i in sel_local]
    
            self.selected_data_indices[cid] = picked
    
        print("[Global-MV] Done multi-view global selection.")

    
    def _extract_client_features(self, client_idx, local_weights):

        if not hasattr(self, "_known_classes") or not hasattr(self, "_total_classes"):
            raise ValueError("_known_classes or _total_classes is not defined.")
    
        client_dataset = self._get_client_dataset(client_idx)
        local_train_loader = DataLoader(
            client_dataset, batch_size=self.args["local_bs"], shuffle=False,
            num_workers=self.args["num_worker"], pin_memory=True,
            multiprocessing_context=self.args["mulc"], persistent_workers=True, drop_last=False
        )
    
        num_views = self.args['k_value'] + 1
        feat_lists = [[] for _ in range(num_views)]

        local_indices = []

        self._network.eval()
        with torch.no_grad():
            for batch_idx, (_, images, labels) in enumerate(local_train_loader):
                images = images.cuda()
                output_list = self._network(images)
                feature = output_list["att"]
                feat_lists[0].append(feature.cpu())

            for batch_idx, (_, images, labels) in enumerate(local_train_loader):
                images = images.cuda()

                for v, local_weight in enumerate(local_weights):
                    local_model = copy.deepcopy(self._network)
                    local_model.load_state_dict(local_weight)
                    local_model.eval()
                    out = local_model(images)
                    feat_lists[v+1].append(out["att"].cpu())

                start_idx = batch_idx * self.args["local_bs"]
                end_idx   = start_idx + images.size(0)
                local_indices.extend(range(start_idx, end_idx))
    
        feat_list = [torch.cat(x, dim=0) if len(x) > 0 else torch.empty(0) for x in feat_lists]
        idx_list  = [local_indices]
        return feat_list, idx_list


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