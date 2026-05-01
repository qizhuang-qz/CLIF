import copy
import logging
import torch
import math
from torch import nn
from convs.cresnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50
from convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from convs.ucir_resnet import resnet18 as cosine_resnet18
from convs.ucir_resnet import resnet34 as cosine_resnet34
from convs.ucir_resnet import resnet50 as cosine_resnet50
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from convs.modified_represnet import resnet18_rep, resnet34_rep
from convs.resnet_cbam import resnet18_cbam, resnet34_cbam, resnet50_cbam
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import itertools


def get_convnet(args, pretrained=False):
    name = args["net"].lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained, args=args)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained, args=args)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained, args=args)
    elif name == "cosine_resnet18":
        return cosine_resnet18(pretrained=pretrained, args=args)
    elif name == "cosine_resnet32":
        return cosine_resnet32()
    elif name == "cosine_resnet34":
        return cosine_resnet34(pretrained=pretrained, args=args)
    elif name == "cosine_resnet50":
        return cosine_resnet50(pretrained=pretrained, args=args)
    elif name == "resnet18_rep":
        return resnet18_rep(pretrained=pretrained, args=args)
    elif name == "resnet18_cbam":
        return resnet18_cbam(pretrained=pretrained, args=args)
    elif name == "resnet34_cbam":
        return resnet34_cbam(pretrained=pretrained, args=args)
    elif name == "resnet50_cbam":
        return resnet50_cbam(pretrained=pretrained, args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        print(self.convnet.out_dim)
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        # self.label_emb = args['label_emb']
        self.le = None
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()
        self.proj = nn.Linear(512, 128)

    def update_fc(self, nb_classes):
        # if self.le is None:
        #     self.le = torch.mean(self.label_emb[:nb_classes], dim=0)
        #     self.le = self.le.detach()
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc.cuda()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        # print("\tIn Model: input size", x.shape)
        x = self.convnet(x)
        # out = self.fc(x["features"].detach() + self.le)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )


class IncrementalNet_ETF(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        self.fc = None
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        """更新 ETF 分类器（不可训练）"""
        self.fc = self.generate_fc(self.feature_dim, nb_classes).cuda()

    def generate_fc(self, in_dim, nb_classes):
        """生成 ETF 分类器"""
        return Proto_Classifier(in_dim, nb_classes)

    def forward(self, x, labels=None):
        x = self.convnet(x)
        features = x["features"]  # shape: [B, d]
        logits = self.fc(features, labels)
        x.update({"logits": logits})
        if self.gradcam:
            x["gradcam_gradients"] = self._gradcam_gradients
            x["gradcam_activations"] = self._gradcam_activations
        return x

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]


class IncrementalNet_CausalETF(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        self.fc = None
        self.embed = None  # 推理阶段的混淆向量
        if self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes, num_head=2, tau=16.0, alpha=0.15, gamma=0.03125, use_effect=True):
        """更新 Causal Norm 分类器"""
        self.fc = CausalNormETFClassifier(
            num_classes=nb_classes,
            feat_dim=self.feature_dim,
            use_effect=use_effect,
            num_head=num_head,
            tau=tau,
            alpha=alpha,
            gamma=gamma
        ).cuda()

    def forward(self, x, embed=None):
        x = self.convnet(x)
        features = x["features"]  # shape: [B, d]
        if self.training:
            logits = self.fc(features, embed=None)
        else:
            logits = self.fc(features, embed=embed)
        x.update({"logits": logits})
        if self.gradcam:
            x["gradcam_gradients"] = self._gradcam_gradients
            x["gradcam_activations"] = self._gradcam_activations
        return x

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)

    def unset_gradcam_hook(self):
        for h in self._gradcam_hooks:
            if h:
                h.remove()
        self._gradcam_hooks = [None, None]
        self._gradcam_gradients, self._gradcam_activations = [None], [None]


class IncrementalNet_Distance(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        self.label_emb = args['label_emb']
        self.lte_norm = args['lte_norm']
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()
        self.iter = 0

    def update_fc(self, nb_classes):
        self.le = copy.deepcopy(self.label_emb[:nb_classes])
        fc = self.generate_fc(nb_classes, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        self.iter += 1
        x = self.convnet(x)
        f = x["features"]
        f = torch.sigmoid(f) * self.lte_norm
        d = (torch.cdist(f, self.le.detach()))
        d = torch.exp(-(d - 0.3) * 1) * 10
        # print("------")
        # print(f[0])
        # print(self.le[0])
        if self.iter % 20 == 0:
            print(d[0])
        # print("------")
        # out = self.fc(d)
        # out = d
        # out.update(x)
        x['logits'] = d
        # if hasattr(self, "gradcam") and self.gradcam:
        #     out["gradcam_gradients"] = self._gradcam_gradients
        #     out["gradcam_activations"] = self._gradcam_activations

        return x

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )


class IL2ANet(IncrementalNet):

    def update_fc(self, num_old, num_total, num_aux):
        fc = self.generate_fc(self.feature_dim, num_total + num_aux)
        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:num_old] = weight[:num_old]
            fc.bias.data[:num_old] = bias[:num_old]
        del self.fc
        self.fc = fc


class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
                self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logics: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim:])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args))
        else:
            self.convnets.append(get_convnet(self.args))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim:])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.convnets.append(get_convnet(self.args))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma


class HybridAutoencoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.latent_dim = 128  # 潜在空间维度
        self.num_classes = 0  # 动态增长的类别数
        self.task_size = []  # 记录每个任务的类别数

        # 编码器 (基于ResNet-18修改)
        self.encoder = nn.Sequential(
            *list(resnet18(pretrained=False, args=args).children())[:-1])  # 移除原始全连接层
        self.encoder_fc = nn.Linear(512, self.latent_dim)  # 自定义潜在空间映射

        # 解码器 (4层CNN)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.Unflatten(1, (512, 1, 1)),
            nn.ConvTranspose2d(512, 256, 4, stride=2),  # 输出: 256x4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2),  # 输出: 128x10x10
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2),  # 输出: 64x22x22
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 6, stride=2),  # 输出: 3x50x50
            nn.Sigmoid()
        )

        # 动态类别中心管理
        self.register_buffer('class_centroids', torch.zeros(0, self.latent_dim))
        self.centroid_masks = {}  # 记录每个任务对应的中心索引

    def forward(self, x, return_features=False):
        # 编码过程
        z = self.encoder(x)  # [batch, 512, 1, 1]
        z = z.view(z.size(0), -1)  # [batch, 512]
        z = self.encoder_fc(z)  # [batch, latent_dim]
        print(z.shape)
        # 解码过程
        recon = self.decoder(z.view(-1, self.latent_dim, 1, 1))

        # 分类预测
        if self.class_centroids.size(0) > 0:
            distances = torch.cdist(z, self.class_centroids)  # [batch, num_classes]
            preds = torch.argmin(distances, dim=1)
        else:
            preds = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        return (z, recon, preds) if return_features else (recon, preds)

    def update_fc(self, total_classes, task_size=5):
        """动态扩展分类中心"""
        self.task_size.append(task_size)
        prev_centroids = self.class_centroids

        # 初始化新任务的中心（使用正态分布）
        new_centroids = torch.randn(task_size, self.latent_dim) * 0.01
        new_centroids = new_centroids.to(self.class_centroids.device)

        # 合并中心
        self.class_centroids = torch.cat([prev_centroids, new_centroids], dim=0)
        self.num_classes = self.class_centroids.size(0)

        # 记录新中心的索引范围
        start_idx = prev_centroids.size(0)
        end_idx = start_idx + task_size
        self.centroid_masks[len(self.task_size) - 1] = (start_idx, end_idx)

    def get_task_centroids(self, task_id):
        """获取指定任务的中心索引"""
        if task_id not in self.centroid_masks:
            raise ValueError(f"Invalid task ID: {task_id}")
        start, end = self.centroid_masks[task_id]
        return self.class_centroids[start:end]

    def set_class_centroids(self, new_centroids):
        """服务器更新全局中心"""
        if new_centroids.size(1) != self.latent_dim:
            raise ValueError(f"Dimension mismatch! Expected {self.latent_dim}, got {new_centroids.size(1)}")
        self.class_centroids = new_centroids.clone()
        self.num_classes = new_centroids.size(0)

    def get_encoder_params(self):
        """获取编码器参数（用于参数聚合）"""
        return list(self.encoder.parameters()) + list(self.encoder_fc.parameters())

    def get_decoder_params(self):
        """获取解码器参数（本地训练）"""
        return self.decoder.parameters()

    def classify(self, z):
        """单独的分类接口"""
        distances = torch.cdist(z, self.class_centroids)
        return -distances  # 负距离可以视为logits


class Proto_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Proto_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes - 1)) * torch.matmul(P, I - ((1 / num_classes) * one))
        self.proto = M.cuda()  # shape: [d, C]

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.randn(feat_in, num_classes)
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-06), \
            torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def load_proto(self, proto):
        self.proto = copy.deepcopy(proto)

    def get_proto(self):
        return self.proto

    def forward(self, features):
        """
        输入特征向量，输出每类的 logits（余弦相似度）
        features: [B, d]
        proto: [d, C]
        return: [B, C]
        """
        features = F.normalize(features, dim=1)  # [B, d]
        proto_norm = F.normalize(self.proto, dim=0)  # [d, C]
        logits = torch.matmul(features, proto_norm)  # [B, C]
        return logits


class CausalNormETFClassifier(nn.Module):
    def __init__(self, num_classes, feat_dim, use_effect=True, num_head=2, tau=16.0, alpha=0.15, gamma=0.03125):
        super(CausalNormETFClassifier, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_effect = use_effect
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.scale = tau / num_head
        self.norm_scale = gamma
        self.alpha = alpha

        # === 固定 ETF 原型 ===
        self.register_buffer("weight", self.build_etf(feat_dim, num_classes))  # [C, d]

        self.relu = nn.ReLU(inplace=True)

    def build_etf(self, d, C):
        """生成标准 ETF 原型矩阵 [C, d]"""
        a = np.random.randn(d, C)
        Q, _ = np.linalg.qr(a)
        P = torch.tensor(Q[:, :C]).float()  # [d, C]

        I = torch.eye(C)
        one = torch.ones(C, C)
        M = math.sqrt(C / (C - 1)) * torch.matmul(P, I - (1 / C) * one)  # [d, C]
        M = M.T  # [C, d]
        return F.normalize(M, dim=1)  # ETF 原型向量做归一化

    def forward(self, x, embed=None):
        """
        输入：
            - x: [B, d] 特征向量
            - embed: [d] 混淆变量嵌入向量（仅用于测试阶段）
        输出：
            - y_TDE: 移除混淆项后的 logits（测试阶段）
            - y_TE: 原始 logits（训练阶段）
        """
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)  # [C, d]
        normed_x = self.multi_head_call(self.l2_norm, x)  # [B, d]
        y_TE = torch.matmul(normed_x * self.scale, normed_w.T)  # [B, C]
        y_TDE = y_TE.clone()

        # 测试阶段移除 confounder 影响
        if (not self.training) and self.use_effect and embed is not None:
            if isinstance(embed, np.ndarray):
                embed = torch.from_numpy(embed).to(x.device)
            embed = embed.view(1, -1)

            normed_c = self.multi_head_call(self.l2_norm, embed)  # [1, d]

            x_heads = torch.chunk(normed_x, self.num_head, dim=1)
            c_heads = torch.chunk(normed_c, self.num_head, dim=1)
            w_heads = torch.chunk(normed_w, self.num_head, dim=1)

            output = []
            for xh, ch, wh in zip(x_heads, c_heads, w_heads):
                cos_val, _ = self.get_cos_sin(xh, ch)
                removed = xh - self.alpha * cos_val * ch
                output.append(torch.matmul(removed * self.scale, wh.T))
            y_TDE = sum(output)

        return y_TDE

    def forward_logits(self, x, embed=None):
        """
        输入：
            - x: [B, d] 特征向量
            - embed: [d] 混淆变量嵌入向量（仅用于测试阶段）
        输出：
            - y_TDE: 移除混淆项后的 logits（测试阶段）
            - y_TE: 原始 logits（训练阶段）
        """
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)  # [C, d]
        normed_x = self.multi_head_call(self.l2_norm, x)  # [B, d]
        y_TE = torch.matmul(normed_x * self.scale, normed_w.T)  # [B, C]
        # y_TDE = y_TE.clone()

        # 测试阶段移除 confounder 影响

        if isinstance(embed, np.ndarray):
            embed = torch.from_numpy(embed).to(x.device)
        embed = embed.view(1, -1)

        normed_c = self.multi_head_call(self.l2_norm, embed)  # [1, d]

        x_heads = torch.chunk(normed_x, self.num_head, dim=1)
        c_heads = torch.chunk(normed_c, self.num_head, dim=1)
        w_heads = torch.chunk(normed_w, self.num_head, dim=1)

        output = []
        for xh, ch, wh in zip(x_heads, c_heads, w_heads):
            cos_val, _ = self.get_cos_sin(xh, ch)
            removed = xh - self.alpha * cos_val * ch
            output.append(torch.matmul(removed * self.scale, wh.T))
        y_TDE = sum(output)

        return y_TE, y_TDE


    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(dim=1, keepdim=True) / (
                torch.norm(x, dim=1, keepdim=True) * torch.norm(y, dim=1, keepdim=True) + 1e-8
        )
        sin_val = torch.sqrt(1 - cos_val ** 2 + 1e-8)
        return cos_val, sin_val

    def l2_norm(self, x):
        return x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)

    def causal_norm(self, x, weight):
        norm = torch.norm(x, dim=1, keepdim=True)
        return x / (norm + weight)

    def multi_head_call(self, func, x, weight=None):
        x_chunks = torch.chunk(x, self.num_head, dim=1)
        if weight is not None:
            y_chunks = [func(chunk, weight) for chunk in x_chunks]
        else:
            y_chunks = [func(chunk) for chunk in x_chunks]
        return torch.cat(y_chunks, dim=1)

#
# class CausalNormETFClassifier(nn.Module):
#     def __init__(self, num_classes, feat_dim, use_effect=True, num_head=2, tau=32.0, alpha=0.15, gamma=0.03125):
#         super(CausalNormETFClassifier, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.use_effect = use_effect
#         self.num_head = num_head
#         self.head_dim = feat_dim // num_head
#         self.scale = tau / num_head
#         self.norm_scale = gamma
#         self.alpha = alpha
#
#         self.register_buffer("weight", self.build_etf(feat_dim, num_classes))
#         self.relu = nn.ReLU(inplace=True)
#
#     def build_etf(self, d, C):
#         a = np.random.randn(d, C)
#         Q, _ = np.linalg.qr(a)
#         P = torch.tensor(Q[:, :C]).float()
#         I = torch.eye(C)
#         one = torch.ones(C, C)
#         M = math.sqrt(C / (C - 1)) * torch.matmul(P, I - (1 / C) * one)
#         M = M.T
#         return F.normalize(M, dim=1)
#
#     def forward(self, x, embed=None):
#         normed_x = self.multi_head_call(self.l2_norm, x)  # [B, d]
#
#         if (not self.training) and self.use_effect and embed is not None:
#             if isinstance(embed, np.ndarray):
#                 embed = torch.from_numpy(embed).to(x.device)
#             embed = embed.view(1, -1)
#             normed_c = self.multi_head_call(self.l2_norm, embed)  # [1, d]
#
#             normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)  # [C, d]
#             w_heads = torch.chunk(normed_w, self.num_head, dim=1)
#             c_heads = torch.chunk(normed_c, self.num_head, dim=1)
#
#             de_biased_w_heads = []
#             for wh, ch in zip(w_heads, c_heads):
#                 cos_val = (wh * ch).sum(dim=1, keepdim=True)  # [C, 1]
#                 removed = wh - self.alpha * cos_val * ch
#                 de_biased_w_heads.append(removed)
#
#             de_biased_w = torch.cat(de_biased_w_heads, dim=1)  # [C, d]
#             y_TDE = torch.matmul(normed_x * self.scale, de_biased_w.T)
#             return y_TDE
#
#         else:
#             normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)  # [C, d]
#             y_TE = torch.matmul(normed_x * self.scale, normed_w.T)
#             return y_TE
#
#     def get_cos_sin(self, x, y):
#         cos_val = (x * y).sum(dim=1, keepdim=True) / (
#             torch.norm(x, dim=1, keepdim=True) * torch.norm(y, dim=1, keepdim=True) + 1e-8
#         )
#         sin_val = torch.sqrt(1 - cos_val ** 2 + 1e-8)
#         return cos_val, sin_val
#
#     def l2_norm(self, x):
#         return x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
#
#     def causal_norm(self, x, weight):
#         norm = torch.norm(x, dim=1, keepdim=True)
#         return x / (norm + weight)
#
#     def multi_head_call(self, func, x, weight=None):
#         x_chunks = torch.chunk(x, self.num_head, dim=1)
#         if weight is not None:
#             y_chunks = [func(chunk, weight) for chunk in x_chunks]
#         else:
#             y_chunks = [func(chunk) for chunk in x_chunks]
#         return torch.cat(y_chunks, dim=1)
#
#
#






