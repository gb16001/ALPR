# train class here
'''
dataset->model(amp)->loss/eval->optimizer(lr schedual;  ;...)->logger(print cli, tensorborad,save resume ckpt)->[train loop]
we can build class to assamble different api functions
'''

# wheels
from dynaconf import Dynaconf
from abc import ABC, abstractmethod
import torch
from torch import optim
import torch.nn as nn
import os
from torch.amp import autocast, GradScaler
import threading
from tqdm import tqdm
# our libs
import datasets
import models
import models.fullModel
import models.Loss
import tools as tools
from tools import Logger, resolve_attr, CudaPrefetcher

class BaseTainer_a_config:
    def __init__(self,conf_file:str):
        '''conf_file in yaml format'''
        args = self.args = Dynaconf(settings_files=[conf_file])
        self.train_loader=self.build_dataloader(args.dataset)
        self.model=self.build_model(args)
        self.criterion=self.build_lossFun(args.criterion) #loss
        self.optimizer=self.build_optim(self.model,self.args.optim)
        self.scheduler=self.build_lr_scheduler(self.optimizer,self.args,steps_per_epoch=len(self.train_loader))
        self.logger=self.build_logger(args)
        return
    
    @staticmethod
    @abstractmethod
    def build_dataloader(args):
        return
    @staticmethod
    @abstractmethod
    def build_model(args):
        return
    @staticmethod
    @abstractmethod
    def build_lossFun(args) -> models.Loss.CELoss:
        return
    @staticmethod
    @abstractmethod
    def build_optim(model,args):
        return
    @staticmethod
    @abstractmethod
    def build_lr_scheduler(optimizer,args):
        return
    @staticmethod
    @abstractmethod
    def build_logger(args)->Logger:
        return
    def train(self):
        '''train the hole config file'''
        self.train_an_epoch()
        return
    @abstractmethod
    def train_an_epoch(self):
        self.train_a_step()
        return
    @abstractmethod
    def train_a_step(self):
        return
    pass


class Trainer_a_conf(BaseTainer_a_config):
    def __init__(self, conf_file):
        super().__init__(conf_file)
        # int AMP grad scaler
        self.scaler=self.build_scaler()
        self.amp_cfg = self.build_amp_cfg(self.args)
        if "val_set" in self.args:
            self.eval_loader = self.build_dataloader(self.args.val_set)
            self.evaluator:models.Loss.LPs_evaluator = resolve_attr(models.Loss, self.args.val_set.evaluator.name)()
        else:
            self.eval_loader =  None
        return

    def build_scaler(self):
        scaler = GradScaler(self.args.device,enabled=self.args.GradScaler_enable)
        # scaler = GradScaler(enabled=self.args.GradScaler_enable)
        return scaler

    @staticmethod
    def build_dataloader(args):

        dataset_name = args.name
        preprocFn = getattr(
            datasets.PreprocFns.PreprocFuns, getattr(args, "preprocFun", 'None'), None
        )
        if dataset_name == None:
            dataset = datasets.Dataset_rand(20)
        else:
            # datasets.CCPD.CCPD_base()
            DatasetType = getattr(datasets, dataset_name)
            dataset: datasets.CCPD_base = (
                DatasetType(args.csvPath, preprocFun=preprocFn, **args.args)
                if hasattr(args, "args")
                else DatasetType(args.csvPath, preprocFun=preprocFn)
            )
        n_worker = args.n_worker
        dataloader = datasets.DataLoader(
            dataset,
            args.batch_size,
            shuffle=True,
            num_workers=n_worker,
            collate_fn=dataset.collate_fn,
            drop_last=getattr(args, 'drop_last', False),
            pin_memory=getattr(args, 'pin_memory', True),      # 固定内存，配合 non_blocking H→D 传输
            persistent_workers=n_worker > 0,                   # 跨 epoch 复用 worker 进程，避免反复 fork
            prefetch_factor=getattr(args, 'prefetch_factor', 4) if n_worker > 0 else None,  # 每个 worker 预取批数
            worker_init_fn=datasets._worker_init_fn
        )
        return dataloader

    @staticmethod
    def build_model(args):
        '''args: whole yaml cfg'''
        model_name=args.model.name
        device=args.device
        model_type=getattr(models.fullModel,model_name)
        model:torch.nn.Module=model_type().to(device)
        return model
        # if args.model.get('complile',True):
        #     return torch.compile(model,mode="reduce-overhead")
        # else:
        #     return model

    @staticmethod
    def build_lossFun(args):
        lossName=args.name
        lossType=resolve_attr(models.Loss, lossName)
        criterion= lossType(args)
        return  criterion
    @staticmethod
    def build_optim(model:nn.Module, args):

        optim_type:optim.Adam=getattr(optim,args.name)
        if not hasattr(args,'backbone_lr'):
            optimizer=optim_type(model.parameters(),lr=args.lr,betas=args.betas,weight_decay=args.weight_decay)
            return optimizer
        # 过滤出 backbone 的参数 ID
        bone_params_ids = list(map(id, model.bone.parameters()))

        # 过滤出除 backbone 以外的所有参数
        base_params = [p for p in model.parameters() if id(p) not in bone_params_ids]

        optimizer = optim_type([
            # 第一组：Backbone，使用更小的学习率
            {'params': model.bone.parameters(), 'lr': args.backbone_lr},
            # 第二组：其余所有组件，使用全局学习率
            {'params': base_params, 'lr': args.lr}
        ], 
        # 这里可以放公共参数，如果组内没写，就会默认用这里的
        betas=args.betas, 
        weight_decay=args.weight_decay
        )
        return optimizer
    @staticmethod
    def build_lr_scheduler(optimizer, args, steps_per_epoch:int=0):
        '''OneCycleLR needs steps_per_epoch'''
        if args.lr_scheduler.name == "LambdaLR":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1.0
            )
        elif args.lr_scheduler.name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.lr_scheduler.step_size,
                gamma=args.lr_scheduler.gamma,
            )
        elif args.lr_scheduler.name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=args.lr_scheduler.mode,       
                factor=args.lr_scheduler.factor,       
                patience=args.lr_scheduler.patience, 
            )
        elif args.lr_scheduler.name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.num_epochs,
                eta_min=args.lr_scheduler.eta_min
            )
        elif args.lr_scheduler.name=='OneCycleLR':
            scheduler=torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr_scheduler.max_lr,
                epochs=args.num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=args.lr_scheduler.pct_start,
                div_factor=args.lr_scheduler.div_factor,
                final_div_factor=args.lr_scheduler.final_div_factor,
            )
        elif args.lr_scheduler.name=='OneCycleRexLR':
            scheduler = tools.OneCycleRexLR(
                optimizer,
                max_lr=args.lr_scheduler.max_lr,
                epochs=args.num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=args.lr_scheduler.pct_start,
                div_factor=args.lr_scheduler.div_factor
            )
        else:
            scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler.name)(optimizer)
        return scheduler

    @staticmethod
    def build_logger(args,backup_cfg:bool=True):
        return Logger(args,backup_cfg=backup_cfg)

    @staticmethod
    def build_amp_cfg(cfg):
        if cfg.amp_dtype=='bf16':
            amp_cfg={'enabled':True,'dtype':torch.bfloat16}
        elif cfg.amp_dtype=='fp16':
            amp_cfg={'enabled':True,'dtype':torch.float16}
        else:
            amp_cfg={'enabled':False}
        return amp_cfg

    def _freeze_backbone_for_finetuning(self):
        """
        微调时冻结主干网络参数
        仅保留 LPR_tr.decoder 进行参数更新
        
        冻结的模块：
        - bone: Backbone（骨干网络）
        - neck: STN 模块
        - enPosEncoder: 编码器位置编码
        - mae_Tren: MAE Transformer encoder
        - decnn: 解码网络
        - LPR_tr.encoder: LPR Transformer encoder
        
        保持活跃的模块：
        - LPR_tr.decoder: LPR Transformer decoder
        - dePosEncode: 解码器位置编码
        - prior_Q: 先验查询
        """
        frozen_modules = [
            self.model.bone,
            self.model.neck,
            self.model.enPosEncoder,
            self.model.mae_Tren,
            self.model.decnn,
            self.model.LPR_tr.encoder
        ]

        frozen_count = 0
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
                frozen_count += 1

        # 确保 decoder 和其他必要模块是活跃的
        active_modules = [
            self.model.LPR_tr.decoder,
            self.model.dePosEncode,
            self.model.prior_Q
        ]

        active_count = 0
        for module in active_modules:
            if isinstance(module, nn.Parameter):
                module.requires_grad = True
                active_count += 1
            else:
                for param in module.parameters():
                    param.requires_grad = True
                    active_count += 1

        print(f"\n微调模式激活：")
        print(f"  ✓ 冻结参数数量: {frozen_count}")
        print(f"  ✓ 活跃参数数量: {active_count}")
        print(f"  ✓ 冻结的模块: bone, neck, enPosEncoder, mae_Tren, decnn, LPR_tr.encoder")
        print(f"  ✓ 活跃的模块: LPR_tr.decoder, dePosEncode, prior_Q")

    def train(self):
        self.model.to(self.args.device)
        self.criterion.to(self.args.device)   # 把 register_buffer（如 class_weights）移到 GPU，消除 forward 里的 DeviceCopy
        ckpt = self.logger.load_ckpt(self.model, self.optimizer, self.scheduler, self.scaler)

        # 在初始epoch加载预训练的MAE权重（如果指定）
        if self.logger.epoch == 0 and hasattr(self.args.model, 'pretrain_ckpt'):
            pretrain_ckpt_path = self.args.model.pretrain_ckpt
            if pretrain_ckpt_path and os.path.exists(pretrain_ckpt_path):
                self._load_pretrain_mae_weights(pretrain_ckpt_path)
                print(f"已加载预训练MAE权重: {pretrain_ckpt_path}")

        # 微调模式：冻结主干网络，仅优化 LPR decoder
        if self.logger.epoch == 0 and getattr(self.args.model, 'finetune_mode', False):
            self._freeze_backbone_for_finetuning()

        # mode="reduce-overhead"
        self.model_cpl=torch.compile(self.model, mode="reduce-overhead") if self.args.model.get('complile',True) else self.model
        self.criterion = torch.compile(self.criterion, mode="reduce-overhead") if self.args.model.get('complile',True) else self.criterion
        # 训练循环
        for epoch in range(self.logger.epoch, self.args.num_epochs):
            self.train_a_epoch(epoch)
        self.logger.close() # wait logger 
        return

    def _load_pretrain_mae_weights(self, ckpt_path):
        """
        从预训练checkpoint中加载MAE相关的权重到当前模型
        支持两种方式：
        1. 直接加载state_dict
        2. 从CkptBox对象中提取model_state_dict
        """
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=self.args.device)

        # 判断checkpoint的格式
        if hasattr(ckpt, 'model_state_dict'):
            # CkptBox格式
            pretrain_state_dict = ckpt.model_state_dict
        else:
            # 直接的state_dict
            pretrain_state_dict = ckpt

        # 获取当前模型的state_dict
        current_state_dict = self.model.state_dict()

        # 提取MAE相关的权重（同时兼容其他可加载的权重）
        mae_related_keys = {
            'mae_Tren', 'decnn',  # MAE特定的模块
            'LPR_tr', 'bone', 'neck', 'enPosEncoder',  # 可能共享的主干
            'dePosEncode', 'prior_Q', 'lpr_head'
        }

        loaded_count = 0
        for name, param in pretrain_state_dict.items():
            # 检查是否是MAE相关的模块
            is_mae_module = any(key in name for key in mae_related_keys)

            if is_mae_module and name in current_state_dict:
                if current_state_dict[name].shape == param.shape:
                    current_state_dict[name] = param
                    loaded_count += 1

        # 加载更新后的state_dict
        self.model.load_state_dict(current_state_dict)
        print(f"成功加载 {loaded_count} 个预训练权重")

    def train_a_epoch(self, epoch):
        self.model_cpl.train()
        self.logger.epoch_init(epoch,len(self.train_loader))

        prefetcher = CudaPrefetcher(self.train_loader, self.args.device)
        pbar = tqdm(prefetcher, total=len(self.train_loader), desc=f'Epoch {epoch}/{self.args.num_epochs}', dynamic_ncols=True)
        for batch_data in pbar:
            # batch_data 已在 GPU，无需再调用 mv_to_device
            batch_data: datasets.BatchBox
            self.optimizer.zero_grad(set_to_none=True)  # set_to_none 省去清零开销
            # AMP前向传播
            with autocast(self.args.device,**self.amp_cfg):
                outputs = self.model_cpl.forward(batch_data,**self.args.model.args)
                loss_box = self.criterion.forward(outputs, batch_data)
                loss=loss_box.total_loss
            # AMP反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 学习率调整对于 OneCycleLR
            if self.args.lr_scheduler.name in ['OneCycleLR', 'OneCycleRexLR']:
                self.scheduler.step()

            self.logger.step_add_1(loss=loss.detach())

            if (self.logger.global_step % 100) == 0:
                self.logger.log_step(loss_box)
            pass

        # eval model
        eval_results = self.eval_dateset(self.args)
        # 学习率调整逻辑
        if self.args.lr_scheduler.name == "ReduceLROnPlateau":
            # 自动缩小的 scheduler 需要传入监控指标
            self.scheduler.step(eval_results.LP_acc)
        elif self.args.lr_scheduler.name not in ['OneCycleLR', 'OneCycleRexLR']:
            # 传统的 scheduler 直接运行即可
            self.scheduler.step()

        # log epoch
        self.logger.log_epoch(eval_results, print_cli=True, optimizer=self.optimizer)

        # 保存检查点
        val_acc = eval_results.LP_acc
        best_acc = self.logger.save_ckpt(
            val_acc,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            need_history_ckpt=getattr(self.args.LOGGER, "need_history_ckpt", False),
        )

        return

    def eval_dateset(self, cfg):
        if self.eval_loader==None:
            return None
        self.model_cpl.eval()
        prefetcher = CudaPrefetcher(self.eval_loader, cfg.device)
        pbar = tqdm(prefetcher, total=len(self.eval_loader), desc='Eval', dynamic_ncols=True, leave=False)
        with torch.inference_mode():
            batch_box:datasets.BatchBox
            for batch_box in pbar:
                # batch_box 已在 GPU（prefetcher 异步搬运）
                with autocast(cfg.device,**self.amp_cfg):
                    outputs = self.model_cpl(batch_box,denoise=False,**cfg.model.args)
                self.evaluator.forward_batch(outputs,batch_box)
        eval_result=self.evaluator.statistic_Dataset()
        # print(eval_result)
        return eval_result

    pass

class Eval_a_conf(Trainer_a_conf):
    def __init__(self, conf_file):
        '''conf_file in yaml format'''
        args = self.args = Dynaconf(settings_files=[conf_file])
        # self.train_loader=self.build_dataloader(args.dataset)
        self.eval_loader = self.build_dataloader(self.args.val_set)
        self.model = self.build_model(self.args)
        # self.criterion=self.build_lossFun(args.criterion) #loss
        self.logger=self.build_logger(self.args,backup_cfg=False)
        self.evaluator = resolve_attr(models.Loss, self.args.val_set.evaluator.name)()
        self.amp_cfg = self.build_amp_cfg(self.args)
        return
    def change_dataset(self,csvPath:str):
        self.args.val_set.csvPath=csvPath
        self.eval_loader = self.build_dataloader(self.args.val_set)
        return
    def test(self):
        self.model.to(self.args.device)
        # self._load_pretrain_mae_weights(pretrain_ckpt_path)
        #         print(f"已加载预训练MAE权重: {pretrain_ckpt_path}")
        ckpt= self.logger.load_ckpt(self.model,ckpt_name=self.args.ckpt_name)
        self.model_cpl=torch.compile(self.model, mode="reduce-overhead") if self.args.model.get('complile',True) else self.model
        eval_results=self.eval_dateset(self.args)
        print('Final Eval Results:\n',eval_results)
        return eval_results
    pass
