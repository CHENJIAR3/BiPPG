import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam,AdamW,SGD
from tqdm import tqdm
import pandas as pd
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import logging



class PPG_trainer_twohands:
    def __init__(self, model, args):
        self.device = args.device
        self.lambda_mean = 15
        self.pred_th = 1.0
        self.model = model.to(self.device)
        self.model_type = args.model_type
        self.modelpath = args.modelpath
        self.criterion = nn.MSELoss(reduction="mean").to(self.device)
        self.label_mean = torch.tensor([120,74,73,63]).to(self.device)
        self.label_num = len(args.label_key)

        self.label_std = torch.tensor([18,11,13,14]).to(self.device)
        self.optimizer = Adam(self.model.parameters(),lr=args.lr)
        warmup_epochs = 10
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            eta_min = 0.01*args.lr,
            T_max=args.epochs - warmup_epochs
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        self.args = args

        # ── 新增：推理策略，默认 "split"，可设为 "mask" ──
        self.pred_strategy = getattr(args, "pred_strategy", "split")
        self.train_norm = getattr(args, "train_norm", True )
        # self.args = args
    def one_step(self,x,f,y):

        C = x.shape[1]
        # 一致性损失
        # ── 原策略：拆开左右手，各自独立推理 ──────────────────
        x_all = torch.cat([x[:, :C // 2], x[:, C // 2:]
                           #                       # ,f[:,:C],f[:,C:]
                           ], dim=0)

        pred_all = self.model(x_all)
        pred_left, pred_right = pred_all.chunk(2, dim=0)
        pred_mean = (pred_left + pred_right) / 2



        if y.shape[1]==len(self.label_mean) and self.train_norm:
            pred_norm = (pred_left - self.label_mean) / self.label_std
            pred_right_norm = (pred_right - self.label_mean) / self.label_std
            pred_mean_norm = (pred_mean - self.label_mean) / self.label_std
            y_norm = (y - self.label_mean[:self.label_num]) / self.label_std[:self.label_num]
        else:
            pred_norm = pred_left
            pred_right_norm = pred_right
            pred_mean_norm = pred_mean
            y_norm = y
        # # 简单的损失函数
        # 单手损失
        loss_left = self.criterion(pred_norm[:,:self.label_num], y_norm)
        loss_right = self.criterion(pred_right_norm[:,:self.label_num], y_norm)
        # 平均值损失
        loss_mean = self.criterion(pred_mean_norm[:,:self.label_num], y_norm)
        loss = ( loss_left + loss_right  + loss_mean )
        return loss

    def train(self, train_loader, val_loader):
        early_stop_counter = 0
        patience = self.args.patience #耐心值等于15
        best_loss = float('inf')
        early_stop_flag = False  # 早停触发标志
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            if early_stop_flag:
                break  # 触发早停，终止训练
            self.model.train()

            total_loss = 0.0
            mask_prob = 0.5
            for x,f,y in tqdm(train_loader):
                self.optimizer.zero_grad()

                x = x.to(self.device, dtype=torch.float32, non_blocking=True)
                f = f.to(self.device, dtype=torch.float32, non_blocking=True)
                y = y.to(self.device, dtype=torch.float32, non_blocking=True)
                loss = self.one_step(x,f,y)


                # self.optimizer_aux.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                # self.optimizer_aux.step()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪，防止爆炸
                total_loss += loss.item()
            # self.scheduler.step()
            # total_loss.backward()
            avg_train_loss = total_loss / len(train_loader)

            # 验证循环
            val_loss = self.validate(val_loader)

            print(f"Epoch [{epoch+1}/{self.args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            # 日志
            logging.info(
                f"Epoch {epoch+1}/{self.args.epochs} | "
                f"Train Loss: {avg_train_loss:.6f}| "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.modelpath)
                print("Saved best model")
                early_stop_counter = 0  # 重置计数器
            else:
                # 验证损失未改善，计数器+1
                early_stop_counter += 1
                print(f"Early stop counter: {early_stop_counter}/{patience} (Val Loss not improved)")

                # 计数器超过容忍度，触发早停
                if early_stop_counter >= patience:
                    print(
                        f"Early stopping triggered! No improvement for {patience} epochs. Best Val Loss: {best_loss:.4f}")
                    early_stop_flag = True

            # 训练结束，加载最佳模型（确保返回的是最优模型）
        # self.model.load_state_dict(torch.load(self.modelpath))
        model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        torch.save(model_to_save.state_dict(), self.modelpath)

        print(f"Training finished. Loaded best model with Val Loss: {best_loss:.4f}")
        return self.model
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x,f,y in tqdm(val_loader):
                # B, C, L = x.shape
                # 更高效的写法
                x = x.to(self.device, dtype=torch.float32, non_blocking=True)
                f = f.to(self.device, dtype=torch.float32, non_blocking=True)
                y = y.to(self.device, dtype=torch.float32, non_blocking=True)
                loss = self.one_step(x,f,y)

                total_loss += loss.item()
        avg_val_loss = total_loss / len(val_loader)
        return avg_val_loss

    def predict(self, test_loader):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.modelpath, weights_only=True))

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, f,y in tqdm(test_loader):
                x = x.to(self.device, dtype=torch.float, non_blocking=True)
                f = f.to(self.device, dtype=torch.float, non_blocking=True)
                y = y.to(self.device, dtype=torch.float, non_blocking=True)

                B, C, L = x.shape

                x_left  = x[:, :C//2]   # (N, C/2, L)
                x_right = x[:, C//2:]   # (N, C/2, L)
                pred_left  = self.model(x_left)
                pred_right = self.model(x_right)
                pred_con = torch.cat([
                pred_left,pred_right, ], dim=1)

                all_preds.append(pred_con.detach().cpu())
                all_targets.append(y.detach().cpu())

        predictions = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        return predictions, targets

