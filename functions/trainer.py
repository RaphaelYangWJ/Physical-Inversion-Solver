import os
import math
import json
import timm
import torch
import random
from tqdm import tqdm
from torch.optim import Adam
from models.model import GlobalDiffuser
from datetime import datetime
from functions.data import dataloader
from models.FM import FlowMatching
from torch.utils.tensorboard import SummaryWriter
from functions.helper import CosineAnnealingDropper, batch_wise_random_drop, validation_sparsity_list
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# === Class: Diffuser Trainer
class FMTrainer:

    def __init__(self,args):
        # === set random seed
        torch.manual_seed(args.random_state)
        # === Training params
        self.device = [i for i in range(args.device)]
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.s1_epochs = args.s1_epochs
        self.cfg = args.cfg
        self.s2_epochs = args.s2_epochs
        self.min_obs = args.min_obs
        self.warmup_ratio = args.warmup_ratio
        self.forget_mix_ratio = args.forget_mix_ratio
        self.EMA = args.EMA
        self.grad_clip = args.grad_clip
        # === Diffuser params
        self.num_res_blocks = args.num_res_blocks
        self.dropout = args.dropout
        self.num_heads = args.num_heads
        self.global_feat_len = args.global_feat_len
        self.spatial_recon = args.spatial_recon
        self.time_stamps = args.time_stamps
        self.uncertain_quant = args.uncertain_quant
        # === Dataset params
        self.num_workers = args.num_workers
        self.data_type = args.data_type
        self.shuffle = args.shuffle
        self.trainset_dir = f"data/{args.data_type}/trainset.h5" # trainset path
        self.testset_dir = f"data/{args.data_type}/testset.h5" # testset path

        # === 1.0 Create output directory
        self.running_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_dir = f"output/{args.data_type}-{self.running_time}/logs"
        self.checkpoint_dir = f"output/{args.data_type}-{self.running_time}/checkpoints"

        # === 2.0 Train and Test dataloader create
        self.train_loader, self.size_params_train = dataloader(
            data_dir = self.trainset_dir,
            shuffle = args.shuffle,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
        )
        self.test_loader, self.size_params_test = dataloader(
            data_dir = self.testset_dir,
            shuffle = args.shuffle,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            )

        print("-> Dataloader: Data loaded for trainset and testset.")

        # Get params for datasets
        self.in_channels = self.size_params_train["field_channels"]
        self.out_channels = self.size_params_train["field_channels"]
        self.field_size = self.size_params_train["field_size"]
        self.obs_vector_len = self.size_params_train["obs_vector_len"]
        self.obs_element_num = self.size_params_train["obs_element_num"]
        self.obs_label_size = self.size_params_train["obs_label_size"]
        self.obs_label_channels = self.size_params_train["obs_label_channels"]
        self.trainset_size = self.size_params_train["data_length"]
        self.testset_size =     self.size_params_test["data_length"]

        print("-> Dataloader: Dataset Information:\n"
              f"# in_channels: {self.in_channels}\n"
              f"# out_channels: {self.out_channels}\n"
              f"# field_size: {self.field_size}\n"
              f"# obs_vector_len: {self.obs_vector_len}\n"
              f"# obs_element_num: {self.obs_element_num}\n"
              f"# obs_label_size: {self.obs_label_size}\n"
              f"# obs_label_channels: {self.obs_label_channels}\n"
              f"# trainset_size: {self.trainset_size}\n"
              f"# testset_size: {self.testset_size}\n"
              )


        # === 3.0 Cosine Annealing Dropper Create for Stage 2 Training
        self.cosine_dropper_train = CosineAnnealingDropper(
                            total_training_steps=math.ceil(self.trainset_size/self.batch_size)*self.s2_epochs,
                            start_obs=self.obs_element_num,
                            end_obs=self.min_obs,
                            warmup_ratio=self.warmup_ratio
        )


        # === 4.0 Backbone model initialization
        self.diffuser = GlobalDiffuser(
            field_size=self.field_size,
            global_feat_len=self.global_feat_len,
            field_channels=self.in_channels,
            obs_vector_len=self.obs_vector_len,
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=(16, 8),
            dropout=self.dropout,
            channel_mult=(1, 2, 4, 8),
            num_heads=self.num_heads,
            obs_element_num=self.obs_element_num,
            spatial_recon=self.spatial_recon,
            uncertain_quant=self.uncertain_quant
        )
        self.diffuser = self.diffuser.cuda(self.device[0])
        print("-> FM: Global Diffuser (Backbone model) loaded.")

        # === 5.0 DDPM framework initialization
        self.model = FlowMatching(model = self.diffuser)
        self.model = self.model.cuda(self.device[0])
        print("-> FM: FM Framework loaded.")

        # === 6.0 Remaining parts
        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        print(f"-> Optimizer: Initialized with learning rate={self.lr}")
        if self.lr_decay == True:
            self.lr_1_scheduler = CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=self.s1_epochs,
                T_mult=2,
                eta_min=1e-6,
            )
            self.lr_2_scheduler = CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=self.s2_epochs,
                T_mult=2,
                eta_min=1e-6,
            )
        print(f"-> Lr Schedule: Decayed Lr setup.")

        if self.EMA == True:
            self.ema = timm.utils.ModelEma(
                self.model,
                decay=0.9999,
                device=self.device[0]
            )
        print(f"-> EMA: EMA used.")
        print("\n")

        # Log save
        self.writer = SummaryWriter(self.log_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Initialize training epochs
        self.s1_current_epochs = 0
        self.s1_best_test_loss = float('inf')
        self.s1_loss_recorder = {"header": ["train_loss","test_loss"]}
        self.s2_current_epochs = 0
        self.s2_best_test_loss = float('inf')
        self.validation_sparsities = validation_sparsity_list(max_val=self.obs_element_num, min_val=self.min_obs)
        self.s2_loss_recorder = {"header": ["train_loss","test_loss"] + self.validation_sparsities}

    # ========== Stage 1 Training ==========
    def s1_train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Stage 1 Training: Epoch {self.s1_current_epochs + 1}")
        for inputs in progress_bar:
            # load data to GPU
            target = inputs["target"].to(self.device[0])
            obs = inputs["obs"].to(self.device[0])

            # clear gradients
            self.optimizer.zero_grad()

            # compute loss
            loss = self.model.forward(target, obs, self.cfg, self.uncertain_quant, 1)

            # back propagation
            loss.backward()

            # if gradient clipping
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.grad_clip)

            self.optimizer.step()

            # if EMA
            if self.EMA == True:
                self.ema.update(self.model)

            # record loss
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # If Lr_schedule
        if self.lr_decay == True:
            self.lr_1_scheduler.step()

        # compute average loss for one epoch
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/train", avg_loss, self.s1_current_epochs)
        return avg_loss
    def s1_validate_one_epoch(self):
        if self.EMA == True:
            self.ema.ema.eval()
            model_to_validate = self.ema.ema
        else:
            self.model.eval()
            model_to_validate = self.model

        total_loss = 0.0

        with torch.no_grad():
            for inputs in tqdm(self.test_loader, desc="Stage 1 Validating"):
                # load data to GPU
                target = inputs["target"].to(self.device[0])
                obs = inputs["obs"].to(self.device[0])

                # compute loss
                loss = model_to_validate.forward(target, obs, self.cfg, self.uncertain_quant, 2)
                
                # record loss
                total_loss += loss.item()

        # compute average loss
        avg_loss = total_loss / len(self.test_loader)
        self.writer.add_scalar("Loss/val", avg_loss, self.s1_current_epochs)
        return avg_loss
    def s1_train(self):
        # ==> Stage 1 Training...
        print("-> ################ Train: Stage 1 Training starts. ################")
        for epoch in range(self.s1_current_epochs, self.s1_epochs):
            self.s1_current_epochs = epoch
            # Train and validate
            train_loss = self.s1_train_one_epoch()
            test_loss = self.s1_validate_one_epoch()
            # save best checkpoints.
            if test_loss < self.s1_best_test_loss:
                self.s1_best_test_loss = test_loss
                self.save_checkpoint(train_loss, test_loss, 1, True, None)
            # print training information
            print(f"-> Train [Stage1] [Epoch:{epoch + 1}/{self.s1_epochs}]: "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
                  )
            # save latest checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(train_loss, test_loss, 1, False, None)

            # save train_loss and test_loss
            self.s1_loss_recorder[f"Epoch_{epoch + 1}"] = [train_loss, test_loss]

            # save training details
            self.training_details_save()

        print("-> Train: Stage 1 Training completed.\n")

    # ========== Stage 2 Training ==========
    def s2_train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Stave 2 Training: Epoch {self.s2_current_epochs + 1}")
        for inputs in progress_bar:
            # random dropper
            current_obs_num = self.cosine_dropper_train.get_current_obs_num()
            # load data to GPU
            target = inputs["target"].to(self.device[0])
            obs = inputs["obs"].to(self.device[0])

            # Mixed Dropping for Model Catastrophic Forgetting
            if (torch.rand(1).item() < self.forget_mix_ratio) and (current_obs_num < int(self.obs_element_num/2)):  # N%的batch review dense obs
                # Take Dense observations
                high_obs_nums = self.validation_sparsities[:len(self.validation_sparsities) // 2]
                review_obs_num = random.choice(high_obs_nums)
                obs_dropped = batch_wise_random_drop(obs, review_obs_num)
                difficulty_tag = f"[Review:{review_obs_num}]"
            else:
                # Normal training
                obs_dropped = batch_wise_random_drop(obs, current_obs_num)
                difficulty_tag = f"[Current:{current_obs_num}]"

            # clear Gradient
            self.optimizer.zero_grad()

            # compute loss
            loss = self.model.forward(target, obs_dropped, self.cfg, self.uncertain_quant, 2)

            # back propagation
            loss.backward()

            # if gradient clipping
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.grad_clip)

            self.optimizer.step()

            # if EMA
            if self.EMA == True:
                self.ema.update(self.model)

            # record loss
            total_loss += loss.item()
            progress_bar.set_postfix({
                "loss": loss.item(),
                "mode": difficulty_tag  # show curren tag
            })

            # update dropper
            self.cosine_dropper_train.step()

        # If Lr_schedule
        if self.lr_decay == True:
            self.lr_2_scheduler.step()

        # compute Average loss
        avg_loss = total_loss / len(self.train_loader)
        print(f"# Cosine Annealing Dropper: Current Obs Num = {current_obs_num}")
        self.writer.add_scalar("Loss/train", avg_loss, self.s2_current_epochs)
        return avg_loss
    def s2_validate_one_epoch(self):
        if self.EMA == True:
            self.ema.ema.eval()
            model_to_validate = self.ema.ema
        else:
            self.model.eval()
            model_to_validate = self.model
        total_loss = 0.0
        validation_results = {}  # Record validation result
        sparse_validation_results = []

        # validate
        with torch.no_grad():
            for inputs in tqdm(self.test_loader, desc="Stager 2 Validating"):
                # load data to GPU
                target = inputs["target"].to(self.device[0])
                obs = inputs["obs"].to(self.device[0])  #[B, 4096, 23]

                # validate foe each sparsities
                batch_results = {}
                for obs_num in self.validation_sparsities:

                    # Random drop for observations
                    sparse_obs = batch_wise_random_drop(obs, obs_num)

                    # compute loss
                    loss = model_to_validate.forward(target, sparse_obs, self.cfg, self.uncertain_quant, 2)

                    # record loss
                    batch_results[obs_num] = loss.item()
                    total_loss += loss.item()

                # record result for each batch
                for obs_num, loss_val in batch_results.items():
                    if obs_num not in validation_results:
                        validation_results[obs_num] = []
                    validation_results[obs_num].append(loss_val)

        # compute average loss for each sparsity
        for obs_num, losses in validation_results.items():
            avg_loss = sum(losses) / len(losses)
            self.writer.add_scalar(f"Loss/val_obs_{obs_num}", avg_loss, self.s2_current_epochs)
            print(f"-> Observation [{obs_num}] Loss: {avg_loss:.4f}")
            sparse_validation_results.append(avg_loss)

        # Return the loss of closest sparsity
        current_obs_num = self.cosine_dropper_train.get_current_obs_num()
        # Return the closest validating sparsity
        closest_obs = min(self.validation_sparsities, key=lambda x: abs(x - current_obs_num))
        main_avg_loss = sum(validation_results[closest_obs]) / len(validation_results[closest_obs])
        self.writer.add_scalar("Loss/val_main", main_avg_loss, self.s2_current_epochs)
        return main_avg_loss, sparse_validation_results
    def s2_train(self):
        # ==> Stage 2 Training...
        print("-> ################ Train: Stage 2 Training starts.################")
        for epoch in range(self.s2_current_epochs, self.s2_epochs):
            self.s2_current_epochs = epoch
            # Train and validate
            train_loss = self.s2_train_one_epoch()
            test_loss, sparse_validation_results = self.s2_validate_one_epoch()
            # save best checkpoints.
            if test_loss < self.s2_best_test_loss:
                self.s2_best_test_loss = test_loss
                self.save_checkpoint(train_loss, test_loss, 2, True, None)
            # print training information
            print(f"-> Train [Stage2] [Epoch:{epoch+1}/{self.s2_epochs}]: "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
                  )
            # save latest checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(train_loss, test_loss, 2, False, None)

            # key obs level checkping_save
            current_num = self.cosine_dropper_train.get_current_obs_num()
            if current_num in self.validation_sparsities:
                self.save_checkpoint(train_loss, test_loss, 2, False, current_num)

            # save train_loss and test_loss
            self.s2_loss_recorder[f"Epoch_{epoch + 1}"] = [train_loss, test_loss] + sparse_validation_results

            # save training details
            self.training_details_save()

        print("-> Train: Stage 2 Training completed.\n")

    # ========== Training Functions ==========
    def save_checkpoint(self, train_loss, test_loss, stage, is_best, s2_num):
        if is_best:
            torch.save({
                        'mode_stage': stage,
                        'epoch': self.s1_current_epochs+1 if stage == 1 else self.s2_current_epochs+1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'size_params': self.size_params_train,
                        'ema_state_dict': self.ema.ema.state_dict() if self.EMA==True else None,
                        'scheduler_state_dict_s1': self.lr_1_scheduler.state_dict() if self.lr_decay==True else None,
                        'scheduler_state_dict_s2': self.lr_2_scheduler.state_dict() if self.lr_decay == True else None,
                        }, self.checkpoint_dir+f"/stage_{stage}_best.pt")
            print(f"-> Save Checkpoints: [BEST] stage_{stage}_best.pt")

        else:
            if s2_num == None:
                torch.save({
                            'mode_stage': stage,
                            'epoch': self.s1_current_epochs+1 if stage == 1 else self.s2_current_epochs+1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'train_loss': train_loss,
                            'test_loss': test_loss,
                            'size_params': self.size_params_train,
                            'ema_state_dict': self.ema.ema.state_dict() if self.EMA==True else None,
                            'scheduler_state_dict_s1': self.lr_1_scheduler.state_dict() if self.lr_decay==True else None,
                            'scheduler_state_dict_s2': self.lr_2_scheduler.state_dict() if self.lr_decay == True else None,
                            }, self.checkpoint_dir+f"/stage_{stage}_latest.pt")
                print(f"-> Save Checkpoints: [LATEST] stage_{stage}_latest.pt")
            else:
                torch.save({
                    'mode_stage': stage,
                    'epoch': self.s1_current_epochs+1 if stage == 1 else self.s2_current_epochs+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'size_params': self.size_params_train,
                    'ema_state_dict': self.ema.ema.state_dict() if self.EMA==True else None,
                    'scheduler_state_dict_s1': self.lr_1_scheduler.state_dict() if self.lr_decay==True else None,
                    'scheduler_state_dict_s2': self.lr_2_scheduler.state_dict() if self.lr_decay == True else None,
                    }, self.checkpoint_dir+f"/stage_{stage}_{s2_num}.pt")
                print(f"-> Save Checkpoints: [LATEST] stage_{stage}_{s2_num}.pt")
                
    def training_pipeline(self, cont_path=None, stage=None, model_type=None):
        # new training
        if cont_path is None:
            self.s1_train()
            # training complete, load best.checkpoint for s2 training
            with open(self.checkpoint_dir+"/training_details.json", "r", encoding="utf-8") as f:
                training_details = json.load(f)
            checkpoint = torch.load(training_details[f"s1_checkpoint_best_path"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.s2_train()
        # continue training
        else:
            # read training details
            with open(cont_path+"/training_details.json", "r", encoding="utf-8") as f:
                training_details = json.load(f)
            if stage == 1:
                checkpoint = torch.load(training_details[f"s1_checkpoint_{model_type}_path"])
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.test_loss = checkpoint["test_loss"]
                self.train_loss = checkpoint["train_loss"]
                self.size_params_train = checkpoint["size_params"]
                print(f"-> Training Pipeline: Continue training [Stage 1] from epoch {checkpoint["epoch"]}")
                self.s1_current_epochs = checkpoint["epoch"]
                self.s1_loss_recorder = training_details["s1_loss_recorder"]
                if self.lr_decay == True:
                    self.lr_1_scheduler = checkpoint["scheduler_state_dict_s1"]
                    self.lr_2_scheduler = checkpoint["scheduler_state_dict_s2"]
                if self.EMA == True:
                    self.ema.ema.load_state_dict(checkpoint['ema_state_dict'])
                self.s1_train()
                self.s2_train()
            else:
                checkpoint = torch.load(training_details[f"s2_checkpoint_{model_type}_path"])
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.test_loss = checkpoint["test_loss"]
                self.train_loss = checkpoint["train_loss"]
                self.size_params_train = checkpoint["size_params"]
                print(f"-> Training Pipeline: Continue training [Stage 2] from epoch {checkpoint["epoch"]}")
                self.s2_current_epochs = checkpoint["epoch"]
                self.s1_loss_recorder = training_details["s1_loss_recorder"]
                self.s2_loss_recorder = training_details["s2_loss_recorder"]
                if self.lr_decay == True:
                    self.lr_1_scheduler = checkpoint["scheduler_state_dict_s1"]
                    self.lr_2_scheduler = checkpoint["scheduler_state_dict_s2"]
                if self.EMA == True:
                    self.ema.ema.load_state_dict(checkpoint['ema_state_dict'])
                self.s2_train()

        self.writer.close()
        print("-> Train: Training pipeline completed.")
    def training_details_save(self):
        train_params = {
            "training_date": self.running_time,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "s1_epochs": self.s1_epochs,
            "s2_epochs": self.s2_epochs,
            "cfg_value": self.cfg,
            "min_obs": self.min_obs,
            "warmup_ratio": self.warmup_ratio,
            "forget_mix_ratio": self.forget_mix_ratio,
            "EMA": self.EMA,
            "grad_clip": self.grad_clip,
            "lr_decay": self.lr_decay,
        }

        model_params = {
            "num_res_blocks": self.num_res_blocks,
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "global_feat_len": self.global_feat_len,
            "spatial_recon": self.spatial_recon,
            "time_stamps": self.time_stamps,
            "uncertain_quant": self.uncertain_quant,
        }

        dataset_params = {
            "data_type": self.data_type,
            "field_size": self.field_size,
            "field_channel": self.in_channels,
            "obs_vector_len": self.obs_vector_len,
            "obs_element_num": self.obs_element_num,
            "trainset_size": self.trainset_size,
            "testset_size": self.testset_size,
        }

        log_data = {
            "train_params": train_params,
            "model_params": model_params,
            "dataset_params": dataset_params,
            "s1_loss_recorder": self.s1_loss_recorder,
            "s2_loss_recorder": self.s2_loss_recorder,
            "s1_checkpoint_best_path": self.checkpoint_dir+"/stage_1_best.pt",
            "s1_checkpoint_latest_path": self.checkpoint_dir + "/stage_1_latest.pt",
            "s2_checkpoint_best_path": self.checkpoint_dir+"/stage_2_best.pt",
            "s2_checkpoint_latest_path": self.checkpoint_dir + "/stage_2_latest.pt"
        }

        # save to json file
        with open(self.checkpoint_dir + f"/training_details.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        print("->Training Details: Training Details saved.")