import math
import torch
import json
from functions.data import dataloader
import matplotlib.pyplot as plt
import pandas as pd


# === Function: Cosine Annealing Dropper for Observations
class CosineAnnealingDropper:
    def __init__(self, total_training_steps, start_obs=4096, end_obs=16, warmup_ratio=0.1):
        self.total_steps = total_training_steps
        self.start_obs = start_obs
        self.end_obs = end_obs
        self.warmup_steps = int(total_training_steps * warmup_ratio)
        self.current_step = 0

    def get_current_obs_num(self, current_step=None):
        if current_step is None:
            current_step = self.current_step

        if current_step < self.warmup_steps:
            return self.start_obs

        effective_step = current_step - self.warmup_steps
        effective_total = self.total_steps - self.warmup_steps

        progress = min(effective_step / effective_total, 1.0)

        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        current_obs = self.end_obs + (self.start_obs - self.end_obs) * cosine_decay

        return int(round(current_obs))

    def step(self):
        self.current_step += 1
        return self.get_current_obs_num()


# === Function: Random Drop for Observations
def batch_wise_random_drop(obs_batch, current_obs_num):

    batch_size, total_obs, feat_dim = obs_batch.shape

    # Generate a distinct random index for each sample in the batch.
    indices = torch.stack([
        torch.randperm(total_obs)[:current_obs_num]
        for _ in range(batch_size)
    ]).to(device=obs_batch.device)  # (B, current_obs_num)

    # Apply advanced indexing to collect data
    batch_indices = indices.unsqueeze(-1).expand(-1, -1, feat_dim)
    sparse_observations = torch.gather(obs_batch, 1, batch_indices).to(device=obs_batch.device)

    return sparse_observations


# === Function: Observation Spatial Reconstructions
def obs_field_reconstruction(obs_batch, field_size):
    B, N, T = obs_batch.shape
    width = T - 2

    # Initialize the output tensor
    obs_recon_batch = torch.zeros((B, width, field_size, field_size),
                                 device=obs_batch.device)

    # Batch calculate all coordinates
    x_pos = ((obs_batch[..., 0] + 1) / 2 * field_size).long()  # (B, N)
    y_pos = ((obs_batch[..., 1] + 1) / 2 * field_size).long()  # (B, N)

    # Batch Assignment Using Advanced Indexing
    for b in range(B):
        for w in range(width):
            obs_recon_batch[b, w, x_pos[b], y_pos[b]] = obs_batch[b, :, w + 2]

    return obs_recon_batch


# === Function: Create list for Stage 2 validation sparsity
def validation_sparsity_list(max_val, min_val):
    result = []
    current = max_val

    while current >= min_val:
        int_val = int(current)
        if not result or int_val != result[-1]:  # Avoid duplicate values
            result.append(int_val)

        if current / 2 < min_val:
            if min_val != result[-1]:  # Ensure that the minimum value is not duplicated.
                result.append(min_val)
            break

        current = current / 2

    return result


# === Function: Sampler: checkpoint_load
def sampler_checkpoint_load(path, stage, obs_sample, mode, device):
    from models.model import GlobalDiffuser
    from models.FM import FlowMatching
    with open(path+"/checkpoints/training_details.json", "r", encoding="utf-8") as f:
        training_details = json.load(f)

    # load model
    if stage == 1:
        model_path = training_details["s1_checkpoint_best_path"]
    else:
        if mode == "closest":
            model_path = training_details["s2_checkpoint_latest_path"].replace("latest", str(min(training_details["s2_loss_recorder"]["header"][2:], key=lambda x: abs(x - obs_sample))))
        elif mode ==  "best":
            model_path = training_details["s2_checkpoint_best_path"]
        else:
            model_path = training_details["s2_checkpoint_latest_path"]

    checkpoint = torch.load("../"+model_path)

    # Create Diffuser
    diffuser = GlobalDiffuser(
                field_size=checkpoint["size_params"]["field_size"],
                global_feat_len=training_details["model_params"]["global_feat_len"],
                field_channels=checkpoint["size_params"]["field_channels"],
                obs_vector_len=checkpoint["size_params"]["obs_vector_len"],
                num_res_blocks=training_details["model_params"]["num_res_blocks"],
                attention_resolutions=(16, 8),
                dropout=training_details["model_params"]["dropout"],
                channel_mult=(1, 2, 4, 8),
                num_heads=training_details["model_params"]["num_heads"],
                obs_element_num=checkpoint["size_params"]["obs_element_num"],
                spatial_recon = training_details["model_params"]["spatial_recon"],
                uncertain_quant = training_details["model_params"]["uncertain_quant"],
        )

    diffuser = diffuser.cuda(device)
    model = FlowMatching(model = diffuser,) 
    model = model.cuda(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"-> Field Inversion Checkpoint loaded: {model_path}")
    
    return model, training_details["model_params"]["uncertain_quant"]


# === Function: Sampler: test data generator
def sampler_test_data_generator(path, stage, obs_sample, mode, device, shuffle=True, position=None):
    with open(path+"/checkpoints/training_details.json", "r", encoding="utf-8") as f:
        training_details = json.load(f)

    data_loader,_ = dataloader(
                    data_dir = f"../data/{training_details["dataset_params"]["data_type"]}/testset.h5",
                    shuffle = shuffle,
                    batch_size = 32 if position==None else position+20,
                    num_workers = 4,
                )

    for inputs in data_loader:

        target_batch = inputs["target"].to(device)
        obs_batch = inputs["obs"].to(device)
        
        target = inputs["target"][0].unsqueeze(0).to(device) if position==None else inputs["target"][position].unsqueeze(0).to(device)
        obs = inputs["obs"][0].unsqueeze(0).to(device) if position==None else inputs["obs"][position].unsqueeze(0).to(device)
        break

    # obs_drop
    drop_obs = batch_wise_random_drop(obs, obs_sample)
    drop_obs_batch = batch_wise_random_drop(obs_batch, obs_sample)

    print(f"-> Dataloader: target shape = {target.shape} | observation shape = {obs.shape} | dropped_observation shape = {drop_obs.shape}")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 6))  # figsize控制整体画布大小
    axes[0].imshow(target.squeeze(0).squeeze(0).detach().cpu().numpy(), cmap="viridis")
    axes[0].set_title('Target')
    
    axes[1].imshow(obs_field_reconstruction(obs,64)[0][0].detach().cpu().numpy(), cmap="viridis")
    axes[1].set_title('Observation')
    
    axes[2].imshow(obs_field_reconstruction(drop_obs,64)[0][0].detach().cpu().numpy(), cmap="viridis")
    axes[2].set_title('Random Dropped Observation')
    
    plt.tight_layout()

    return target, obs, drop_obs, target_batch, obs_batch, drop_obs_batch


# === Function: Sampler: training loss visualizer
def sampler_training_visualizer(path):
    with open(path+"/checkpoints/training_details.json", "r", encoding="utf-8") as f:
        training_details = json.load(f)
    
    # stage 1 loss
    s1_loss =  training_details["s1_loss_recorder"]
    headers = s1_loss.pop('header') 
    s1_loss = pd.DataFrame(s1_loss, index=headers).T 
    plt.figure(figsize=(18, 6))
    plt.plot(range(1, len(s1_loss) + 1), s1_loss['train_loss'], label='Train Loss')
    plt.plot(range(1, len(s1_loss) + 1), s1_loss['test_loss'], label='Test Loss')
    plt.title('Field Diffuser: Stage 1 Training Loss Diagram')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # stage 2 loss
    s2_loss =  training_details["s2_loss_recorder"]
    headers = s2_loss.pop('header') 
    s2_loss = pd.DataFrame(s2_loss, index=headers).T 
    plt.figure(figsize=(18, 6))
    for loss_name in s2_loss.columns.tolist():
        if loss_name == "train_loss" or loss_name == "test_loss":
            plt.plot(range(1, len(s2_loss) + 1), s2_loss[loss_name], label=str(loss_name))
        else:
            plt.plot(range(1, len(s2_loss) + 1), s2_loss[loss_name], label=f"Random Sample: {loss_name}")
    
    plt.title('Field Diffuser: Stage 2 Training Loss Diagram')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)

