import torch
import argparse
from functions.trainer import FMTrainer


'''
Physical Inversion Solver Training Pipeline
'''



# Training Params Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Physical Inversion Solver")

    # === Training params
    parser.add_argument("--device", type=int, default=torch.cuda.device_count(), help="GPU device")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_decay", type=bool, default=False, help="decayed learning rate")
    parser.add_argument("--EMA", type=bool, default=False, help="EMA fro training")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping for Training")
    parser.add_argument("--s1_epochs", type=int, default=300, help="Epochs for Stage 1 Training")
    parser.add_argument("--cfg", type=float, default=0.1, help="Classifier Free Guidance Level")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for training")
    parser.add_argument("--s2_epochs", type=int, default=1000, help="Epochs for Stage 2 training")
    parser.add_argument("--min_obs", type=int, default=12, help="Minimum num of observations for inversion")
    parser.add_argument("--warmup_ratio", type=float, default=0.001, help="Warmup ratio for Stage 2 training")
    parser.add_argument("--forget_mix_ratio", type=float, default=0.1, help="Catastrophic Forgetting")

    # === Diffuser params (Backbone model)
    parser.add_argument("--num_res_blocks", type=int, default=3, help="DDPM Backbone: Num of ResBlocks")
    parser.add_argument("--dropout", type=float, default=0.1, help="DDPM Backbone: Dropout")
    parser.add_argument("--num_heads", type=int, default=16, help="DDPM Backbone: Num of heads")
    parser.add_argument("--global_feat_len", type=int, default=512, help="DDPM Backbone: Length of Global Feature")
    parser.add_argument("--spatial_recon", type=bool, default=True, help="DDPM Backbone: Spatial Reconstruction")
    parser.add_argument("--time_stamps", type=int, default=1000,  help="DDPM: Steps of denoising")
    parser.add_argument("--uncertain_quant", type=bool, default=True,  help="DDPM: Uncertain Quant activation")

    # === Data params
    parser.add_argument("--data_type", type=str, default="SHM", help="Dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="Num of workers")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle Dataset")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    trainer = FMTrainer(args=parse_args())
    trainer.training_pipeline()