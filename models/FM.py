import torch
import torch.nn as nn




# Class Flow Matching
class FlowMatching(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # 你的 Backbone UNet

    def get_xt(self, x1, x0, t):

        t = t.view(-1, 1, 1, 1)
        xt = (1 - t) * x0 + t * x1
        return xt

    def get_velocity_target(self, x1, x0):

        return x1 - x0

    def forward(self, x1, conditions, cfg, uncertainty=False, stage=None):

        b = x1.shape[0]
        device = x1.device

        # 1. sample t ~ U(0, 1)
        t = torch.rand(b, device=device)

        # 2. sample noise x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # 3. Generate interpolated samples xt and target velocity ut
        xt = self.get_xt(x1, x0, t)
        ut = self.get_velocity_target(x1, x0)

        # 4. Predict
        # 4.1 CFG - Classifier-Free Guidance
        if cfg != False:
            mask = torch.rand(x1.shape[0], device=x1.device) < cfg # 10%
            mask_indices = torch.where(mask)[0].to(x1.device)
            zeros = torch.zeros_like(conditions).to(x1.device)
            conditions[mask_indices] = zeros[mask_indices]

        model_output = self.model(xt, t, conditions)

        # 5. MSE Loss
        # loss = torch.mean((v_pred - ut) ** 2)
        if uncertainty == True:
            v_pred = model_output[:, 0:1, :, :] # Noise pred
            variance_pred = model_output[:, 1:2, :, :] # Variance pred
            if stage == 1:
                return torch.mean((v_pred - ut) ** 2)
            else: # stage 2
                main_loss =  torch.mean((v_pred - ut) ** 2)
                variance_reg = 0.001 * (variance_pred ** 2).mean()
                return main_loss + variance_reg
        else:
            return torch.mean((model_output - ut) ** 2) # Normal Loss

    @torch.no_grad()
    def sample(self, shape, steps=50, device="cuda", conditions=None):

        
        # 1. start from x(0)
        x = torch.randn(shape, device=device)
        dt = 1.0 / steps

        for i in range(steps):
            # current t
            t_val = i / steps
            t = torch.full((shape[0],), t_val, device=device)

            # 2. Vt pred
            v = self.model(x, t, conditions)[:, 0:1, :, :]

            # 3. Euler: x = x + v * dt
            x = x + v * dt

        return x