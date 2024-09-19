from dataclasses import dataclass
from diffusers import UNet1DModel
import torch
from diffusers import DDIMScheduler
from safetensors.torch import load_model
import torch.nn as nn



@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 64
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 20
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-hamed-256"  # the model name locally and on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()



model_diff = UNet1DModel(
    in_channels=1,
    out_channels=1,
    extra_in_channels=16,
    block_out_channels = (32, 32, 64, 64, 128, 128),
    down_block_types=(
    "DownBlock1DNoSkip",  # a regular ResNet downsampling block
    "AttnDownBlock1D",
    "AttnDownBlock1D",
    "AttnDownBlock1D",
    "AttnDownBlock1D",  # a ResNet downsampling block with spatial self-attention
    "DownBlock1D",
    ),
    up_block_types=(
        "UpBlock1D",  # a regular ResNet upsampling block
        "AttnUpBlock1D",  # a ResNet upsampling block with spatial self-attention
        "AttnUpBlock1D",
        "AttnUpBlock1D",
        "AttnUpBlock1D",
        "UpBlock1DNoSkip",
    )
) 


class FullyConnectedNN(nn.Module):
    def __init__(self, input_channels=256, output_channels=1, fc_list=None, dp_ratio=0.3):
        super(FullyConnectedNN, self).__init__()
        if fc_list is None:
            fc_list = [256, 256]

        fc_layers = []
        pre_channel = input_channels
        
        for k in range(len(fc_list)):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            
            if dp_ratio >= 0 and k == 0:
                fc_layers.append(nn.Dropout(dp_ratio))
        
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc_layers(x)


# model_bc = FullyConnectedNN()
# model_bc.load_state_dict(torch.load('/space/userfiles/khatouna/OpenPCDet_FS/pcdet/model_Adv_clss_epoch_40.pth'))
# # model_bc.load_state_dict(torch.load('/space/userfiles/khatouna/OpenPCDet_FS/pcdet/model_Adv_clss_epoch_40.pth'), weights_only=True)
# model_bc = model_bc.to("cuda")


optimizer = torch.optim.AdamW(model_diff.parameters(), lr=config.learning_rate)
noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule = 'squaredcos_cap_v2', rescale_betas_zero_snr = True, timestep_spacing = "linspace" )
noise_scheduler.set_timesteps(100)
args = (config, model_diff, noise_scheduler, optimizer)
safetensors_file = "/space/userfiles/khatouna/OpenPCDet_FS/ddpm-hamed-2561/unet/diffusion_pytorch_model.safetensors"
load_model(model_diff, safetensors_file)
model_diff = model_diff.to("cuda")


def image_guided_generation(image_tensor, state):
    model_bc = FullyConnectedNN()
    model_bc = model_bc.to("cuda")

    if image_tensor.shape[0] == 0:
        return [], []
    
    new_state_dict = {}
    for key in state.keys():
        new_key = 'fc_layers.' + key if key[0].isdigit() else key  # Add 'fc_layers.' to keys that start with digits
        new_state_dict[new_key] = state[key]

    model_bc.load_state_dict(new_state_dict)
    model_bc.eval()

    min_val = torch.zeros((image_tensor.shape[0], 1, 1), device=image_tensor.device)
    max_val = torch.zeros((image_tensor.shape[0], 1, 1), device=image_tensor.device)
    for i in range(image_tensor.shape[0]):

        min_val[i, 0, 0], _ = torch.min(image_tensor[i, 0, :], dim=0, keepdim=True)
        max_val[i, 0, 0], _ = torch.max(image_tensor[i, 0, :], dim=0, keepdim=True)
    
    normalized_tensor = (image_tensor - min_val) / (max_val - min_val)
    
    # Scale to range [-1, 1]
    normalized_image = normalized_tensor * 2 - 1


    target_score = model_bc(normalized_image.permute(0, 2, 1).detach()).detach()
    # a = torch.sigmoid(target_score)
    mask_harsh_samples = (torch.sigmoid(target_score) > 0.65).detach()
    normalized_image_harsh = normalized_image[mask_harsh_samples.view(-1, )]
    if normalized_image_harsh.shape[0] == 0:
        return [], []

    
    # torch.manual_seed(42)
    noise = torch.randn_like(normalized_image_harsh)    
    reverse = 0.5
    xt = noise_scheduler.add_noise(original_samples = normalized_image_harsh.clone().detach(), noise =noise,  timesteps=torch.tensor([noise_scheduler.timesteps[int(len(noise_scheduler.timesteps)* reverse)]]))
    g_image =normalized_image_harsh.clamp(-1, 1).detach()
    inference_timesteps = noise_scheduler.timesteps[int(-len(noise_scheduler.timesteps)*(1 - reverse)):]
    model_diff.eval()
    cnt = 0
    border = 0
    for t in inference_timesteps:
        # t = (torch.ones(normalized_image.shape[0]) * t).long().to(normalized_image.device)
        xt = xt.requires_grad_()

        model_output = model_diff(xt, t).sample
        iteration_step = noise_scheduler.step(
            model_output, t, xt, eta=0
        )
        x_t_minus_one, pred_original = iteration_step.prev_sample, iteration_step.pred_original_sample
        if g_image is not None and noise_scheduler.alphas_cumprod[t] !=0:
            mask1 = (((g_image != -1).detach()) * 1.0).float()
            mask2 = (((g_image == -1).detach()) * -1.0).float()
            pred_orig = pred_original * mask1 +  mask2
            difference = (g_image - pred_orig) 
            norm = torch.linalg.norm(difference)
            # norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0] 

            # a = model_bc(pred_orig.permute(0, 2, 1))
            target_score = model_bc(pred_orig.permute(0, 2, 1))
            source_dist = torch.sum(target_score) 
            context = norm 
            if torch.mean(torch.sigmoid(target_score)) < 0.40:
                # alpha = 0
                border = 1
                output = 6.0 * context
        
            elif t != 494 and torch.mean(torch.sigmoid(target_score)) >= 0.40:
                alpha = 1.0
                output = alpha * source_dist + (1.0 - alpha) * context
                cnt = cnt +1
                if cnt > 10 and border == 0:
                    return [], []


            elif t==494:
                alpha = 100
                output = alpha * source_dist

            # output = alpha * torch.sum(target_score) + (1.0 - alpha) * norm
            norm_grad = torch.autograd.grad(outputs= output, inputs=xt)[0] 
            # # print(alpha)
            print(norm)
            print(torch.mean(torch.sigmoid(target_score)))
            print(output)
            print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

            x_t_minus_one = x_t_minus_one - 1.0 *  norm_grad
        xt = x_t_minus_one
        xt = xt.detach()

    xt = xt.clamp(-1, 1)
    # print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
    # print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
    # print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
    # mask = g_image != -1
    # bs = xt.shape[0]
    # print(torch.sum(mask))
    # difference = mask * (g_image - xt)
    # norm = torch.linalg.norm(difference)
    # norm1 = torch.sum(torch.abs(difference))/torch.sum(mask)
    # norm2 = torch.sum(torch.abs(difference))/bs

    mask1 = (((g_image != -1).detach()) * 1.0).float()
    mask2 = (((g_image == -1).detach()) * -1.0).float()
    pred = xt * mask1 +  mask2
    a = torch.sigmoid(model_bc(pred.permute(0, 2, 1))).detach()
    mask = torch.sigmoid(model_bc(pred.permute(0, 2, 1))).detach() < 0.50
    if mask.sum() == 0:
        return [], []
    return normalized_image_harsh[mask.view(-1,)], xt[mask.view(-1,)] 