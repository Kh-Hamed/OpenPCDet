from dataclasses import dataclass
from diffusers import UNet1DModel
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm.auto import tqdm
from diffusers import DDIMScheduler
import matplotlib
import pycuda.driver as cuda
from resize_right import resize
from safetensors.torch import load_model
matplotlib.use('Agg')

def get_num_gpus():
    cuda.init()
    return cuda.Device.count()

def image_guided_generation(model, noise_scheduler, image_tensor):
    noise = torch.randn_like(image_tensor)    
    reverse = 0.5
    x = noise_scheduler.add_noise(original_samples = image_tensor.clone(), noise =noise,  timesteps=torch.tensor([noise_scheduler.timesteps[int(len(noise_scheduler.timesteps)* reverse)]]))
    # g_image = noise_scheduler.add_noise(original_samples = image_tensor.clone(), noise =noise,  timesteps=torch.tensor([noise_scheduler.timesteps[int(len(noise_scheduler.timesteps)*0.6)]]))
    g_image =image_tensor.clamp(-1, 1)
    # g_image = g_image.clamp(-1, 1)
    inference_timesteps = noise_scheduler.timesteps[int(-len(noise_scheduler.timesteps)*(1 - reverse)):-1]
    model.eval()
    for t in tqdm(inference_timesteps):
        t = (torch.ones(image_tensor.shape[0]) * t).long().to(image_tensor.device)
        xt = (x.clone().requires_grad_(True))

        model_output = model(xt, t).sample
        iteration_step = noise_scheduler.step(
            model_output, t, xt, eta=0
        )
        x, pred_original = iteration_step.prev_sample, iteration_step.pred_original_sample
        if g_image is not None and noise_scheduler.alphas_cumprod[t[0]] !=0:
            difference = g_image - pred_original
            norm = torch.linalg.norm(difference)
            print(norm)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
            x = x -  1 * norm_grad
        x = x.detach()

    x = x.clamp(-1, 1)
    difference = abs(g_image - x)
    norm = torch.linalg.norm(difference)
    a = torch.mean(difference)
    import torch.nn.functional as F
    cos_sim = 1- F.cosine_similarity(g_image, x, dim=2)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(g_image.squeeze(0).squeeze(0).cpu().numpy(), label='Tensor 1')
    plt.title('Tensor 1')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x.squeeze(0).squeeze(0).cpu().numpy(), label='Tensor 2')
    plt.title('Tensor 2')
    plt.legend()

    plt.show()
    return image
    


class CustomTwoChannelImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.pth'))]
        self.transformation = True

    def transform(self, image):

        assert image.ndim == 1, "Tensor must be 3-dimensional (c, h, w)"
        
        # Calculate min and max values per channel
        min_values = image.amin(dim=(0), keepdim=True)
        max_values = image.amax(dim=(0), keepdim=True)
        
        # Normalize to range [0, 1]
        normalized_tensor = (image - min_values) / (max_values - min_values)
        
        # Scale to range [-1, 1]
        normalized_image = normalized_tensor * 2 - 1

        
        return normalized_image     
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = torch.load(img_path, weights_only=True)
        
        if self.transformation:

            image = self.transform(image)
            
        
        return image.unsqueeze(0)


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
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()



model = UNet1DModel(
in_channels=1,
out_channels=1,
extra_in_channels=16,
block_out_channels = (32,64,128)
)

# Create the dataset and DataLoader
dataset_path = '/space/userfiles/khatouna/OpenPCDet_FS/data/roi_features'  # Replace with your images directory path
dataset = CustomTwoChannelImageDataset(image_dir=dataset_path)
train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),

)

noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule = 'squaredcos_cap_v2', rescale_betas_zero_snr = True, timestep_spacing = "linspace" )
noise_scheduler.set_timesteps(100)
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
CUDA_VISIBLE_DEVICES = "0,3"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
count = len(CUDA_VISIBLE_DEVICES.split(','))


###########################################################################3

# Step 1: Read the image
image_path = '/space/userfiles/khatouna/OpenPCDet_FS/data/roi_features/roi_0.pth'  # Replace with your image path
image = torch.load(image_path,weights_only=True)
min_values = image.amin(dim=(0), keepdim=True)
max_values = image.amax(dim=(0), keepdim=True)

# Normalize to range [0, 1]
normalized_tensor = (image - min_values) / (max_values - min_values)

# Scale to range [-1, 1]
image_tensor = normalized_tensor * 2 - 1
print(image_tensor.min(), image_tensor.max())
a = noise_scheduler.timesteps
safetensors_file = "/space/userfiles/khatouna/OpenPCDet_FS/ddpm-hamed-256/unet/diffusion_pytorch_model.safetensors"
# Load the weights using safetensors
load_model(model, safetensors_file)
model = model.to("cuda")
image_tensor = image_tensor.to("cuda")
image_guided_generation(model, noise_scheduler, image_tensor.unsqueeze(0).unsqueeze(1))

