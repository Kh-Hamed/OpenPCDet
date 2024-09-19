from dataclasses import dataclass
from torchvision import transforms
from diffusers import UNet1DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
from diffusers import DDIMPipeline
from diffusers.utils import make_image_grid
import os
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
from accelerate import notebook_launcher
from diffusers import DDIMScheduler
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import pycuda.driver as cuda
from resize_right import resize
from safetensors import safe_open
from safetensors.torch import load_model
matplotlib.use('Agg')

def get_num_gpus():
    cuda.init()
    # This subprocess will call `torch.cuda.device_count()` without initializing CUDA in the main process
    return cuda.Device.count()

def image_guided_generation(model, noise_scheduler, image_tensor):
    noise = torch.randn_like(image_tensor)    
    # a = noise_scheduler.timesteps[int(len(noise_scheduler.timesteps)*0.9)]
    # b = noise_scheduler.timesteps[int(len(noise_scheduler.timesteps)/2)]
    reverse = 0.5
    noisy_image = noise_scheduler.add_noise(original_samples = image_tensor.clone(), noise =noise,  timesteps=torch.tensor([noise_scheduler.timesteps[int(len(noise_scheduler.timesteps)* reverse)]]))
    g_image = noise_scheduler.add_noise(original_samples = image_tensor.clone(), noise =noise,  timesteps=torch.tensor([noise_scheduler.timesteps[int(len(noise_scheduler.timesteps)*0.97)]]))
    inference_timesteps = noise_scheduler.timesteps[int(-len(noise_scheduler.timesteps)*(1 - reverse)):-1]
    x = noisy_image.clone()
    torch.save((128*(g_image.clamp(-1, 1) + 1)).type(torch.uint8), '/space/userfiles/khatouna/Diffusion-Models-pytorch/x1.pt')
    model.eval()
    for t in tqdm(inference_timesteps):
        t = (torch.ones(image_tensor.shape[0]) * t).long().to(image_tensor.device)
        xt = (x.clone().requires_grad_(True))

        model_output = model(xt, t).sample
        iteration_step = noise_scheduler.step(
            model_output, t, xt, eta=0
        )
        x, pred_original = iteration_step.prev_sample, iteration_step.pred_original_sample
        D = 1
        shape = xt.shape
        shape_u = (shape[0], 3, shape[2], shape[3])
        shape_d = (shape[0], 3, int(shape[2] / D), int(shape[3] / D))
        if g_image is not None and noise_scheduler.alphas_cumprod[t[0]] !=0:
            difference = resize(resize(g_image, scale_factors=1.0/D, out_shape=shape_d), scale_factors=D, out_shape=shape_u) - resize(resize(pred_original, scale_factors=1.0/D, out_shape=shape_d), scale_factors=D, out_shape=shape_u)
            # difference = g_image - pred_original
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
            # l2_grad = torch.autograd.grad(l2_loss, xt)[0]
            x = x - 6 * norm_grad
        x = x.detach()

    x = (x.clamp(-1, 1) + 1) / 2
    image = (x * 255).type(torch.uint8)
    torch.save(image, '/space/userfiles/khatouna/Diffusion-Models-pytorch/x.pt')
    image = image.cpu().permute(0, 2, 3, 1).numpy()
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

    def resize(self, image):

        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # resize to 256x256
            transforms.ToTensor()           # convert to torch tensor
        ])
        
        return transform(image)  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = torch.load(img_path, weights_only=True)
        # image = Image.open(img_path).convert("RGB")
        
        # # Convert to two channels (grayscale and another custom channel)
        # image = image.convert("L").convert("RGB")  # Simulate two channels (gray image repeated)
        # image = np.array(image)[..., :2]  # Keep only two channels
        
        if self.transformation:

            image = self.transform(image)
            
        
        return image.unsqueeze(0)


@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 64
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 40
    gradient_accumulation_steps = 1
    learning_rate = 1e-3
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


def evaluate(config, epoch, pipeline):

    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images
    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)
    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")



model = UNet1DModel(
in_channels=1,
out_channels=1,
extra_in_channels=16,
block_out_channels = (32,64,128)
) 



def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),

    )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id

        accelerator.init_trackers("train_example")
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    global_step = 0
    # Now you train the model
    for epoch in range(config.num_epochs):

        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, leave= False)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model

        if accelerator.is_main_process:
            pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            #     evaluate(config, epoch, pipeline)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

                else:
                    pipeline.save_pretrained(config.output_dir)



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

train = True
noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule = 'squaredcos_cap_v2', rescale_betas_zero_snr = True, timestep_spacing = "linspace" )
noise_scheduler.set_timesteps(100)
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
CUDA_VISIBLE_DEVICES = "0,3"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
count = len(CUDA_VISIBLE_DEVICES.split(','))
if train:
    notebook_launcher(train_loop, args, num_processes= count)
else:

###########################################################################3

    # Step 1: Read the image
    image_path = '/space/userfiles/khatouna/Diffusion-Models-pytorch/images/archive/2/00000140_(2).jpg'  # Replace with your image path
    image = Image.open(image_path)
    image = image.resize((256, 256))
    # Step 2: Convert image to tensor and normalize values to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to PyTorch tensor with values in [0, 1]
    ])
    image_tensor = transform(image)
    print(image_tensor.min(), image_tensor.max())
    # Step 3: Normalize values to [-1, 1]
    image_tensor = (image_tensor - 0.5) * 2
    print(image_tensor.min(), image_tensor.max())
    a = noise_scheduler.timesteps
    safetensors_file = "/space/userfiles/khatouna/Diffusion-Models-pytorch/ddpm-hamed-256/unet/diffusion_pytorch_model.safetensors"
    # Load the weights using safetensors
    load_model(model, safetensors_file)
    weights = {}
    with safe_open(safetensors_file, framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    # Load the weights into the model
    # model.load_state_dict(weights)
    model = model.to("cuda")
    image_tensor = image_tensor.to("cuda")
    image_guided_generation(model, noise_scheduler, image_tensor.unsqueeze(0))

