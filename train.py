import torch
from torch.nn import functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import gridspec
from model.denomamba_arch import DenoMamba
from data import create_loaders_mix
from options import TrainOptions

np.random.seed(392)
torch.manual_seed(392)


opt = TrainOptions().parse()


full_dose_path = opt.full_dose_path
quarter_dose_path = opt.quarter_dose_path
dataset_ratio = opt.dataset_ratio
train_ratio = opt.train_ratio
batch_size = opt.batch_size
learning_rate = opt.learning_rate
max_epoch = opt.max_epoch
continue_to_train = opt.continue_to_train
ckpt_path = opt.ckpt_path
batch_number = opt.batch_number
validation_freq = opt.validation_freq
save_freq = opt.save_freq
path_to_save = opt.path_to_save

in_ch = opt.in_ch
out_ch = opt.out_ch
dim = opt.dim
num_blocks = opt.num_blocks
n_layer = opt.n_layer
num_refinement_blocks = opt.num_refinement_blocks


if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

def get_pixel_loss(target, prediction):
    return F.l1_loss(prediction, target)

def calculate_psnr(original, reconstructed):
    mse = F.mse_loss(original, reconstructed)
    if mse == 0:
        return float('inf')

    max_pixel_value = 1

    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


trainloader, validloader = create_loaders_mix(
    full_dose_path=full_dose_path,
    quarter_dose_path=quarter_dose_path,
    dataset_ratio=dataset_ratio,
    train_ratio=train_ratio,
    batch_size=batch_size
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


net_model = DenoMamba(
    inp_channels=in_ch,
    out_channels=out_ch,
    dim=dim,
    num_blocks=num_blocks,
    num_refinement_blocks=num_refinement_blocks,
).to(device)

lr = learning_rate

optimG = torch.optim.Adam(net_model.parameters(), lr=lr)

# Calculate model size
model_size = sum(param.numel() for param in net_model.parameters())
print("Model params: %.2f M" % (model_size / 1e6))

# Load checkpoint if continuing training
if continue_to_train:
    ckpt_model = torch.load(ckpt_path)
    net_weights = ckpt_model["net_model"]
    optimG_weights = ckpt_model["optimG"]
    net_model.load_state_dict(net_weights)
    optimG.load_state_dict(optimG_weights)
    start_epoch = ckpt_model["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0

train_loss = []
for epoch in range(start_epoch, max_epoch):
    with tqdm(trainloader, unit="batch") as tepoch:
        tmp_tr_loss = 0
        tr_sample = 0
        net_model.train()
        total_psnr = []
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")

            # Move data to device
            condition = data.to(device)
            x_0 = target.to(device)

            # Forward pass
            recon_train = net_model(condition)

            # Compute loss
            optimG.zero_grad()
            pixel_loss = get_pixel_loss(recon_train, x_0)

            loss_G = pixel_loss
            loss_G.backward()

            # Gradient clipping and optimizer step
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), 1)
            optimG.step()

            tmp_tr_loss += loss_G.item()
            tr_sample += len(data)

            tepoch.set_postfix({"Loss": loss_G.item()})

            # Calculate PSNR
            psnr_score = calculate_psnr(recon_train, x_0)
            total_psnr.append(psnr_score)
    avg_train_psnr = np.mean(total_psnr)
    print("Average PSNR in Train: {:.4f}".format(avg_train_psnr))
    train_loss.append(tmp_tr_loss / tr_sample)
    
    # Validation step
    if (epoch + 1) % validation_freq == 0:
        valid_psnr = []
        net_model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(validloader):

                condition = data.to(device)
                x_0 = target.to(device)
                recon_train = net_model(condition)

                psnr_score = calculate_psnr(recon_train, x_0)
                valid_psnr.append(psnr_score)

                if batch_idx == batch_number:
                    fig = plt.figure()
                    fig.set_figheight(8)
                    fig.set_figwidth(28)
                    spec = gridspec.GridSpec(
                        ncols=3,
                        nrows=1,
                        width_ratios=[1, 1, 1],
                        wspace=0.01,
                        hspace=0.01,
                        height_ratios=[1],
                        left=0,
                        right=1,
                        top=1,
                        bottom=0,
                    )

                    img = condition[0]
                    img = img.data.squeeze().cpu()

                    ax = fig.add_subplot(spec[0])
                    ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
                    ax.axis("off")
                    ax.set_title("Input")

                    img = recon_train[0]
                    img = img.data.squeeze().cpu()

                    ax = fig.add_subplot(spec[1])
                    ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
                    ax.axis("off")
                    ax.set_title("Reconstructed")

                    img = target[0]
                    img = img.data.squeeze().cpu()

                    ax = fig.add_subplot(spec[2])
                    ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
                    ax.axis("off")
                    ax.set_title("Ground Truth")

                    plt.show()
        avg_valid_psnr = np.mean(valid_psnr)
        print("Average PSNR in Validation: {:.4f}".format(avg_valid_psnr))

    # Save model checkpoint
    if (epoch + 1) % save_freq == 0:
        ckpt = {
            "net_model": net_model.state_dict(),
            "optimG": optimG.state_dict(),
            "epoch": epoch
        }
        save_path = os.path.join(path_to_save, f"model_epoch_{epoch + 1}.pkl")
        torch.save(ckpt, save_path)
        print(f"Model saved at {save_path}")
