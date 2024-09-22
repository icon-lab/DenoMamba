import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from model.denomamba_arch import DenoMamba
from options import TestOptions 
from data import create_loaders_seperate


def torchPSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    max_pixel = 1.0 
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))

normalize_matrix = lambda matrix: ((matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) * 255).astype(np.uint8)


opt = TestOptions().parse()


full_dose_path = opt.full_dose_path
quarter_dose_path = opt.quarter_dose_path
dataset_ratio = opt.dataset_ratio
train_ratio = opt.train_ratio
batch_size = opt.batch_size
in_ch = opt.in_ch
out_ch = opt.out_ch
dim = opt.dim
num_blocks = opt.num_blocks
num_refinement_blocks = opt.num_refinement_blocks
n_layer = opt.n_layer
ckpt_path = opt.ckpt_path
output_root = opt.output_root


mm_types = ["1mm", "3mm"]
mm_img_count = []
for mm_type in mm_types:
    print("MM TYPE :", mm_type)

    
    trainloader, validloader = create_loaders_seperate(
        full_dose_path=full_dose_path,
        quarter_dose_path=quarter_dose_path,
        dataset_ratio=dataset_ratio,
        train_ratio=train_ratio,
        batch_size=batch_size,
        mm_type=mm_type )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_model = DenoMamba(
        inp_channels=in_ch,
        out_channels=out_ch,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
    ).to(device)


    ckpt_model = torch.load(ckpt_path, map_location=device)
    net_weights = ckpt_model["net_model"]
    net_model.load_state_dict(net_weights)


    psnr_list = []
    ssim_list = []
    rmse_list = []
    net_model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(validloader)):
            condition = data.to(device)
            recon_train = net_model(condition)

            recon = recon_train.data.squeeze().cpu()
            img = target[0].data.squeeze().cpu()
            psnr_s = torchPSNR(img, recon)
            ssim_s = ssim(img.numpy(), recon.numpy(), data_range=np.max(img.numpy()) - np.min(img.numpy()))
            rmse_s = RMSELoss(img, recon)
            psnr_list.append(psnr_s)
            ssim_list.append(ssim_s)
            rmse_list.append(rmse_s)

            
            output_folder = os.path.join(output_root, mm_type)
            os.makedirs(output_folder, exist_ok=True)
            recon_image_path = os.path.join(output_folder, f"recon_{batch_idx}.png")
            img_image_path = os.path.join(output_folder, f"img_{batch_idx}.png")
            Image.fromarray((normalize_matrix(recon.numpy())).astype(np.uint8)).save(recon_image_path)
            Image.fromarray((img.numpy() * 255).astype(np.uint8)).save(img_image_path)

    print("DENOISED PSNR :", np.mean(psnr_list))
    print("DENOISED SSIM :", np.mean(ssim_list))
    print("DENOISED RMSE :", np.mean(rmse_list))



