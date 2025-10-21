import os
import logging
import time
import glob
import SimpleITK as sitk
import json
import random

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from functions.denoising import *
from functions.calc_fid import *
from datasets import inverse_data_transform
from datasets.AtoB import *
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu

import torch.nn.functional as F
from generative.networks.nets import DiffusionModelUNet
from monai.apps import DecathlonDataset
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, ToTensord
from monai.utils import set_determinism

def scaling(x, old_min, old_max, new_min, new_max):
    # x = torch.clamp(x, min=old_min, max=old_max)
    return ((x - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

def random_crop(tensor, crop_size=(64, 512, 512)):
    batch_size, channels, depth, height, width = tensor.shape
    crop_depth, crop_height, crop_width = crop_size

    z_start = torch.randint(0, depth - crop_depth + 1, (1,)).item()
    y_start = torch.randint(0, height - crop_height + 1, (1,)).item()
    x_start = torch.randint(0, width - crop_width + 1, (1,)).item()

    cropped_tensor = tensor[
        :, 
        :, 
        z_start:z_start + crop_depth, 
        y_start:y_start + crop_height, 
        x_start:x_start + crop_width
    ]

    return cropped_tensor

def select_random_axis(tensor, num_axis=8):
    start_index = random.randint(0, tensor.size(1) - num_axis)
    return tensor[:, start_index:start_index + num_axis, :, :]

def cosine_similarity(x1, x2):
    # 텐서를 1D 벡터로 평탄화
    x1_flat = x1.view(-1)
    x2_flat = x2.view(-1)
    
    # 코사인 유사도 계산
    similarity = F.cosine_similarity(x1_flat.unsqueeze(0), x2_flat.unsqueeze(0))
    
    return similarity.item()


def compute_alpha(beta, t, device):
    beta = torch.cat([torch.zeros(1).to(device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def resample_img(itk_image, out_size, is_label=False):
    # resample images to specified size with simple itk

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_spacing = [
        original_spacing[0] * (original_size[0] / out_size[0]),
        original_spacing[1] * (original_size[1] / out_size[1]),
        original_spacing[2] * (original_size[2] / out_size[2])
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def rescale(arr, src_min, src_max, tar_min, tar_max):
    rescaled_arr = (arr - src_min) / (src_max - src_min) * (tar_max - tar_min) + tar_min
    return rescaled_arr


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        if args.select_optimal_step == True or args.test == True or args.calc_target_domain_fid == True:
            pass
        else:
            self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if args.select_optimal_step == True or args.test == True or args.calc_target_domain_fid == True:
            pass
        else:
            if self.model_var_type == "fixedlarge":
                self.logvar = betas.log()
                # torch.cat(
                # [posterior_variance[1:2], betas[1:]], dim=0).log()
            elif self.model_var_type == "fixedsmall":
                self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        
        data_dir = config.data.model_dir_path
        image_files = sorted(glob(f'{data_dir}/*.nii.gz'))
        
        set_determinism(seed=0)
        
        transforms = Compose([
            LoadImaged(keys=["image"]),
            ToTensord(keys=["image"])
        ])
        
        # Create dataset and dataloader
        dataset = Dataset(data=[{"image": img} for img in image_files], transform=transforms)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.data.num_workers)
    

        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.config.model.resume_training:
            states = torch.load(os.path.join(self.config.model.log_path))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
        # src_batchs = []
        # tar_batchs = []
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            # select_random_axis(tensor, num_axis=config.model.out_ch)
            for i, datas in enumerate(train_loader):
                images = datas['image']
                images = scaling(images, config.data.scaling[0], config.data.scaling[1], config.data.scaling[2], config.data.scaling[3])
                src = images[:,:,:,:,0,0].unsqueeze(1).permute(0, 4, 1, 3, 2)[0]
                tar = images[:,:,:,:,0,1].unsqueeze(1).permute(0, 4, 1, 3, 2)[0]
                
                for z_axis in range(0, src.shape[0], config.training.batch_size):
                    src_2d = src[z_axis:z_axis+config.training.batch_size, :, :, :]
                    tar_2d = tar[z_axis:z_axis+config.training.batch_size, :, :, :]
                    x_c = src_2d.float().to(self.device)
                    x = tar_2d.float().to(self.device)       
          
                    n = x.size(0)
                    data_time += time.time() - data_start
                    model.train()
                    step += 1
                    e = torch.randn_like(x).float() 
                    b = self.betas
                    
                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    loss = loss_registry[config.model.type](model, x, x_c, t, e, b)
                    
                    tb_logger.add_scalar("loss", loss, global_step=step)

                    logging.info(
                        f"step: {step}, noise_loss: {loss.item()}, data time: {data_time / (i+1)}, t: {t[0].item()}, epoch: {epoch}"
                    )

                    optimizer.zero_grad()
                    loss.backward()

                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()

                    if self.config.model.ema:
                        ema_helper.update(model)

                    data_start = time.time()

            # Save model state after each epoch
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())

            torch.save(
                states,
                os.path.join(self.args.log_path, f"ckpt_{epoch}.pth"),
            )
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))   
                

               
            
            
    

    def test(self):
        args, config = self.args, self.config
        
        dataset = test(args.input_folder, config.data.scaling, config.data.clip, config.data.image_size)
        test_loader = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        
        states = torch.load(args.test_model_path, map_location=self.device)
        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        
        axis_ls = []
        for or_size, or_spacing, or_direction, or_origin, path, x in test_loader:
            or_size = tuple(t.item() for t in or_size)
            or_spacing = tuple(t.item() for t in or_spacing)
            or_direction = tuple(t.item() for t in or_direction)
            or_origin = tuple(t.item() for t in or_origin)
            model.eval()

            x = x.to(self.device).float()
            x_c = x.to(self.device).float()
            e = torch.randn_like(x) 
            b = self.betas
            seq = sorted(args.step)
            xs, _ = generalized_steps_condition(x, x_c, e, seq, model, b)
            
            arr = xs[-1].cpu().numpy()
            arr = arr.reshape((1, arr.shape[2], arr.shape[3]))
            
            axis_ls.append(arr)
            if len(axis_ls) == or_size[2]:
                arr_3d = np.concatenate(axis_ls, axis=0)
                
                arr_3d = rescale(arr_3d, config.data.scaling[2], config.data.scaling[3], config.data.scaling[0], config.data.scaling[1])
                axis_ls = []
                arr_3d = sitk.GetImageFromArray(arr_3d)
                arr_3d = resample_img(arr_3d, or_size, is_label=False)
                arr_3d.SetSpacing(or_spacing)
                arr_3d.SetDirection(or_direction)
                arr_3d.SetOrigin(or_origin)
                os.makedirs(args.output_folder, exist_ok=True)
                sitk.WriteImage(arr_3d, f"{args.output_folder}/{os.path.basename(path[0])}")
                print(f"{args.output_folder}/{os.path.basename(path[0])}")                

    def select_optimal_step(self):
        args, config = self.args, self.config
        
        dataset1 = test(args.input_folder)
        dataset2 = test(args.input_folder2)
        
        data_loader_1 = data.DataLoader(
            dataset1,
            batch_size=1,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        
        data_loader_2 = data.DataLoader(
            dataset2,
            batch_size=1,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        
        seq = sorted(args.step)
        fids = []
        ts = []
        for t_ in seq:
            x1ts = []
            x2ts = []
            for idx, ((_, _, _, _, _, x1), (_, _, _, _, _, x2)) in enumerate(zip(data_loader_1, data_loader_2)):
                n = x1.size(0)
                x1 = x1
                x2 = x2
                e = torch.randn_like(x1) 
                b = self.betas.to('cpu')               
                t = torch.tensor(t_)
                at = compute_alpha(b, t.long(), 'cpu')
                x1t = x1 * at.sqrt() + e * (1.0 - at).sqrt()
                x1t = torch.cat([x1t, x1t, x1t], dim=1)
                x2t = x2 * at.sqrt() + e * (1.0 - at).sqrt()
                x2t = torch.cat([x2t, x2t, x2t], dim=1)
                x1ts.append(x1t)
                x2ts.append(x2t)
            total_x1t = torch.cat(x1ts, dim=0)
            total_x2t = torch.cat(x2ts, dim=0)
            if total_x1t.shape[0] > total_x2t.shape[0]:
                total_x1t = total_x1t[:total_x2t.shape[0]]
            else:
                total_x2t = total_x2t[:total_x1t.shape[0]]
            fid = calculate_fid(total_x1t, total_x2t, 50, self.device, 2048)
            fids.append(fid)
            ts.append(t_)
            print(f"t: {t_}, fid: {fid}")
            
        import pandas as pd
        pd.DataFrame({'t': ts, 'fid': fids}).to_excel(f"{args.output_folder}/fidscores_over_t.xlsx")  

    def calc_target_domain_fid(self):
        args, config = self.args, self.config
        
        dataset1 = test(args.input_folder)
        
        data_loader_1 = data.DataLoader(
            dataset1,
            batch_size=1,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        
        seq = sorted(args.step)
        fids = []
        ts = []
        for t_ in seq:
            x1ts = []
            x2ts = []
            for idx, (_, _, _, _, _, x1) in enumerate(data_loader_1):
                n = x1.size(0)
                x1 = x1
                e = torch.randn_like(x1) 
                b = self.betas.to('cpu')               
                t = torch.tensor(t_)
                at = compute_alpha(b, t.long(), 'cpu')
                x1t = x1 * at.sqrt() + e * (1.0 - at).sqrt()
                x1t = torch.cat([x1t, x1t, x1t], dim=1)
                x1ts.append(x1t)
            total_x1t = torch.cat(x1ts, dim=0)
            
            total_x2t = total_x1t[1::2, :, :, :]
            total_x1t = total_x1t[0::2, :, :, :]
            if total_x1t.shape[0] > total_x2t.shape[0]:
                total_x1t = total_x1t[:total_x2t.shape[0]]
            else:
                total_x2t = total_x2t[:total_x1t.shape[0]]
            fid = calculate_fid(total_x1t, total_x2t, 50, self.device, 2048)
            fids.append(fid)
            ts.append(t_)
            print(f"t: {t_}, fid: {fid}")
            
        import pandas as pd
        pd.DataFrame({'t': ts, 'fid': fids}).to_excel(f"{args.output_folder}/fidscores_over_t.xlsx")  
