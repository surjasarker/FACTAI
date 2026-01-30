import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from utils_flux import compute_score
import pdb
import math

# CW L2 attack
def cw_l2_attack(X, model, c=0.1, lr=0.01, iters=100, targeted=False):
    encoder = model.vae.encode
    clean_latents = encoder(X).latent_dist.mean

    def f(x):
        latents = encoder(x).latent_dist.mean
        if targeted:
            return latents.norm()
        else:
            return -torch.norm(latents - clean_latents.detach(), p=2, dim=-1)
    
    w = torch.zeros_like(X, requires_grad=True).cuda()
    pbar = tqdm(range(iters))
    optimizer = optim.Adam([w], lr=lr)

    for step in pbar:
        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, X)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2
        pbar.set_description(f"Loss: {cost.item():.5f} | loss1: {loss1.item():.5f} | loss2: {loss2.item():.5f}")
        # pdb.set_trace()

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
    X_adv = 1/2*(nn.Tanh()(w) + 1)
    return X_adv

# Encoder attack - Targeted / Untargeted
def encoder_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1, targeted=False):
    """
    Processing encoder attack using l_inf norm
    Params:
        X - image tensor we hope to protect
        model - the targeted edit model
        eps - attack budget
        step_size - attack step size
        iters - attack iterations
        clamp_min - min value for the image pixels
        clamp_max - max value for the image pixels
    Return:
        X_adv - image tensor for the protected image
    """
    encoder = model.vae.encode
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    if not targeted:
        loss_fn = nn.MSELoss()
        clean_latent = encoder(X).latent_dist.mean
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_(True)
        latent = encoder(X_adv).latent_dist.mean
        if targeted:
            loss = latent.norm()
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv - grad.detach().sign() * actual_step_size
        else:
            loss = loss_fn(latent, clean_latent)
            grad, = torch.autograd.grad(loss, [X_adv])
            X_adv = X_adv + grad.detach().sign() * actual_step_size

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

        pbar.set_postfix(norm_2=(X_adv - X).norm().item(), norm_inf=(X_adv - X).abs().max().item())

    return X_adv

def vae_attack(X, model, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    """
    Processing encoder attack using l_inf norm
    Params:
        X - image tensor we hope to protect
        model - the targeted edit model
        eps - attack budget
        step_size - attack step size
        iters - attack iterations
        clamp_min - min value for the image pixels
        clamp_max - max value for the image pixels
    Return:
        X_adv - image tensor for the protected image
    """
    vae = model.vae
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).half().cuda(), min=clamp_min, max=clamp_max)
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i

        X_adv.requires_grad_()
        image = vae(X_adv).sample

        loss = (image).norm()
        grad, = torch.autograd.grad(loss, [X_adv])
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

    return X_adv
    
import torch
import torch.nn.functional as F

    
def dct_matrix(N=8, device="cuda"):
    mat = torch.zeros(N, N, device=device)
    pi = torch.tensor(torch.pi, device=device)
    for k in range(N):
        for n in range(N):
            mat[k,n] = torch.cos((2*n+1)*k*pi/(2*N))
    mat[0] /= torch.sqrt(torch.tensor(2.0, device=device))
    return mat * torch.sqrt(torch.tensor(2.0/N, device=device))


def block_dct(x, dct):
    B,C,H,W = x.shape
    x = x.unfold(2,8,8).unfold(3,8,8)
    x = x.contiguous().view(-1,8,8)
    X = dct @ x @ dct.t()
    Bc = B
    Cc = C
    Hb = H//8
    Wb = W//8
    return X.view(Bc,Cc,Hb,Wb,8,8)

def block_idct(X, dct):
    B,C,Hb,Wb,_,_ = X.shape
    X = X.view(-1,8,8)
    x = dct.t() @ X @ dct
    x = x.view(B,C,Hb,Wb,8,8)
    return x.permute(0,1,2,4,3,5).reshape(B,C,Hb*8,Wb*8)

def soft_round(x):
    return torch.round(x) + (x - torch.round(x))**3

# Quantization table
Q_LUMA = torch.tensor([
 [16,11,10,16,24,40,51,61],
 [12,12,14,19,26,58,60,55],
 [14,13,16,24,40,57,69,56],
 [14,17,22,29,51,87,80,62],
 [18,22,37,56,68,109,103,77],
 [24,35,55,64,81,104,113,92],
 [49,64,78,87,103,121,120,101],
 [72,92,95,98,112,100,103,99]
], device="cuda")

def scale_q(Q, q):
    s = 5000/q if q < 50 else 200 - 2*q
    return (Q * s / 100).clamp(min=1)

def rgb_to_ycbcr(x):
    x = (x + 1)/2
    M = x.new_tensor([[0.299,0.587,0.114],
                      [-0.1687,-0.3313,0.5],
                      [0.5,-0.4187,-0.0813]])
    b = x.new_tensor([0,0.5,0.5])
    y = torch.einsum("bchw,dc->bdhw", x, M) + b.view(1,3,1,1)
    return y

def ycbcr_to_rgb(y):
    M = y.new_tensor([[1.0,0.0,1.402],
                      [1.0,-0.34414,-0.71414],
                      [1.0,1.772,0.0]])
    b = y.new_tensor([0,-0.5,-0.5])
    x = torch.einsum("bchw,dc->bdhw", y + b.view(1,3,1,1), M)
    return (x*2 - 1).clamp(-1,1)

def chroma_subsample(ycbcr):
    Y  = ycbcr[:,0:1]
    Cb = F.avg_pool2d(ycbcr[:,1:2], 2,2)
    Cr = F.avg_pool2d(ycbcr[:,2:3], 2,2)
    return Y, Cb, Cr

def jpeg_diff(x, q):
    dct = dct_matrix(device=x.device)
    ycbcr = rgb_to_ycbcr(x)
    Y, Cb, Cr = chroma_subsample(ycbcr)

    QY = scale_q(Q_LUMA, q)
    QC = QY

    Yd  = block_dct(Y, dct)
    Cbd = block_dct(Cb, dct)
    Crd = block_dct(Cr, dct)

    Yq  = soft_round(Yd / QY) * QY
    Cbq = soft_round(Cbd / QC) * QC
    Crq = soft_round(Crd / QC) * QC

    Yr  = block_idct(Yq, dct)
    Cbr = F.interpolate(block_idct(Cbq, dct), scale_factor=2, mode="nearest")
    Crr = F.interpolate(block_idct(Crq, dct), scale_factor=2, mode="nearest")

    ycbcr_rec = torch.cat([Yr, Cbr, Crr], dim=1)
    return ycbcr_to_rgb(ycbcr_rec)


def jpeg_ensemble_grad(X_adv, X_clean, qualities, loss_fn):
    grad_ensemble = torch.zeros_like(X_adv)
    losses_dict = None

    for q in qualities:
        Xq = jpeg_diff(X_adv, q)
        loss_dict = loss_fn(Xq, X_clean)
        total_loss = loss_dict["total"]

        # Compute gradient for this quality and add
        grad = torch.autograd.grad(total_loss, X_adv)[0]
        grad_ensemble += grad

        # Save the last loss_dict for logging
        losses_dict = loss_dict

        # Free memory
        del Xq, total_loss, grad
        torch.cuda.empty_cache()

    grad_ensemble /= len(qualities)  # average
    return grad_ensemble, losses_dict



def facelock(X, model, aligner, fr_model, lpips_fn, eps=0.03, step_size=0.01, iters=100, clamp_min=-1, clamp_max=1):
    X_adv = torch.clamp(X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).to(X.device), min=clamp_min, max=clamp_max).half()
    
    ###
    qualities = [30,50,70,90]  # ensemble of JPEG qualities
    pbar = tqdm(range(iters))
    
    vae = model.vae
    X_adv.requires_grad_(True)
    clean_latent = vae.encode(X).latent_dist.mean

    for i in pbar:
        # actual_step_size = step_size
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        
        latent = vae.encode(X_adv).latent_dist.mean
        image = vae.decode(latent).sample.clip(-1, 1)
        
        #########
        def loss_fn(X_input, X_ref):
            latent = vae.encode(X_input).latent_dist.mean
            image = vae.decode(latent).sample.clip(-1,1)
        
            loss_cvl = compute_score(image.float(), X_ref.float(), aligner, fr_model)
            loss_encoder = F.mse_loss(latent, clean_latent)
            loss_lpips = lpips_fn(image, X_ref)
            total = -loss_cvl * (1 if i >= iters * 0.35 else 0.0) + loss_encoder * 0.2 + loss_lpips * (1 if i > iters * 0.25 else 0.0)
        
            return {"total": total, "loss_cvl": loss_cvl, "loss_encoder": loss_encoder, "loss_lpips": loss_lpips}

        
        
        grad, losses_dict = jpeg_ensemble_grad(X_adv, X, qualities, loss_fn)
        X_adv = X_adv + actual_step_size * grad.sign()
        
        pbar.set_postfix(
            loss_cvl=losses_dict["loss_cvl"].item(),
            loss_encoder=losses_dict["loss_encoder"].item(),
            loss_lpips=losses_dict["loss_lpips"].item(),
            loss=losses_dict["total"].item()
        )


        ##############

        #loss_cvl = compute_score(image.float(), X.float(), aligner=aligner, fr_model=fr_model)
        #loss_encoder = F.mse_loss(latent, clean_latent)
        #loss_lpips = lpips_fn(image, X)
        #loss = -loss_cvl * (1 if i >= iters * 0.35 else 0.0) + loss_encoder * 0.2 + loss_lpips * (1 if i > iters * 0.25 else 0.0)
        #loss = -loss_cvl * 20 + loss_lpips * 1
        #grad, = torch.autograd.grad(loss, [X_adv])
        #X_adv = X_adv + grad.detach().sign() * actual_step_size

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

    return X_adv
