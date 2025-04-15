import os
import argparse
import torch
import pickle
import json
from tqdm import tqdm
import random
import numpy as np
from datasets import load_dataset
from src.prc import KeyGen, Encode, str_to_bin, bin_to_str
import src.pseudogaussians as prc_gaussians
from src.baseline.gs_watermark import Gaussian_Shading_chacha
from src.baseline.treering_watermark import tr_detect, tr_get_noise
from inversion import stable_diffusion_pipe, generate, exact_inversion
from attacks import *
from src.prc import Detect, Decode
import cv2
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

from torchvision import transforms as tfms


parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=10)
parser.add_argument('--method', type=str, default='prc') # gs, tr, prc
parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts') # coco 
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.1)
parser.add_argument('--prc_t', type=int, default=3)
parser.add_argument('--attack', type=str, default='stealthy') #white_noise, min_distortion, stealthy
parser.add_argument('--eps', type=float, default=5)
parser.add_argument('--device', type=str, default="cuda:1")
parser.add_argument('--message_length', type=int, default=1500)
parser.add_argument('--inversion', type=str, default='exact') # 'exact', 'null', 'prompt'

args = parser.parse_args()
print(args)

hf_cache_dir = 'hf_models'
device = args.device
n = 4 * 64 * 64  # the length of a PRC codeword
method = args.method
test_num = args.test_num
model_id = args.model_id
dataset_id = args.dataset_id
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t
eps = args.eps
attack = args.attack
exp_id = f'{method}_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{nowm}_eps_{eps}_attack_{attack}_mess_len_{args.message_length}_inversion_{args.inversion}'

## Inversion
@torch.no_grad()
def invert(
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)
# Sample function (regular DDIM)
@torch.no_grad()
def sample(prompt, start_step=0, start_latents=None,
           guidance_scale=3.5, num_inference_steps=30,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input.type(torch.float32), t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
        latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.decode_latents(latents.type(torch.float32))
    images = pipe.numpy_to_pil(images)

    return images

def add_white_noise(input_latent, eps):
    wn = torch.normal(0, 1, input_latent.shape)
    wn_normalized = wn / (torch.sum(wn**2)**0.5) * (eps)
    dev = input_latent.get_device()
    out = input_latent + wn_normalized.to(dev)
    mean, std = torch.mean(out), torch.std(out)
    out = (out-mean)/std
    return out

def add_stealthy_attack(input_latent, eps):
    input_latent_flat = input_latent.flatten().detach().clone().to(device)
    input_latent_flat_sorted = torch.sort((2*input_latent_flat)**2)
    sort_val = input_latent_flat_sorted.values
    sort_idx = input_latent_flat_sorted.indices
    # get squared sum
    
    # get largest k
    curr_eps_sq = 0
    idx = 0
    while curr_eps_sq < eps**2 and idx < len(sort_idx):
        curr_idx = sort_idx[idx]
        input_latent_flat[curr_idx] *= -1.0
        curr_eps_sq += (2*input_latent_flat[curr_idx])**2
        idx += 1
#         print(input_latent_flat[i])
    return input_latent_flat.reshape(input_latent.shape)

def add_clustering_attack(input_latent, eps):
    input_latent_flat = input_latent.flatten().detach().clone()
    curr_k = 0
    curr_l2_sum = 0
    while curr_k < len(input_latent_flat):
        if curr_l2_sum + (2*input_latent_flat[curr_k])**2 < eps**2:
            curr_k += 1
            curr_l2_sum += (2*input_latent_flat[curr_k])**2 
        else:
            break

    for i in range(curr_k+1):
        input_latent_flat[i] *= -1
    return input_latent_flat.reshape(input_latent.shape)
        
def add_min_distortion_attack(input_latent, eps):
    input_latent_flat = input_latent.flatten().detach().clone()
    input_latent_flat_sorted = torch.sort((input_latent_flat)**2)
#     print(input_latent_flat_sorted.values[64*64])
    sort_val = input_latent_flat_sorted.values
    print(sort_val[2*64*64])
    sort_idx = input_latent_flat_sorted.indices
    # get squared sum
    
    
    
    # get largest k
    curr_eps_sq = 0
    idx = 0
    const = 1e-4
    while curr_eps_sq < eps**2 and idx < len(sort_idx):
        if abs(input_latent_flat[idx]) < const:
            input_latent_flat[idx] *= -1.0
            curr_eps_sq += (2*input_latent_flat[idx])**2
        else:
            input_latent_flat[idx] /= abs(input_latent_flat[idx])
            input_latent_flat[idx] *= const
            curr_eps_sq += (abs(input_latent_flat[idx]) + const)**2
#         print(input_latent_flat[i])
        idx += 1
    return input_latent_flat.reshape(input_latent.shape)

if method == 'prc':
    if not os.path.exists(f'keys/{exp_id}.pkl'):  # Generate watermark key for the first time and save it to a file
        (encoding_key_ori, decoding_key_ori) = KeyGen(n, message_length=args.message_length, 
                                                      false_positive_rate=fpr, t=prc_t)  # Sample PRC keys
        with open(f'keys/{exp_id}.pkl', 'wb') as f:  # Save the keys to a file
            pickle.dump((encoding_key_ori, decoding_key_ori), f)
        with open(f'keys/{exp_id}.pkl', 'rb') as f:  # Load the keys from a file
            encoding_key, decoding_key = pickle.load(f)
        assert encoding_key[0].all() == encoding_key_ori[0].all()
    else:  # Or we can just load the keys from a file
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            encoding_key, decoding_key = pickle.load(f)
        print(f'Loaded PRC keys from file keys/{exp_id}.pkl')
elif method == 'gs':
    gs_watermark = Gaussian_Shading_chacha(ch_factor=1, hw_factor=8, fpr=fpr, user_number=10000)
    if not os.path.exists(f'keys/{exp_id}.pkl'):
        watermark_m_ori, key_ori, nonce_ori, watermark_ori = gs_watermark.create_watermark_and_return_w()
        with open(f'keys/{exp_id}.pkl', 'wb') as f:
            pickle.dump((watermark_m_ori, key_ori, nonce_ori, watermark_ori), f)
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            watermark_m, key, nonce, watermark = pickle.load(f)
        assert watermark_m.all() == watermark_m_ori.all()
    else:  # Or we can just load the keys from a file
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            watermark_m, key, nonce, watermark = pickle.load(f)
            print(f'Loaded GS keys from file keys/{exp_id}.pkl')
elif method == 'tr':
    # need to generate watermark key for the first time then save it to a file, we just load previous key here
    tr_key = '7c3fa99795fe2a0311b3d8c0b283c5509ac849e7f5ec7b3768ca60be8c080fd9_0_10_rand'
    # tr_key = '4145007d1cbd5c3e28876dd866bc278e0023b41eb7af2c6f9b5c4a326cb71f51_0_9_rand'
    print('Loaded TR keys from file')
else:
    raise NotImplementedError

if dataset_id == 'coco':
    save_folder = f'./results/{exp_id}_coco/original_images'
else:
    save_folder = f'./results/{exp_id}/original_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
print(f'Saving original images to {save_folder}')

random.seed(42)
if dataset_id == 'coco':
    with open('coco/captions_val2017.json') as f:
        all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
else:
    all_prompts = [sample['Prompt'] for sample in load_dataset(dataset_id)['test']]

prompts = random.sample(all_prompts, test_num)

pipe = stable_diffusion_pipe(solver_order=1, model_id=model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)

def seed_everything(seed, workers=False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed
cur_inv_order = 0
# for i in tqdm(range(2)):
for i in tqdm(range(test_num)):
    seed_everything(i)
    current_prompt = prompts[i]
    if nowm:
        init_latents_np = np.random.randn(1, 4, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float64).to(device)
    else:
        if method == 'prc':
            prc_codeword = Encode(encoding_key)
            init_latents = prc_gaussians.sample(prc_codeword).reshape(1, 4, 64, 64).to(device)
        elif method == 'gs':
            init_latents = gs_watermark.truncSampling(watermark_m)
        elif method == 'tr':
            shape = (1, 4, 64, 64)
            init_latents, _, _ = tr_get_noise(shape, from_file=tr_key, keys_path='keys/')
        else:
            raise NotImplementedError
    
#     orig_image, _, _ = generate(prompt=current_prompt,
#                                 init_latents=init_latents,
#                                 num_inference_steps=args.inf_steps,
#                                 solver_order=1,
#                                 pipe=pipe
#                                 )
    orig_image=sample(prompt=current_prompt, start_latents=init_latents, num_inference_steps=args.inf_steps,
                       )[0]
    orig_image.save(f'{save_folder}/{i}.png')
    img_1 = img_as_float(orig_image)
    img_1 = np.squeeze(img_1)
    seed_everything(i)
#     reversed_latents = exact_inversion(orig_image,
#                                        prompt="",
#                                        test_num_inference_steps=args.inf_steps,
#                                        inv_order=cur_inv_order,
#                                        pipe=pipe
#                                        )
    start_step = 30
    
#     convert_tensor = tfms.ToTensor()

    with torch.no_grad(): 
        latent = pipe.vae.encode(tfms.functional.to_tensor(orig_image).unsqueeze(0).to(device)*2-1)
        lat = 0.18215 * latent.latent_dist.sample()
    if args.inversion == 'exact':
        reversed_latents = init_latents
    elif args.inversion == 'null':
        inverted_latents = invert(lat, "", num_inference_steps=50)
        inverted_latents.shape
        reversed_latents =inverted_latents[-(start_step + 1)][None]
#         reversed_latents = exact_inversion(orig_image,
#                                        prompt="",
#                                        test_num_inference_steps=args.inf_steps,
#                                        inv_order=cur_inv_order,
#                                        pipe=pipe
#                                        )
    elif args.inversion == 'prompt':
        inverted_latents = invert(lat, current_prompt, num_inference_steps=50)
        reversed_latents = inverted_latents[-(start_step + 1)][None]
#         reversed_latents = exact_inversion(orig_image,
#                                        prompt=current_prompt,
#                                        test_num_inference_steps=args.inf_steps,
#                                        inv_order=cur_inv_order,
#                                        pipe=pipe
#                                        )
    print(torch.sum((reversed_latents - init_latents)**2))
    if attack == 'white_noise':
        reversed_latents_attack = add_white_noise(reversed_latents, eps)
    elif attack == 'stealthy':
        reversed_latents_attack = add_stealthy_attack(reversed_latents, eps)
    elif attack == 'min_distortion':
        reversed_latents_attack = add_min_distortion_attack(reversed_latents, eps)
    elif attack == 'clustering':
        reversed_latents_attack = add_clustering_attack(reversed_latents, eps)
    print(torch.sum((reversed_latents - reversed_latents_attack)**2)**0.5)
    if args.inversion == 'null':
#         orig_image, _, _ = generate(prompt="",
#                             init_latents=reversed_latents_attack,
#                             num_inference_steps=args.inf_steps,
#                             solver_order=1,
#                             pipe=pipe
#                             )
        orig_image=sample("", start_latents=reversed_latents_attack,
                          start_step=start_step, num_inference_steps=args.inf_steps)[0]
    else:
        
#         orig_image, _, _ = generate(prompt=current_prompt,
#                                 init_latents=reversed_latents_attack,
#                                 num_inference_steps=args.inf_steps,
#                                 solver_order=1,
#                                 pipe=pipe
#                                 )
        orig_image=sample(prompt=current_prompt, start_latents=reversed_latents_attack, num_inference_steps=args.inf_steps,
                           )[0]
#         orig_image=sample(current_prompt, start_latents=reversed_latents_attack,
#                           start_step=start_step, num_inference_steps=args.inf_steps)[0]
    img_2 = img_as_float(orig_image)
    img_2 = np.squeeze(img_2)
    mse_const = mean_squared_error(img_1, img_2)
#     print(img_1.shape)
    ssim_const = ssim(img_1, img_2, data_range=img_2.max() - img_2.min(),channel_axis=2)
    print(f'mse_{mse_const}_ssim_{ssim_const}')
    
    
        # test watermark 
    reversed_latents = exact_inversion(orig_image,
                                       prompt="",
                                       test_num_inference_steps=args.inf_steps,
                                       inv_order=cur_inv_order,
                                       pipe=pipe
                                       )
    

    if method == 'prc':
        reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(torch.float64).flatten().cpu(), variances=float(1.5)).flatten().cpu()
        detection_result = Detect(decoding_key, reversed_prc)
        decoding_result = (Decode(decoding_key, reversed_prc) is not None)
        combined_result = detection_result and decoding_result
#         combined_results.append(combined_result)
        print(f'{i:03d}: Detection: {detection_result}; Decoding: {decoding_result}; Combined: {combined_result}')
    elif method == 'gs':
        gs_watermark = Gaussian_Shading_chacha(ch_factor=1, hw_factor=8, fpr=fpr, user_number=10000)
        
        gs_watermark.nonce=nonce
        gs_watermark.key=key
        gs_watermark.watermark=watermark
        
        acc_metric = gs_watermark.eval_watermark(reversed_latents)
#         combined_results.append(combined_result)
        print(acc_metric)
    is_removed=not combined_result
    orig_image.save(f'{save_folder}/{i}_remove_{attack}_mse_{mse_const}_ssim_{ssim_const}_removed_{is_removed}.png')
print(f'Done generating {method} images')