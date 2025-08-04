import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

# from src.modified_stable_diffusion_xl import ModifiedStableDiffusionXLPipeline
from src.inverse_stable_diffusion_xl import InversableStableDiffusionXLPipeline
from src.optim_utils import set_random_seed, transform_img, get_dataset


def stable_diffusion_xl_pipe(
        solver_order=1,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        cache_dir='/home/xuandong/mnt/hf_models',
):
    # load stable diffusion pipeline
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler(
        beta_end=0.012,
        beta_schedule='scaled_linear',
        beta_start=0.00085,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        steps_offset=1,
        trained_betas=None,
        solver_order=solver_order,
    )

    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.float32,
    #     variant="fp32",
    #     use_safetensors=True
    # )

    # pipe = ModifiedStableDiffusionXLPipeline.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.float32,
    #     variant="fp32",
    #     use_safetensors=True,
    # )

    pipe = InversableStableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        variant="fp32",
        use_safetensors=True,
    )

    pipe = pipe.to(device)
    pipe.scheduler = scheduler

    return pipe


def generate(
        image_num=0,
        prompt=None,
        guidance_scale=3.0,
        num_inference_steps=50,
        solver_order=1,
        image_length=512,
        datasets='Gustavosta/Stable-Diffusion-Prompts',
        model_id='stabilityai/stable-diffusion-2-1-base',
        gen_seed=0,
        pipe=None,
        init_latents=None,
):
    # load stable diffusion pipeline
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    if pipe is None:
        scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset=1,
            trained_betas=None,
            solver_order=solver_order,
        )
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float32,
        )
    pipe = pipe.to(device)

    # load dataset and prompt
    if prompt is None:
        dataset, prompt_key = get_dataset(datasets)
        prompt = dataset[image_num][prompt_key]

    # generate init latent
    seed = gen_seed + image_num
    set_random_seed(seed)

    if init_latents is None:
        init_latents = pipe.get_random_latents()

    # generate image
    output, _ = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=image_length,
        width=image_length,
        latents=init_latents,
    )
    image = output.images[0]

    return image, prompt, init_latents


def exact_inversion(
        image,
        prompt='',
        guidance_scale=3.0,
        num_inference_steps=50,
        solver_order=1,
        test_num_inference_steps=50,
        inv_order=1,
        decoder_inv=True,
        model_id='stabilityai/stable-diffusion-xl-base-1.0',
        pipe=None,
        height=1024,
        width=1024,
):
    # load stable diffusion pipeline
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    if pipe is None:
        scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule='scaled_linear',
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset=1,
            trained_betas=None,
            solver_order=solver_order,
        )

        pipe = InversableStableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            variant="fp32",
            use_safetensors=True,
        )

        pipe.scheduler = scheduler

    pipe = pipe.to(device)

    do_classifier_free_guidance = guidance_scale > 1.0

    # Encode prompt & time embeddings
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=None,
    )

    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)

    # Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = pipe._get_add_time_ids(
        original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # image to latent
    image = transform_img(image).unsqueeze(0).to(prompt_embeds.dtype).to(device)
    if decoder_inv:
        image_latents = pipe.decoder_inv(image)
    else:
        image_latents = pipe.get_image_latents(image, sample=False)

    # forward diffusion : image to noise
    reversed_latents = pipe.forward_diffusion(
        latents=image_latents,
        text_embeddings=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        guidance_scale=guidance_scale,
        num_inference_steps=test_num_inference_steps,
        inverse_opt=(inv_order != 0),
        inv_order=inv_order
    )

    return reversed_latents