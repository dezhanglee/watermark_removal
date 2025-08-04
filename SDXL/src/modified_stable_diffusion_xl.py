from typing import Callable, List, Optional, Union, Any, Dict, Tuple
import copy
import numpy as np
import PIL
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import logging, BaseOutput

logger = logging.get_logger(__name__)


class ModifiedStableDiffusionXLPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    init_latents: Optional[torch.FloatTensor]

class ModifiedStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        scheduler,
        force_zeros_for_empty_prompt=True,
        add_watermarker=None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker,
        )

    # @torch.no_grad()
    # def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None):
    #     (
    #         prompt_embeds,
    #         negative_prompt_embeds,
    #         pooled_prompt_embeds,
    #         negative_pooled_prompt_embeds,
    #     ) = super().encode_prompt(
    #         prompt=prompt,
    #         device=device,
    #         num_images_per_prompt=num_images_per_prompt,
    #         do_classifier_free_guidance=do_classifier_free_guidance,
    #         negative_prompt=negative_prompt,
    #     )
    #
    #     if do_classifier_free_guidance:
    #         return negative_prompt_embeds, prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    #     else:
    #         return prompt_embeds, pooled_prompt_embeds
    #
    # def __call__(
    #         self,
    #         prompt: Union[str, List[str]] = None,
    #         prompt_2: Optional[Union[str, List[str]]] = None,
    #         height: Optional[int] = None,
    #         width: Optional[int] = None,
    #         num_inference_steps: int = 50,
    #         denoising_end: Optional[float] = None,
    #         guidance_scale: float = 5.0,
    #         negative_prompt: Optional[Union[str, List[str]]] = None,
    #         negative_prompt_2: Optional[Union[str, List[str]]] = None,
    #         num_images_per_prompt: Optional[int] = 1,
    #         eta: float = 0.0,
    #         generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    #         latents: Optional[torch.FloatTensor] = None,
    #         prompt_embeds: Optional[torch.FloatTensor] = None,
    #         negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    #         pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    #         negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    #         output_type: Optional[str] = "pil",
    #         return_dict: bool = True,
    #         callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    #         callback_steps: int = 1,
    #         cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    #         guidance_rescale: float = 0.0,
    #         original_size: Optional[Tuple[int, int]] = None,
    #         crops_coords_top_left: Tuple[int, int] = (0, 0),
    #         target_size: Optional[Tuple[int, int]] = None,
    #         watermarking_gamma: float = None,
    #         watermarking_delta: float = None,
    #         watermarking_mask: Optional[torch.BoolTensor] = None,
    # ):
    #     height = height or self.default_sample_size * self.vae_scale_factor
    #     width = width or self.default_sample_size * self.vae_scale_factor
    #
    #     original_size = original_size or (height, width)
    #     target_size = target_size or (height, width)
    #
    #     self.check_inputs(
    #         prompt,
    #         prompt_2,
    #         height,
    #         width,
    #         callback_steps,
    #         negative_prompt,
    #         negative_prompt_2,
    #         prompt_embeds,
    #         negative_prompt_embeds,
    #         pooled_prompt_embeds,
    #         negative_pooled_prompt_embeds,
    #     )
    #
    #     # 2. Define call parameters
    #     if prompt is not None and isinstance(prompt, str):
    #         batch_size = 1
    #     elif prompt is not None and isinstance(prompt, list):
    #         batch_size = len(prompt)
    #     else:
    #         batch_size = prompt_embeds.shape[0]
    #
    #     device = self._execution_device
    #
    #     do_classifier_free_guidance = guidance_scale > 1.0
    #
    #     # 3. Encode input prompt
    #     text_encoder_lora_scale = (
    #         cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    #     )
    #     prompt_embeds = self._encode_prompt(
    #         prompt,
    #         device,
    #         num_images_per_prompt,
    #         do_classifier_free_guidance,
    #         negative_prompt,
    #         prompt_embeds=prompt_embeds,
    #         negative_prompt_embeds=negative_prompt_embeds,
    #         lora_scale=text_encoder_lora_scale,
    #     )
    #
    #     # 4. Prepare timesteps
    #     self.scheduler.set_timesteps(num_inference_steps, device=device)
    #     timesteps = self.scheduler.timesteps
    #
    #     # 5. Prepare latent variables
    #     num_channels_latents = self.unet.config.in_channels
    #     latents = self.prepare_latents(
    #         batch_size * num_images_per_prompt,
    #         num_channels_latents,
    #         height,
    #         width,
    #         prompt_embeds.dtype,
    #         device,
    #         generator,
    #         latents,
    #     )
    #
    #     init_latents = copy.deepcopy(latents)
    #
    #     if watermarking_gamma is not None:
    #         watermarking_mask = torch.rand(latents.shape, device=device) < watermarking_gamma
    #
    #     # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    #     extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    #
    #     # 7. Prepare added time ids & embeddings
    #     add_text_embeds = pooled_prompt_embeds
    #     add_time_ids = self._get_add_time_ids(
    #         original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
    #     )
    #
    #     if do_classifier_free_guidance:
    #         prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    #         add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
    #         add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    #
    #     prompt_embeds = prompt_embeds.to(device)
    #     add_text_embeds = add_text_embeds.to(device)
    #     add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
    #
    #     # 8. Denoising loop
    #     num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    #
    #     # 7.1 Apply denoising_end
    #     if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
    #         discrete_timestep_cutoff = int(
    #             round(
    #                 self.scheduler.config.num_train_timesteps
    #                 - (denoising_end * self.scheduler.config.num_train_timesteps)
    #             )
    #         )
    #         num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
    #         timesteps = timesteps[:num_inference_steps]
    #
    #     with self.progress_bar(total=num_inference_steps) as progress_bar:
    #         for i, t in enumerate(timesteps):
    #             if watermarking_mask is not None:
    #                 latents[watermarking_mask] += watermarking_delta * torch.sign(latents[watermarking_mask])
    #
    #             # expand the latents if we are doing classifier free guidance
    #             latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    #             latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    #
    #             # predict the noise residual
    #             added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    #             noise_pred = self.unet(
    #                 latent_model_input,
    #                 t,
    #                 encoder_hidden_states=prompt_embeds,
    #                 cross_attention_kwargs=cross_attention_kwargs,
    #                 added_cond_kwargs=added_cond_kwargs,
    #                 return_dict=False,
    #             )[0]
    #
    #             # perform guidance
    #             if do_classifier_free_guidance:
    #                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    #
    #             if do_classifier_free_guidance and guidance_rescale > 0.0:
    #                 # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
    #                 noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
    #
    #             # compute the previous noisy sample x_t -> x_t-1
    #             latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    #
    #             # call the callback, if provided
    #             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
    #                 progress_bar.update()
    #                 if callback is not None and i % callback_steps == 0:
    #                     callback(i, t, latents)
    #     # make sure the VAE is in float32 mode, as it overflows in float16
    #     if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
    #         self.upcast_vae()
    #         latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
    #
    #     image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
    #
    #     # apply watermark if available
    #     if self.watermark is not None:
    #         image = self.watermark.apply_watermark(image)
    #
    #     image = self.image_processor.postprocess(image, output_type=output_type)
    #
    #     # Offload last model to CPU
    #     if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
    #         self.final_offload_hook.offload()
    #
    #     if not return_dict:
    #         return (image,)
    #
    #     return ModifiedStableDiffusionXLPipelineOutput(images=image, init_latents=init_latents), latents

    # @torch.inference_mode()
    # def decode_image(self, latents: torch.FloatTensor, **kwargs):
    #     scaled_latents = 1 / 0.13025 * latents
    #     image = self.vae.decode(scaled_latents).sample
    #     return image
    #
    # @torch.inference_mode()
    # def torch_to_numpy(self, image):
    #     image = (image / 2 + 0.5).clamp(0, 1)
    #     image = image.cpu().permute(0, 2, 3, 1).numpy()
    #     return image
    #
    # @torch.inference_mode()
    # def get_image_latents(self, image, sample=True, rng_generator=None):
    #     encoding_dist = self.vae.encode(image).latent_dist
    #     encoding = encoding_dist.sample(generator=rng_generator) if sample else encoding_dist.mode()
    #     latents = encoding * 0.13025
    #     return latents
