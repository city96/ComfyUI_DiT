import os
import torch
import folder_paths
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import comfy.latent_formats

from .models import DiT_models
from .diffusion import create_diffusion

# load these from separate folder
folder_paths.folder_names_and_paths["dit"] = (
	[os.path.join(folder_paths.models_dir,"dit")], 
	folder_paths.supported_pt_extensions
)

class DiTCheckpointLoader:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"ckpt_name": (folder_paths.get_filename_list("dit"),),
				"model": (list(DiT_models.keys()),),
				"image_size": ([256, 512],),
				"num_classes": ("INT", {"default": 1000, "min": 0,}),
			}
		}
	RETURN_TYPES = ("DIT",) # could be MODEL if it is made compatible?
	FUNCTION = "load_checkpoint"
	CATEGORY = "DiT"
	TITLE = "DiTCheckpointLoader"

	def load_checkpoint(self, ckpt_name, model, image_size, num_classes):
		# note: switch to custom comfy.model_base eventually
		model = DiT_models[model](
			input_size=image_size // 8, # latent size
			num_classes=num_classes
		)

		ckpt_path = folder_paths.get_full_path("dit", ckpt_name)
		state_dict = comfy.utils.load_torch_file(ckpt_path)
		model.load_state_dict(state_dict)
		model.eval() # important, apparently
		
		# need these later anyway
		model.latent_format = comfy.latent_formats.SD15()
		model.latent_size = image_size // 8
		model.num_classes = num_classes

		# I didn't expect this to work but it looks like it does.
		model_patcher = comfy.model_patcher.ModelPatcher(
			model,
			load_device=comfy.model_management.get_torch_device(),
			offload_device=comfy.model_management.unet_offload_device(),
			current_device="cpu"
		)

		# return (model,)
		return (model_patcher,)

class DiTSampler:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"model": ("DIT",),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
				"batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
				"class_labels": ([207,],),
			}
		}
	RETURN_TYPES = ("LATENT",)
	FUNCTION = "sample"
	CATEGORY = "DiT"
	TITLE = "DiTSampler"
	
	def sample(self, model, seed, steps, cfg, batch_size, class_labels):
		device = comfy.model_management.get_torch_device()
		diffusion = create_diffusion(str(steps))

		# pre
		comfy.model_management.load_model_gpu(model)
		real_model = model.model

		# Create sampling noise:
		z = torch.randn(batch_size, 4, real_model.latent_size, real_model.latent_size, device=device)
		y = torch.tensor([class_labels] * batch_size, device=device)

		# Setup classifier-free guidance:
		z = torch.cat([z, z], 0)
		y_null = torch.tensor([1000] * batch_size, device=device)
		y = torch.cat([y, y_null], 0)
		model_kwargs = dict(y=y, cfg_scale=cfg)

		# Sample images:
		samples = diffusion.p_sample_loop(
			model.model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
		)
		samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
		samples = real_model.latent_format.process_out(samples.to(torch.float32))
		samples = samples.cpu()

		return ({"samples": samples},)

NODE_CLASS_MAPPINGS = {
	"DiTCheckpointLoader": DiTCheckpointLoader,
	"DiTSampler": DiTSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiTCheckpointLoader": DiTCheckpointLoader.TITLE,
    "DiTSampler": DiTSampler.TITLE,
}
