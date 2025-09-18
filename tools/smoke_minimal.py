import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from uso.flux.pipeline import USOPipeline
import torch

os.environ.setdefault('FLUX_DEV', './weights/FLUX.1-dev/flux1-dev.safetensors')
os.environ.setdefault('PROJECTION_MODEL', './weights/USO/uso_flux_v1.0/projector.safetensors')

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print('device', device)

def run_case(name, pipeline_kwargs=None, call_kwargs=None, env_overrides=None):
	pipeline_kwargs = pipeline_kwargs or {}
	call_kwargs = call_kwargs or {}
	env_overrides = env_overrides or {}
	# apply env overrides
	old_env = {}
	for k, v in env_overrides.items():
		old_env[k] = os.environ.get(k)
		os.environ[k] = str(v)

	print('\n=== RUN CASE:', name, '===')
	pipe = USOPipeline('flux-dev', device, offload=False, hf_download=False, **pipeline_kwargs)
	res = pipe(**call_kwargs)

	# restore env
	for k, v in old_env.items():
		if v is None:
			del os.environ[k]
		else:
			os.environ[k] = v

	# save image
	out_dir = 'output/inference'
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, f"smoke_{name}.png")
	if isinstance(res, tuple):
		img = res[0]
	else:
		img = res
	img.save(out_path)
	print('Saved', out_path)
	return out_path


if __name__ == '__main__':
	# 1) Unconditional: zero text/vec conditioning
	run_case(
		'unconditional',
		pipeline_kwargs={'only_lora': False},
		call_kwargs={'prompt': '', 'width': 256, 'height': 256, 'guidance': 0.0, 'num_steps': 16, 'seed': 100, 'ref_imgs': [], 'siglip_inputs': None},
		env_overrides={'USO_COND_TXT_SCALE': '0.0', 'USO_COND_VEC_SCALE': '0.0'},
	)

	# 2) Text-only: disable siglip, no lora
	run_case(
		'text_only',
		pipeline_kwargs={'only_lora': False},
		call_kwargs={'prompt': 'A photorealistic portrait of a woman', 'width': 256, 'height': 256, 'guidance': 4.0, 'num_steps': 16, 'seed': 101, 'ref_imgs': [], 'siglip_inputs': None},
		env_overrides={'USO_COND_TXT_SCALE': '1.0', 'USO_COND_VEC_SCALE': '1.0'},
	)

	# 3) LoRA-only: load model in only_lora mode (no SigLIP), prompt empty
	run_case(
		'lora_only',
		pipeline_kwargs={'only_lora': True},
		call_kwargs={'prompt': 'A photorealistic portrait of a woman', 'width': 256, 'height': 256, 'guidance': 4.0, 'num_steps': 16, 'seed': 102, 'ref_imgs': [], 'siglip_inputs': None},
		env_overrides={'USO_COND_TXT_SCALE': '1.0', 'USO_COND_VEC_SCALE': '1.0'},
	)
