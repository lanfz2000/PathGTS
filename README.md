# PathGTS

Implementation of *PathGTS: distilling generative and generalist foundation models into pathology specialists for efficient precision oncology*.

PathGTS uses CytoSyn to generate reference-conditioned histology patches. A disease-specific ViT-B/16 then learns from a frozen pathology foundation model with cosine feature distillation and iBOT masked-image modeling. The frozen student backbone produces 768-dimensional patch embeddings for downstream tasks.

## Install

Use Python 3.11 and a CUDA-enabled PyTorch environment for generation and training. The training environment used PyTorch 2.8 with CUDA 12.8.

```bash
pip install -r requirements.txt
```

Obtain the model files from their providers under their respective access terms.

- [CytoSyn](https://huggingface.co/Owkin/CytoSyn) and [H0-mini](https://huggingface.co/bioptimus/H0-mini) for synthetic patch generation. Use the local CytoSyn pipeline snapshot with its upstream pipeline code and a local H0-mini checkpoint directory.
- [Virchow2](https://huggingface.co/paige-ai/Virchow2) for the main distillation teacher. Pass its local model directory to `--virchow2_dir` or authenticate with Hugging Face to load the gated model through timm.
- [ViT-B/16 AugReg](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k) for student initialization. Pass its `model.safetensors` to `--student_ckpt`.

## Prepare patches and generate variants

Tissue patches should be RGB H&E tiles extracted at nominal 10×, with a size of 256 × 256 pixels before resizing. Prepare a `references.csv` with `image_paths,slide_id` columns and a `split.csv` with `slide_id,split` columns. The split values are `train`, `val`, and `test`. Use separate manifests for the breast and HNSC specialists. Paths are interpreted relative to the working directory unless absolute.

```bash
python generate.py --mode cond \
  --cytosyn_path /path/to/CytoSyn --h0_mini_path /path/to/H0-mini \
  --reference_csv references.csv --out_dir generated \
  --num_images 10 --steps 50 --guidance_scale 1.0

python prepare_manifest.py --references references.csv \
  --synthetic_dir generated --output patches.csv
```

Use `--num_images 8` for HNSC. Keep the order of `references.csv` fixed between generation and manifest creation. The manifest associates each generated patch with its reference slide and keeps the pair in the same partition.

## Train a specialist

The command below runs the breast configuration on two GPUs. The batch size is per GPU and the global batch size is 512. The real-to-synthetic ratio is approximately 1:2 in each minibatch.

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --devices 2 --bs 256 --epochs 10 --lr 1e-4 --model_size base \
  --teachers virchow2 --student_ckpt /path/to/model.safetensors \
  --data_csv patches.csv --split_csv split.csv \
  --synthetic_ratio 2 --synthetic_alpha_start 0 --synthetic_alpha_end 0.5 \
  --ibot_lambda_start 0.1 --ibot_lambda_end 0.3 \
  --save_dir runs/pathgts-bc
```

For the HNSC specialist, use its own patches and set `--synthetic_alpha_end 0.3`. The training script also supports teacher ablations with `--teachers conchv1.5 --titan_dir /path/to/TITAN` or `--teachers uni --uni_ckpt /path/to/pytorch_model.bin`. For multiple GPUs, launch with `torchrun` to shard the patch sampler across ranks.

## Extract frozen patch embeddings

```bash
python extract_features.py \
  --checkpoint /path/to/specialist.ckpt \
  --data_csv evaluation_patches.csv --output features.npy
```

The feature extractor reads the backbone from either a training checkpoint or a compact checkpoint in the run's `probe_checkpoints` directory. Output rows follow the order of the input `image_paths` column. Obtain datasets and model files separately.
