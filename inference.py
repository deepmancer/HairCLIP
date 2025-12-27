"""
HairCLIP Batch Inference Script

This script performs batch hair transfer using HairCLIP with reference images.
It follows the same structure as HairCLIPv2 and other baselines for consistency.

For hair transfer using reference images:
- source_id: The face/identity image (keeps identity)
- target_id: The reference image for hairstyle and color
"""

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
HAIRCLIP_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0,'./encoder4editing')

# Import e4e encoder from HairCLIP's encoder4editing
from models.psp import pSp

# Get the HairCLIP root directory
sys.path.insert(0, HAIRCLIP_ROOT)
# Add paths for HairCLIP modules (following predict.py structure)
# Order matters - HairCLIP root first, then encoder4editing
# sys.path.insert(0, HAIRCLIP_ROOT)
from mapper.hairclip_mapper import HairCLIPMapper

# Import hair mask generation from criteria folder
from criteria.parse_related_loss import average_lab_color_loss


# Import HairCLIP mapper (uses models.stylegan2 from HAIRCLIP_ROOT)


def tensor_to_pil(tensor_img):
    """Convert a tensor image to PIL Image."""
    if tensor_img.dim() == 4:
        tensor_img = tensor_img[0]
    # Denormalize from [-1, 1] to [0, 1]
    tensor_img = (tensor_img + 1) / 2
    tensor_img = tensor_img.clamp(0, 1)
    # Convert to numpy
    img_np = tensor_img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def find_image_path(image_dir: Path, image_id: str) -> Path:
    """Find image path by checking multiple extensions."""
    for ext in ['.png', '.jpeg', '.jpg', '.webp']:
        candidate = image_dir / f'{image_id}{ext}'
        if candidate.exists():
            return candidate
    return image_dir / f'{image_id}.png'


def load_e4e_model(device='cuda'):
    """Load the e4e encoder model."""
    e4e_model_path = os.path.join(HAIRCLIP_ROOT, 'pretrained_models/e4e_ffhq_encode.pt')
    
    print(f"Loading e4e model from {e4e_model_path}...")
    ckpt = torch.load(e4e_model_path, map_location='cpu')
    
    opts = ckpt['opts']
    opts['checkpoint_path'] = e4e_model_path
    opts['device'] = device
    opts = Namespace(**opts)
    
    e4e_net = pSp(opts)
    e4e_net.eval()
    e4e_net.to(device)
    
    return e4e_net


def load_hairclip_model(device='cuda'):
    """Load the HairCLIP mapper model."""
    checkpoint_path = os.path.join(HAIRCLIP_ROOT, 'pretrained_models/hairclip.pt')
    parsenet_path = os.path.join(HAIRCLIP_ROOT, 'pretrained_models/parsenet.pth')
    stylegan_path = os.path.join(HAIRCLIP_ROOT, 'pretrained_models/stylegan2-ffhq-config-f.pt')
    
    print(f"Loading HairCLIP model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    opts = ckpt['opts']
    opts = Namespace(**opts)
    
    # Set required paths
    opts.checkpoint_path = checkpoint_path
    opts.parsenet_weights = parsenet_path
    opts.stylegan_weights = stylegan_path
    opts.device = device
    
    # For reference image-based editing
    opts.editing_type = 'both'  # Edit both hairstyle and color
    opts.input_type = 'image_image'  # Use reference images for both
    
    net = HairCLIPMapper(opts)
    net.eval()
    net.to(device)
    
    return net, opts


def run_on_batch(w, hairstyle_text_inputs, color_text_inputs, 
                hairstyle_tensor_hairmasked, color_tensor_hairmasked, net):
    """Run HairCLIP inference on a batch."""
    with torch.no_grad():
        # Apply mapper with scaling factor 0.1
        w_hat = w + 0.1 * net.mapper(
            w, 
            hairstyle_text_inputs, 
            color_text_inputs, 
            hairstyle_tensor_hairmasked, 
            color_tensor_hairmasked
        )
        
        # Decode with StyleGAN
        x_hat, w_hat = net.decoder(
            [w_hat], 
            input_is_latent=True, 
            return_latents=True, 
            randomize_noise=False, 
            truncation=1
        )
        
        # Also decode original for comparison
        x_orig, _ = net.decoder(
            [w], 
            input_is_latent=True, 
            randomize_noise=False, 
            truncation=1
        )
        
    return x_hat, w_hat, x_orig


def main(args):
    """Main function for batch hair transfer using HairCLIP."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup data directory
    data_dir = Path(args.data_dir)
    
    # Load pairs from CSV file
    pairs_csv_path = data_dir / 'pairs.csv'
    if not pairs_csv_path.exists():
        raise FileNotFoundError(f"pairs.csv not found in {data_dir}")
    
    df = pd.read_csv(pairs_csv_path)
    
    # Validate required columns
    if 'source_id' not in df.columns or 'target_id' not in df.columns:
        raise ValueError("pairs.csv must contain 'source_id' and 'target_id' columns")
    
    # Shuffle the pairs for consistent ordering with other methods
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Setup paths
    image_dir = data_dir / 'aligned_image'
    if not image_dir.exists():
        image_dir = data_dir / 'image'
        if not image_dir.exists():
            raise FileNotFoundError(f"Neither 'aligned_image' nor 'image' folder found in {data_dir}")
    
    # Create output and cache directories
    output_base_dir = data_dir / 'baselines' / 'hairclip'
    cache_dir = output_base_dir / '_cache'
    latent_cache_dir = cache_dir / 'latents'
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(latent_cache_dir, exist_ok=True)
    
    # Image transforms
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    img_transform_1024 = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load models
    print("Loading models...")
    e4e_net = load_e4e_model(device)
    hairclip_net, opts = load_hairclip_model(device)
    
    # Load average color loss for hair masking
    avg_color_loss = average_lab_color_loss.AvgLabLoss(opts).to(device).eval()
    
    print("Starting batch processing...")
    
    # Counters
    successful = 0
    skipped = 0
    failed = 0
    
    # Process each pair
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing pairs"):
        source_id = str(row['source_id'])
        target_id = str(row['target_id'])
        
        # Find image paths
        source_path = find_image_path(image_dir, source_id)
        target_path = find_image_path(image_dir, target_id)
        
        # Check if images exist
        if not source_path.exists():
            print(f"Warning: Source image not found: {source_path}")
            failed += 1
            continue
        if not target_path.exists():
            print(f"Warning: Target image not found: {target_path}")
            failed += 1
            continue
        
        # Output directory and path
        sample_id = f'{target_id}_to_{source_id}'
        output_dir = output_base_dir / sample_id
        output_path = output_dir / 'transferred.png'
        
        # Skip if already processed
        if output_path.exists():
            skipped += 1
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # ============================================
            # Step 1: Get latent code for source image
            # ============================================
            latent_cache_path = latent_cache_dir / f'{source_id}.npy'
            
            if latent_cache_path.exists():
                # Load cached latent
                src_latent = torch.from_numpy(np.load(latent_cache_path)).to(device)
            else:
                # Compute latent using e4e
                src_img = Image.open(source_path).convert('RGB')
                src_tensor = img_transform(src_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    _, src_latent = e4e_net(src_tensor, randomize_noise=False, return_latents=True)
                
                # Cache the latent
                np.save(latent_cache_path, src_latent.cpu().numpy())
            
            # ============================================
            # Step 2: Load and process reference image
            # ============================================
            ref_img = Image.open(target_path).convert('RGB')
            ref_tensor = img_transform_1024(ref_img).unsqueeze(0).to(device)
            
            # Generate hair mask for reference image
            with torch.no_grad():
                ref_tensor_hairmasked = ref_tensor * avg_color_loss.gen_hair_mask(ref_tensor)
            
            # ============================================
            # Step 3: Prepare inputs for HairCLIP
            # ============================================
            # For reference image-based editing, text inputs are placeholders
            hairstyle_text_inputs = torch.Tensor([0]).unsqueeze(0).to(device)
            color_text_inputs = torch.Tensor([0]).unsqueeze(0).to(device)
            
            # Use reference image for both hairstyle and color
            hairstyle_tensor_hairmasked = ref_tensor_hairmasked
            color_tensor_hairmasked = ref_tensor_hairmasked
            
            # ============================================
            # Step 4: Run HairCLIP inference
            # ============================================
            x_hat, w_hat, x_orig = run_on_batch(
                src_latent.float(),
                hairstyle_text_inputs,
                color_text_inputs,
                hairstyle_tensor_hairmasked,
                color_tensor_hairmasked,
                hairclip_net
            )
            
            # ============================================
            # Step 5: Save result
            # ============================================
            # x_hat is the edited image
            result_img = tensor_to_pil(x_hat)
            result_img.save(output_path)
            
            successful += 1
            
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"  Successful: {successful}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HairCLIP batch inference')
    
    parser.add_argument('--data_dir', type=str, default="/workspace/celeba_subset",
                        help='Directory containing pairs.csv and image/ folder')
    
    args = parser.parse_args()
    main(args)
