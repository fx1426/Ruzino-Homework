#!/usr/bin/env python3
"""
Python Render Graph to Tensor Test

Demonstrates the complete workflow:
1. Build render graph in Python
2. Execute rendering through USD/Hydra
3. Get output texture and convert to PyTorch tensor
"""

import sys
import os
from pathlib import Path
import numpy as np

# Setup paths
script_dir = Path(__file__).parent.resolve()
workspace_root = script_dir.parent.parent.parent.parent
binary_dir = workspace_root / "Binaries" / "Debug"

# Set environment variables
os.environ['PXR_USD_WINDOWS_DLL_PATH'] = str(binary_dir)
mtlx_stdlib = binary_dir / "libraries"
if mtlx_stdlib.exists():
    os.environ['PXR_MTLX_STDLIB_SEARCH_PATHS'] = str(mtlx_stdlib)
os.environ['PATH'] = str(binary_dir) + os.pathsep + os.environ.get('PATH', '')

if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(str(binary_dir))

os.chdir(str(binary_dir))
sys.path.insert(0, str(binary_dir))

# Import modules
try:
    import hd_USTC_CG_py as renderer
    import torch
    has_torch = True
except ImportError:
    has_torch = False

# Test configuration
WIDTH = 3000
HEIGHT = 3000
SAMPLES = 512

# Create HydraRenderer with USD scene
usd_stage = workspace_root / "Assets" / "shader_ball.usdc"
hydra = renderer.HydraRenderer(str(usd_stage), width=WIDTH, height=HEIGHT)

# Build render graph
import nodes_core_py as core
import nodes_system_py as system

def set_node_inputs(executor, inputs_dict):
    """Helper function to set multiple node inputs at once."""
    for (node, socket_name), value in inputs_dict.items():
        socket = node.get_input_socket(socket_name)
        if socket is None:
            raise ValueError(f"Socket '{socket_name}' not found on node '{node.ui_name}'")
        meta_value = core.to_meta_any(value)
        executor.sync_node_from_external_storage(socket, meta_value)

node_system = hydra.get_node_system()
node_system.load_configuration(str(binary_dir / "render_nodes.json"))
node_system.init()

tree = node_system.get_node_tree()
executor = node_system.get_node_tree_executor()

# Create render nodes
rng = tree.add_node("rng_texture")
rng.ui_name = "RNG"
ray_gen = tree.add_node("node_render_ray_generation")
ray_gen.ui_name = "RayGen"
path_trace = tree.add_node("path_tracing")
path_trace.ui_name = "PathTracer"
accumulate = tree.add_node("accumulate")
accumulate.ui_name = "Accumulate"
rng_buffer = tree.add_node("rng_buffer")
rng_buffer.ui_name = "RNGBuffer"
present = tree.add_node("present_color")
present.ui_name = "Present"

# Connect nodes
tree.add_link(rng.get_output_socket("Random Number"), ray_gen.get_input_socket("random seeds"))
tree.add_link(ray_gen.get_output_socket("Pixel Target"), path_trace.get_input_socket("Pixel Target"))
tree.add_link(ray_gen.get_output_socket("Rays"), path_trace.get_input_socket("Rays"))
tree.add_link(rng_buffer.get_output_socket("Random Number"), path_trace.get_input_socket("Random Seeds"))
tree.add_link(path_trace.get_output_socket("Output"), accumulate.get_input_socket("Texture"))
tree.add_link(accumulate.get_output_socket("Accumulated"), present.get_input_socket("Color"))

# Set parameters
executor.reset_allocator()
executor.prepare_tree(tree, present)

inputs = {
    (ray_gen, "Aperture"): 0.0,
    (ray_gen, "Focus Distance"): 2.0,
    (ray_gen, "Scatter Rays"): False,
    (accumulate, "Max Samples"): 16,
}

set_node_inputs(executor, inputs)

# Render
print(f"Rendering {SAMPLES} samples...")
for i in range(SAMPLES):
    hydra.render()

# Get output texture
texture_data = hydra.get_output_texture()
img_array = np.array(texture_data, dtype=np.float32).reshape(HEIGHT, WIDTH, 4)
rgb = img_array[:, :, :3]

print(f"Image: {img_array.shape}, mean={rgb.mean():.4f}, max={rgb.max():.4f}")

# Convert to tensor
if has_torch:
    tensor = torch.from_numpy(img_array)
    print(f"Tensor: {tensor.shape}, device={tensor.device}")
    
    if torch.cuda.is_available():
        tensor_gpu = tensor.cuda()
        print(f"GPU tensor: {tensor_gpu.device}")

# Save image
try:
    from PIL import Image
    img_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img_uint8 = np.flipud(img_uint8)
    Image.fromarray(img_uint8).save("./output_python_render.png")
    print("Saved: output_python_render.png")
except ImportError:
    pass

print("✓ Complete")
