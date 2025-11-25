#!/usr/bin/env python3
"""
Material BRDF Visualizer Test

Analyzes material BRDF properties:
1. BRDF Eval distribution
2. PDF distribution  
3. Sample frequency distribution

Outputs three visualization images for analysis.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Setup paths
script_dir = Path(__file__).parent.resolve()
workspace_root = script_dir.parent.parent.parent.parent
binary_dir = workspace_root / "Binaries" / "Debug"
assets_dir = workspace_root / "Assets"

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

# Import USD first
from pxr import Usd, UsdGeom, UsdShade, Sdf, UsdMtlx

# Import modules
import hd_USTC_CG_py as renderer
import nodes_core_py as core
import nodes_system_py as system

def set_node_inputs(executor, inputs_dict):
    """Helper to set multiple node inputs."""
    for (node, socket_name), value in inputs_dict.items():
        socket = node.get_input_socket(socket_name)
        if socket is None:
            raise ValueError(f"Socket '{socket_name}' not found on node '{node.ui_name}'")
        meta_value = core.to_meta_any(value)
        executor.sync_node_from_external_storage(socket, meta_value)

def bind_material_to_shader_ball(shader_ball_path, material_path, output_path):
    """Bind a MaterialX material to shader_ball meshes"""
    stage = Usd.Stage.Open(str(shader_ball_path))
    if not stage:
        return False
    
    # Find meshes
    preview_mesh = stage.GetPrimAtPath("/root/Preview_Mesh/Preview_Mesh")
    calibration_mesh = stage.GetPrimAtPath("/root/Calibration_Mesh/Calibration_Mesh")
    
    if not preview_mesh or not calibration_mesh:
        print(f"✗ Could not find meshes in shader_ball")
        return False
    
    # Create materials scope
    materials_scope_path = "/root/_materials"
    materials_scope = stage.GetPrimAtPath(materials_scope_path)
    if not materials_scope:
        materials_scope = stage.DefinePrim(materials_scope_path, "Scope")
    
    # Get material name
    material_name = material_path.stem
    material_path_in_stage = f"{materials_scope_path}/{material_name}"
    
    # Remove old material if exists
    if stage.GetPrimAtPath(material_path_in_stage):
        stage.RemovePrim(material_path_in_stage)
    
    # Open MaterialX file to find material
    mtlx_stage = Usd.Stage.Open(str(material_path))
    if not mtlx_stage:
        print(f"✗ Could not open MaterialX file: {material_path}")
        return False
    
    mtlx_material_path = None
    for prim in mtlx_stage.Traverse():
        if prim.IsA(UsdShade.Material):
            mtlx_material_path = prim.GetPath()
            break
    
    if not mtlx_material_path:
        print(f"✗ No material found in {material_path}")
        return False
    
    # Create material prim and reference the .mtlx file
    material_prim = stage.DefinePrim(material_path_in_stage, "Material")
    material_prim.GetReferences().AddReference(str(material_path), mtlx_material_path)
    material = UsdShade.Material(material_prim)
    
    # Connect surface output
    shader_prim = None
    for child in material_prim.GetChildren():
        if child.IsA(UsdShade.Shader):
            shader_prim = child
            break
    
    if shader_prim:
        shader = UsdShade.Shader(shader_prim)
        shader_surface_output = shader.GetOutput("surface")
        if shader_surface_output:
            material_surface_output = material.CreateSurfaceOutput()
            material_surface_output.ConnectToSource(shader_surface_output)
    
    # Bind material to meshes
    UsdShade.MaterialBindingAPI(preview_mesh).Bind(material)
    UsdShade.MaterialBindingAPI(calibration_mesh).Bind(material)
    
    # Save
    stage.GetRootLayer().Export(str(output_path))
    print(f"✓ Material bound and saved to: {output_path}")
    return True

# Test configuration
RESOLUTION = 512
NUM_SAMPLES = 100000

print("="*70)
print("Material BRDF Analyzer")
print("="*70)

# Bind material to shader_ball
shader_ball_path = assets_dir / "shader_ball.usdc"
material_path = assets_dir / "matx_library" / "Aluminum_1k_8b_tAdaTTp" / "Aluminum.mtlx"
output_usd = binary_dir / "shader_ball_with_material.usdc"

print(f"Shader ball: {shader_ball_path}")
print(f"Material: {material_path.name}")
print(f"Material exists: {material_path.exists()}")

if not bind_material_to_shader_ball(shader_ball_path, material_path, output_usd):
    print("✗ Failed to bind material")
    exit(1)

print()

# Create HydraRenderer with material-bound USD scene
print(f"Loading USD stage: {output_usd}")
try:
    hydra = renderer.HydraRenderer(str(output_usd), width=512, height=512)
    print("✓ HydraRenderer created successfully")
except Exception as e:
    print(f"✗ Failed to create HydraRenderer: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Render a frame to initialize materials
print("Rendering frame to initialize materials...")
try:
    hydra.render()
    print("✓ Frame rendered, materials should be loaded")
except Exception as e:
    print(f"✗ Failed to render: {e}")

print()

# Build analysis graph
node_system = hydra.get_node_system()
config_path = binary_dir / "render_nodes.json"
print(f"Loading configuration from: {config_path}")
print(f"Config file exists: {config_path.exists()}")
node_system.load_configuration(str(config_path))
node_system.init()

tree = node_system.get_node_tree()
executor = node_system.get_node_tree_executor()

print(f"Available node types: {tree.get_all_node_types() if hasattr(tree, 'get_all_node_types') else 'N/A'}")

# Create BRDF analyzer node
try:
    analyzer = tree.add_node("material_brdf_analyzer")
    print(f"✓ Successfully created analyzer node")
except Exception as e:
    print(f"✗ Failed to create analyzer node: {e}")
    exit(1)
analyzer.ui_name = "BRDFAnalyzer"

# Create present nodes for each output
present_eval = tree.add_node("present_color")
present_eval.ui_name = "PresentEval"
present_pdf = tree.add_node("present_color")
present_pdf.ui_name = "PresentPDF"
present_sample = tree.add_node("present_color")
present_sample.ui_name = "PresentSample"

# Connect analyzer outputs to present nodes
tree.add_link(analyzer.get_output_socket("BRDF Eval"), present_eval.get_input_socket("Color"))
tree.add_link(analyzer.get_output_socket("PDF"), present_pdf.get_input_socket("Color"))
tree.add_link(analyzer.get_output_socket("Sample Distribution"), present_sample.get_input_socket("Color"))

# Set analysis parameters
incident_dir_x, incident_dir_y, incident_dir_z = 0.0, 0.0, 1.0  # Normal incidence
uv_x, uv_y = 0.5, 0.5

inputs = {
    (analyzer, "Incident Direction X"): incident_dir_x,
    (analyzer, "Incident Direction Y"): incident_dir_y,
    (analyzer, "Incident Direction Z"): incident_dir_z,
    (analyzer, "UV X"): uv_x,
    (analyzer, "UV Y"): uv_y,
    (analyzer, "Material ID"): 0,
    (analyzer, "Resolution"): RESOLUTION,
    (analyzer, "Num Samples"): NUM_SAMPLES,
}

print(f"Analysis parameters:")
print(f"  Incident direction: ({incident_dir_x}, {incident_dir_y}, {incident_dir_z})")
print(f"  Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"  Number of samples: {NUM_SAMPLES}")
print()

# Execute analysis for BRDF Eval
print("1/3 Computing BRDF Eval distribution...")
executor.reset_allocator()
executor.prepare_tree(tree, present_eval)
set_node_inputs(executor, inputs)
executor.execute_tree(tree)

brdf_eval_data = hydra.get_output_texture("PresentEval")
eval_array = np.array(brdf_eval_data, dtype=np.float32).reshape(RESOLUTION, RESOLUTION, 4)
print(f"✓ BRDF Eval: mean={eval_array[:,:,:3].mean():.6f}, max={eval_array[:,:,:3].max():.6f}")

# Execute analysis for PDF
print("2/3 Computing PDF distribution...")
executor.reset_allocator()
executor.prepare_tree(tree, present_pdf)
set_node_inputs(executor, inputs)
executor.execute_tree(tree)

pdf_data = hydra.get_output_texture("PresentPDF")
pdf_array = np.array(pdf_data, dtype=np.float32).reshape(RESOLUTION, RESOLUTION, 4)
print(f"✓ PDF: mean={pdf_array[:,:,:3].mean():.6f}, max={pdf_array[:,:,:3].max():.6f}")

# Execute analysis for Sample Distribution
print("3/3 Computing Sample Distribution...")
executor.reset_allocator()
executor.prepare_tree(tree, present_sample)
set_node_inputs(executor, inputs)
executor.execute_tree(tree)

sample_data = hydra.get_output_texture("PresentSample")
sample_array = np.array(sample_data, dtype=np.float32).reshape(RESOLUTION, RESOLUTION, 4)
print(f"✓ Sample Distribution: mean={sample_array[:,:,:3].mean():.6f}, max={sample_array[:,:,:3].max():.6f}")
print()

# Save images
try:
    from PIL import Image
    
    def save_texture(array, filename):
        rgb = array[:, :, :3]
        img_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        img_uint8 = np.flipud(img_uint8)
        Image.fromarray(img_uint8).save(filename)
        print(f"✓ Saved: {filename}")
    
    save_texture(eval_array, "./brdf_eval.png")
    save_texture(pdf_array, "./brdf_pdf.png")
    save_texture(sample_array, "./brdf_sample_distribution.png")
    
except ImportError:
    print("✗ PIL not available, skipping image save")

print()
print("="*70)
print("✓ BRDF Analysis Complete!")
print("="*70)
