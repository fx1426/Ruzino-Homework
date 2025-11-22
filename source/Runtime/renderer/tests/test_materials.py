"""
Script to test materials by binding them to shader_ball meshes and rendering
"""
import sys
import os
import glob
import subprocess
from pathlib import Path

# Setup paths
tests_dir = os.path.dirname(os.path.abspath(__file__))
binary_dir = os.path.abspath(os.path.join(tests_dir, "..", "..", "..", "..", "Binaries", "Debug"))

# Set PXR_USD_WINDOWS_DLL_PATH so USD can find its DLLs
os.environ['PXR_USD_WINDOWS_DLL_PATH'] = binary_dir
print(f"Set PXR_USD_WINDOWS_DLL_PATH={binary_dir}")

# Set MaterialX standard library path for USD MaterialX plugin
mtlx_stdlib = os.path.join(binary_dir, "libraries")
if os.path.exists(mtlx_stdlib):
    os.environ['PXR_MTLX_STDLIB_SEARCH_PATHS'] = mtlx_stdlib
    print(f"Set PXR_MTLX_STDLIB_SEARCH_PATHS={mtlx_stdlib}")
else:
    print(f"Warning: MaterialX stdlib not found at {mtlx_stdlib}")

# Add to Python path
sys.path.insert(0, binary_dir)

# DON'T change working directory here - keep it in the original location
# This prevents MaterialX from generating absolute paths in shader imports
print(f"Working directory: {os.getcwd()}\n")

from pxr import Usd, UsdGeom, UsdShade, Sdf, UsdMtlx

def find_all_materials():
    """Find all .mtlx files in the matx_library"""
    # Calculate path relative to script location
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.abspath(os.path.join(tests_dir, "..", "..", "..", "..", "Assets"))
    matx_library = os.path.join(assets_dir, "matx_library")
    
    materials = []
    for mtlx_file in glob.glob(os.path.join(matx_library, "**", "*.mtlx"), recursive=True):
        materials.append(mtlx_file)
    
    materials.sort()
    print(f"Found {len(materials)} material files")
    return materials

def bind_material_to_meshes(shader_ball_path, material_path, output_path):
    """
    Bind a MaterialX material to both meshes in shader_ball.usdc
    and save to a new USD file
    """
    # Open the shader ball stage
    stage = Usd.Stage.Open(shader_ball_path)
    if not stage:
        print(f"Failed to open {shader_ball_path}")
        return False
    
    # Find the two mesh prims
    preview_mesh_path = "/root/Preview_Mesh/Preview_Mesh"
    calibration_mesh_path = "/root/Calibration_Mesh/Calibration_Mesh"
    
    preview_mesh = stage.GetPrimAtPath(preview_mesh_path)
    calibration_mesh = stage.GetPrimAtPath(calibration_mesh_path)
    
    if not preview_mesh or not calibration_mesh:
        print(f"Failed to find meshes in {shader_ball_path}")
        return False
    
    # Create or get the materials scope
    materials_scope_path = "/root/_materials"
    materials_scope = stage.GetPrimAtPath(materials_scope_path)
    if not materials_scope:
        materials_scope = stage.DefinePrim(materials_scope_path, "Scope")
    
    # Get material name from file
    material_name = Path(material_path).stem
    material_path_in_stage = f"{materials_scope_path}/{material_name}"
    
    # Remove old material if exists
    if stage.GetPrimAtPath(material_path_in_stage):
        stage.RemovePrim(material_path_in_stage)
    
    print(f"  Using MaterialX file reference method...")
    
    # First, open the MaterialX file to find the material prim path
    mtlx_stage = Usd.Stage.Open(material_path)
    if not mtlx_stage:
        print(f"  ERROR: Failed to open MaterialX file: {material_path}")
        return False
    
    # Find the material in the MaterialX file
    mtlx_material_path = None
    for prim in mtlx_stage.Traverse():
        if prim.IsA(UsdShade.Material):
            mtlx_material_path = prim.GetPath()
            print(f"  Found material in .mtlx file at: {mtlx_material_path}")
            break
    
    if not mtlx_material_path:
        print(f"  ERROR: No material found in {material_path}")
        return False
    
    # Create a Material prim and reference the specific material from the .mtlx file
    material_prim = stage.DefinePrim(material_path_in_stage, "Material")
    
    # Add reference to the MaterialX file with the specific prim path
    material_prim.GetReferences().AddReference(material_path, mtlx_material_path)
    
    # Wrap in UsdShade.Material
    material = UsdShade.Material(material_prim)
    
    # Debug: Print what we got
    print(f"  Created material prim at: {material_path_in_stage}")
    print(f"  Material has {len(list(material_prim.GetChildren()))} children")
    
    # The reference brings in the shader, but we need to manually connect the surface output
    # Find the shader that was brought in
    shader_prim = None
    for child in material_prim.GetChildren():
        if child.IsA(UsdShade.Shader):
            shader_prim = child
            break
    
    if shader_prim:
        print(f"  Found shader: {shader_prim.GetPath()}")
        shader = UsdShade.Shader(shader_prim)
        
        # Get or create the shader's surface output
        shader_surface_output = shader.GetOutput("surface")
        if not shader_surface_output:
            print(f"  WARNING: Shader has no surface output!")
        else:
            # Connect the material's surface output to the shader's surface output
            material_surface_output = material.CreateSurfaceOutput()
            material_surface_output.ConnectToSource(shader_surface_output)
            print(f"  Connected material surface output to shader surface output")
    else:
        print(f"  WARNING: No shader found in material!")
    
    # Check for surface output
    surface_output = material.GetSurfaceOutput()
    if surface_output:
        sources = surface_output.GetConnectedSources()
        if sources and len(sources) > 0 and len(sources[0]) > 0:
            source_info = sources[0][0]
            print(f"  ✓ Surface output connected to: {source_info.source.GetPath()}")
        else:
            print(f"  ✗ WARNING: Surface output exists but not connected!")
    else:
        print(f"  ✗ WARNING: No surface output found on material!")
    
    # List all children to see what was imported (optional debug info)
    # for child in material_prim.GetChildren():
    #     print(f"    Child: {child.GetPath()} (Type: {child.GetTypeName()})")
    #     if child.IsA(UsdShade.Shader):
    #         shader = UsdShade.Shader(child)
    #         for output in shader.GetOutputs():
    #             print(f"      Output: {output.GetFullName()}")
    
    # Bind the material to both meshes
    binding_api_preview = UsdShade.MaterialBindingAPI(preview_mesh)
    binding_api_calibration = UsdShade.MaterialBindingAPI(calibration_mesh)
    
    binding_api_preview.Bind(material)
    binding_api_calibration.Bind(material)
    
    print(f"  Bound material to Preview_Mesh")
    print(f"  Bound material to Calibration_Mesh")
    
    # Verify bindings
    bound_mat_preview = binding_api_preview.ComputeBoundMaterial()[0]
    bound_mat_calibration = binding_api_calibration.ComputeBoundMaterial()[0]
    
    if bound_mat_preview:
        print(f"  Verified Preview_Mesh binding: {bound_mat_preview.GetPath()}")
    else:
        print(f"  ERROR: Preview_Mesh binding verification failed!")
        
    if bound_mat_calibration:
        print(f"  Verified Calibration_Mesh binding: {bound_mat_calibration.GetPath()}")
    else:
        print(f"  ERROR: Calibration_Mesh binding verification failed!")
    
    # Save to output file
    stage.GetRootLayer().Export(output_path)
    print(f"  Saved modified stage to: {output_path}")
    
    return True

def render_scene(usd_file, output_image, width=1920, height=1080, samples=4):
    """
    Render a USD scene using headless_render.exe
    """
    # Calculate paths relative to script location
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    binary_dir = os.path.abspath(os.path.join(tests_dir, "..", "..", "..", "..", "Binaries", "Debug"))
    render_exe = os.path.join(binary_dir, "headless_render.exe")
    assets_dir = os.path.abspath(os.path.join(tests_dir, "..", "..", "..", "..", "Assets"))
    render_nodes = os.path.join(assets_dir, "render_nodes_save.json")
    
    # Convert paths to relative paths from the binary directory
    # This matches the manual execution which uses relative paths
    usd_file_rel = os.path.relpath(usd_file, binary_dir)
    output_image_rel = os.path.relpath(output_image, binary_dir)
    render_nodes_rel = os.path.relpath(render_nodes, binary_dir)
    
    cmd = [
        render_exe,  # Use absolute path to the exe
        usd_file_rel,
        render_nodes_rel,
        output_image_rel,
        str(width),
        str(height),
        str(samples)
    ]
    
    print(f"  Rendering from directory: {binary_dir}")
    print(f"  Command: {os.path.basename(render_exe)} {' '.join(cmd[1:])}")
    
    try:
        # CRITICAL: Set cwd to binary_dir so relative paths work correctly
        # This matches manual execution where you cd into binary_dir first
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=binary_dir)
        if result.returncode != 0:
            print(f"  ERROR: Render failed with code {result.returncode}")
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
            return False
        else:
            print(f"  Render succeeded!")
            return True
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Render timed out after 300 seconds")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    # Calculate paths relative to script location
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.abspath(os.path.join(tests_dir, "..", "..", "..", "..", "Assets"))
    binary_dir = os.path.abspath(os.path.join(tests_dir, "..", "..", "..", "..", "Binaries", "Debug"))
    shader_ball = os.path.join(assets_dir, "shader_ball.usdc")
    output_dir = os.path.join(binary_dir, "material_tests")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all materials
    materials = find_all_materials()
    
    if not materials:
        print("No materials found!")
        return 1
    
    # Test with first N materials
    test_count = min(10, len(materials))  # Test first 10 materials
    print(f"\nTesting first {test_count} materials...\n")
    
    success_count = 0
    fail_count = 0
    
    for i, material_path in enumerate(materials[:test_count]):
        material_name = Path(material_path).stem
        print(f"[{i+1}/{test_count}] Testing material: {material_name}")
        print(f"  Material path: {material_path}")
        
        # Create modified USD file with material bound
        modified_usd = os.path.join(output_dir, f"shader_ball_{material_name}.usdc")
        if not bind_material_to_meshes(shader_ball, material_path, modified_usd):
            print(f"  FAILED: Could not bind material")
            fail_count += 1
            continue
        
        # Render
        output_image = os.path.join(output_dir, f"{material_name}.png")
        if render_scene(modified_usd, output_image):
            print(f"  SUCCESS: Rendered to {output_image}")
            success_count += 1
        else:
            print(f"  FAILED: Render failed")
            fail_count += 1
        
        print()
    
    print("=" * 80)
    print(f"Results: {success_count} succeeded, {fail_count} failed out of {test_count} tests")
    print("=" * 80)
    
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
