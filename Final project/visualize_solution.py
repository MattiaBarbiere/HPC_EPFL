#!/usr/bin/env python

import os
import h5py
import glob
import numpy as np
import re
import tqdm
import imageio.v2 as imageio
from natsort import natsorted

import matplotlib.pyplot as plt

import pyvista as pv
# pyvista needs this
if not hasattr(np, 'bool'):
    np.bool = bool


import argparse

def extract_index_from_path(path):
    filename = os.path.basename(path)
    match = re.search(r"_(\d+)\.*", filename)
    if not match:
        raise ValueError(f"Filename format not recognized: {filename}")
    return int(match.group(1))


if __name__ == "__main__":
    print("Visualizing SWE results...")
    parser = argparse.ArgumentParser(description="Program for visualizing SWE results.")
    parser.add_argument("--path", type=str, help="Path to the generated files. Defaults to current directory.", default=".")
    parser.add_argument("--basename", type=str, help="Base name for generated files (eg. water_drops). Defaults to 'water_drops'.", default="water_drops")
    parser.add_argument("--vis-type", type=str, 
                        help="Output type for visualization. Supported: '2d', '3d'. Defaults to '2d'.", 
                        default="2d")
    parser.add_argument("--frame", type=int, 
                        help="Integer number of the frame to visualize. -1 will output a video of all frames. Defaults to 0.", 
                        default=0)
    parser.add_argument("--output", type=str, 
                        help="Output folder name for the visualization, without extension. Images are saved as PNG. If not provided, the output will be shown in a window and not saved to a file.",
                        default=None) 

    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")
    basename = args.basename

    # Load mesh data
    mesh_file = os.path.join(path, f"{basename}_mesh.h5")
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh file {mesh_file} does not exist.")
    with h5py.File(mesh_file, "r") as f:
        points = f["vertices"][:]

    # Load topography data
    topo_file = os.path.join(path, f"{basename}_topography.h5")
    if not os.path.exists(topo_file):
        raise FileNotFoundError(f"Topography file {topo_file} does not exist.")
    with h5py.File(topo_file, "r") as f:
        topography = f["topography"][:]

    # Load solution data
    if args.frame >= 0:
        solution_files = [os.path.join(path, f"{basename}_h_{args.frame}.h5")]
    else:
        glob_pattern = os.path.join(path, f"{basename}_h*.h5")
        solution_files = sorted(glob.glob(glob_pattern), key=extract_index_from_path)
    if len(solution_files) == 0:
        raise FileNotFoundError(f"No solution files found.")
    
    h_data = []
    for fname in solution_files:
        # Before opening the file:
        if not (fname.endswith('.h5') or fname.endswith('.hdf5')):
            print(f"Skipping non-HDF5 file: {fname}")
            continue

        with h5py.File(fname, "r") as f:
            h_data.append(f["h"][:])

    
    # They keep results per-cell, but we plot them per-point.
    # So there is one more vertex than cell in each direction.
    # To get the number of cells, we subtract 1.
    nx = len(np.unique(points[:,0])) - 1
    ny = len(np.unique(points[:,1])) - 1

    # Cells remain uniformly spaced, and I think absolute coordinates
    # are irrelevant for the visualization, so we can just generate
    # a new meshgrid with uniform spacing. 
    x = np.linspace(points[:,0].min(), points[:,0].max(), nx)
    y = np.linspace(points[:,1].min(), points[:,1].max(), ny)
    X, Y = np.meshgrid(x, y)

    if args.frame >= 0:
        H = [h_data[0].reshape((ny, nx))]
    else:
        H = [h.reshape((ny, nx)) for h in h_data]
    
    B = topography.reshape((ny, nx))

    if args.output:
        output_filename_base = os.path.join(args.output, f"{basename}_vis") 
        os.makedirs(args.output, exist_ok=True)

    if args.vis_type == "2d":
        for i in tqdm.tqdm(range(len(H))):
            plt.figure(figsize=(8, 6))
            plt.pcolormesh(X, Y, H[i], shading="auto", cmap="viridis")
            plt.colorbar(label="Water height h")
            plt.title(f"Shallow Water Height at timestep {i}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("equal")
            plt.tight_layout()
            if args.output:
                plt.savefig(f"{output_filename_base}_{i:04d}.png")
                plt.close()
            else:
                plt.show()
                plt.close()
    

    elif args.vis_type == "3d":
        z_scale = 10 # Exaggerate the z-axis for better visualization

        # Set up a consistent camera position for the 3D plot
        if args.frame < 0 and args.output:
            Z_ref = z_scale * (B + np.max(h_data, axis=0).reshape((ny, nx)))
            grid_ref = pv.StructuredGrid(X, Y, Z_ref)
            # Create a temp plotter just to extract camera position
            p_ref = pv.Plotter(off_screen=True)
            p_ref.add_mesh(grid_ref)
            p_ref.view_isometric()
            p_ref.camera.zoom(0.9)
            camera_position = p_ref.camera_position
            p_ref.close()  # Important!

        for i in tqdm.tqdm(range(len(H))):
            Z = B + H[i]
            
            Z_exaggerated = z_scale * (B + h_data[i].reshape((ny, nx)))
            
            grid = pv.StructuredGrid(uinput = X, y=Y, z=Z_exaggerated)
            if args.output:
                plotter = pv.Plotter(off_screen=True)
            else:
                plotter = pv.Plotter(off_screen=False)
            
            plotter.add_mesh(grid, scalars=grid.points[:, 2], cmap="viridis")

            plotter.set_background("white")
            # Update camera position to avoid wobbling
            if args.frame < 0 and args.output:
                plotter.camera_position = camera_position
            
            if args.output:
                plotter.screenshot(f"{output_filename_base}_{i:04d}.png")
                plotter.close()
            else:
                plotter.show()
                plotter.close()


    ## Make the annimation video from the frames
    frame_dir = "frames"
    pattern = os.path.join(frame_dir, f"{basename}_vis_*.png")

    # Output video file
    output_video = f"animations/{basename}_animation.mp4"

    # Collect frame filenames and sort them
    frame_files = natsorted(glob.glob(pattern))

    # Check if frames exist
    if not frame_files:
        raise FileNotFoundError(f"No PNG files found in {pattern}")

    # Set frame rate
    fps = 20

    # Write video
    with imageio.get_writer(output_video, fps=fps, codec='libx264') as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"Video saved as {output_video}")

    # Clean up frame directory if it exists
    if os.path.exists(frame_dir):
        for file in frame_files:
            os.remove(file)
        os.rmdir(frame_dir)
        print(f"Cleaned up temporary files in {frame_dir}")
    else:
        print(f"No temporary files to clean up in {frame_dir}")
