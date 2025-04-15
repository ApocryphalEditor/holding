# --- START OF FILE run_srm_sweep_multi_threshold.py ---

import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import traceback
import datetime
import json
import matplotlib.cm as cm # For color mapping

# Constants
DIMENSION = 3072 # Expected dimension of the vectors

def load_vectors_from_npz(file_path, expected_dim):
    """Loads vectors from an .npz file, validating their shape."""
    # (Keep this function exactly the same as in the previous run_srm_sweep.py)
    vectors = []
    print(f"Loading vectors from: {file_path}")
    try:
        with np.load(file_path, allow_pickle=True) as loaded_data:
            keys = list(loaded_data.files)
            # print(f"Found {len(keys)} keys.") # Less verbose during loading
            valid_count = 0
            skipped_count = 0
            for key in tqdm(keys, desc=f"Loading {os.path.basename(file_path)}", leave=False, unit="vec"):
                try:
                    vec = loaded_data[key]
                    if not isinstance(vec, np.ndarray):
                        # print(f"\nWarning: Skipping key '{key}'. Value is not a NumPy array (type: {type(vec)}).")
                        skipped_count += 1
                        continue
                    if vec.shape == (expected_dim,):
                        vectors.append(vec)
                        valid_count +=1
                    else:
                         # print(f"\nWarning: Skipping key '{key}'. Expected shape ({expected_dim},), got {vec.shape}.")
                         skipped_count += 1
                except Exception as e:
                    # print(f"\nError processing key '{key}' in {file_path}: {e}")
                    skipped_count += 1
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path}': {e}")
        traceback.print_exc()
        return None

    # print(f"Successfully loaded {len(vectors)} vectors with dimension {expected_dim} from {file_path}.")
    if skipped_count > 0:
         print(f"Skipped {skipped_count} invalid entries in {file_path}.")
    if not vectors:
        print(f"Warning: No valid vectors loaded from {file_path}.")
        return None

    return np.array(vectors) # Shape: [num_vectors, dimension]


def run_srm_sweep_multi(data_vectors, neuron_a_idx, neuron_b_idx, thresholds, num_angles=360):
    """
    Performs the SRM sweep analysis for multiple thresholds.

    Args:
        data_vectors (np.ndarray): Array of shape [N, D] containing data vectors.
        neuron_a_idx (int): Index of the first neuron defining the plane.
        neuron_b_idx (int): Index of the second neuron defining the plane.
        thresholds (list[float]): List of similarity thresholds for counting.
        num_angles (int): Number of angles to sweep (0 to 360 degrees).

    Returns:
        pd.DataFrame: DataFrame containing results for each angle & threshold.
                      Columns: 'angle_deg', 'mean_similarity', 'count_thresh_X', ...
    """
    if data_vectors is None or data_vectors.shape[0] == 0:
        print("Error: No data vectors provided for SRM sweep.")
        return None

    N, D = data_vectors.shape
    print(f"\nRunning SRM sweep on {N} vectors in {D}D space.")
    print(f"Plane defined by neurons: {neuron_a_idx} and {neuron_b_idx}")
    print(f"Testing thresholds: {sorted(thresholds)}") # Show sorted thresholds
    print(f"Sweeping {num_angles} angles (0 to 360 degrees).")

    # --- Normalize data vectors once ---
    norms = np.linalg.norm(data_vectors, axis=1, keepdims=True)
    zero_norm_mask = (norms == 0)
    if np.any(zero_norm_mask):
        print(f"Warning: Found {np.sum(zero_norm_mask)} zero-norm vectors. These will have zero similarity.")
        norms[zero_norm_mask] = 1.0
    normalized_data_vectors = data_vectors / norms

    # --- Define basis vectors ---
    e_A = np.zeros(D); e_A[neuron_a_idx] = 1.0
    e_B = np.zeros(D); e_B[neuron_b_idx] = 1.0
    if not (0 <= neuron_a_idx < D and 0 <= neuron_b_idx < D):
        print(f"Error: Neuron indices ({neuron_a_idx}, {neuron_b_idx}) out of bounds for dimension {D}.")
        return None

    # --- Perform the sweep ---
    results_list = []
    angles_deg = np.linspace(0, 360, num_angles, endpoint=False)

    for angle_deg in tqdm(angles_deg, desc="SRM Sweep Multi-Threshold"):
        angle_rad = np.radians(angle_deg)
        spotlight_vec = np.cos(angle_rad) * e_A + np.sin(angle_rad) * e_B
        spotlight_norm = np.linalg.norm(spotlight_vec)
        if spotlight_norm == 0: continue
        normalized_spotlight_vec = spotlight_vec / spotlight_norm

        # Calculate similarities once per angle
        similarities = normalized_data_vectors @ normalized_spotlight_vec

        # Calculate metrics for this angle
        angle_results = {"angle_deg": angle_deg}
        angle_results["mean_similarity"] = np.mean(similarities)

        # Calculate count for each threshold
        for thresh in thresholds:
            count_col_name = f"count_thresh_{thresh}" # Column name for DataFrame
            angle_results[count_col_name] = np.sum(similarities > thresh)

        results_list.append(angle_results)

    print("SRM sweep complete.")
    return pd.DataFrame(results_list)

def plot_srm_results_multi(results_df, thresholds, neuron_a_idx, neuron_b_idx, plot_filename):
    """Plots the multi-threshold SRM sweep results."""
    if results_df is None or results_df.empty:
        print("No results to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7)) # Wider plot

    # Define colors for threshold lines using a colormap
    colors = cm.viridis(np.linspace(0, 0.9, len(thresholds))) # Use viridis map, avoid yellow end

    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Count Above Threshold')
    ax1.grid(True, axis='x', linestyle=':')

    # Plot count for each threshold on left axis (ax1)
    sorted_thresholds = sorted(thresholds, reverse=True) # Plot higher thresholds first
    for i, thresh in enumerate(sorted_thresholds):
        col_name = f"count_thresh_{thresh}"
        ax1.plot(results_df['angle_deg'], results_df[col_name],
                 color=colors[i], marker='.', markersize=4, linestyle='-',
                 label=f'Count > {thresh}')

    ax1.tick_params(axis='y') # No specific color needed if black is okay
    ax1.legend(loc='upper left', title="Threshold Counts")


    # Plot mean similarity on right axis (ax2)
    ax2 = ax1.twinx()
    color_mean = 'tab:grey' #'tab:blue'
    ax2.set_ylabel('Mean Cosine Similarity', color=color_mean)
    ax2.plot(results_df['angle_deg'], results_df['mean_similarity'],
             color=color_mean, marker=None, linestyle='--', linewidth=2,
             label='Mean Similarity')
    ax2.tick_params(axis='y', labelcolor=color_mean)
    ax2.legend(loc='upper right')


    fig.tight_layout()
    plt.title(f'SRM Sweep: Plane ({neuron_a_idx}, {neuron_b_idx}) in {DIMENSION}D Space (Multi-Threshold)')
    plt.xticks(np.arange(0, 361, 45))

    try:
        plt.savefig(plot_filename)
        print(f"Saved combined SRM plot to: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot to {plot_filename}: {e}")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run SRM sweep on {DIMENSION}D vectors for multiple thresholds.")
    parser.add_argument("--input_files", type=str, nargs='+', required=True,
                        help="Paths to the input .npz files containing mean vectors.")
    parser.add_argument("--neuron_a", type=int, required=True, help="Index of the first neuron for the plane.")
    parser.add_argument("--neuron_b", type=int, required=True, help="Index of the second neuron for the plane.")
    parser.add_argument("--output_dir_prefix", type=str, required=True,
                        help="Prefix for the output directory name (e.g., srm_intervention_373_2202). Timestamp will be added.")
    parser.add_argument("--thresholds", type=float, nargs='+', required=True,
                        help="List of cosine similarity thresholds to test (e.g., 0.7 0.5 0.3 0.1).")
    parser.add_argument("--num_angles", type=int, default=72, # 5 degree steps
                        help="Number of angles to sweep between 0 and 360 degrees.")
    parser.add_argument("--save_csv", action='store_true',
                        help="Save the sweep results data to a CSV file.")

    args = parser.parse_args()

    # --- Create Unique Output Directory ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"{args.output_dir_prefix}_N{args.neuron_a}_N{args.neuron_b}_A{args.num_angles}_{timestamp}"

    try:
        os.makedirs(output_dir_name, exist_ok=True)
        print(f"Created output directory: {output_dir_name}")
    except OSError as e:
        print(f"Error creating output directory '{output_dir_name}': {e}")
        traceback.print_exc()
        exit(1)

    # --- Prepare Output Paths inside the directory ---
    csv_filename = "srm_multi_threshold_data.csv"
    csv_path = os.path.join(output_dir_name, csv_filename)
    plot_filename = "srm_multi_threshold_plot.png"
    plot_path = os.path.join(output_dir_name, plot_filename)
    metadata_filename = "run_metadata.json"
    metadata_path = os.path.join(output_dir_name, metadata_filename)

    # --- Load and combine vectors ---
    print("--- Loading Input Vectors ---")
    all_vecs_list = []
    valid_load = True
    total_loaded_count = 0
    input_file_basenames = [os.path.basename(f) for f in args.input_files] # For metadata
    for file_path in args.input_files:
        vecs = load_vectors_from_npz(file_path, DIMENSION)
        if vecs is not None and vecs.shape[0] > 0:
            all_vecs_list.append(vecs)
            print(f"Loaded {vecs.shape[0]} vectors from {os.path.basename(file_path)}")
            total_loaded_count += vecs.shape[0]
        else:
            # valid_load = False # Allow partial success maybe?
            print(f"Warning: Failed to load valid vectors from {file_path}.")
            # break # Stop if any file fails? Or continue? Continuing for now.

    if all_vecs_list:
        combined_vectors = np.concatenate(all_vecs_list, axis=0)
        print(f"\nTotal vectors loaded for analysis: {combined_vectors.shape[0]}")

        # --- Save Metadata ---
        run_metadata = {
            "script_name": os.path.basename(__file__),
            "neuron_a": args.neuron_a,
            "neuron_b": args.neuron_b,
            "dimension": DIMENSION,
            "input_files": input_file_basenames, # Just basenames
            "full_input_paths": args.input_files, # Full paths for reproducibility if needed
            "total_vectors_loaded": combined_vectors.shape[0],
            "tested_thresholds": sorted(args.thresholds),
            "num_angles": args.num_angles,
            "timestamp": timestamp,
            "output_directory": output_dir_name,
            "csv_file": csv_filename,
            "plot_file": plot_filename,
            "metadata_file": metadata_filename
        }
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f_meta:
                json.dump(run_metadata, f_meta, indent=4)
            print(f"Saved run metadata to: {metadata_path}")
        except Exception as e:
            print(f"Warning: Could not save metadata file '{metadata_path}': {e}")

        # --- Run SRM Sweep ---
        srm_results_df = run_srm_sweep_multi(
            data_vectors=combined_vectors,
            neuron_a_idx=args.neuron_a,
            neuron_b_idx=args.neuron_b,
            thresholds=args.thresholds,
            num_angles=args.num_angles
        )

        # --- Plot and Save ---
        if srm_results_df is not None:
            plot_srm_results_multi(
                results_df=srm_results_df,
                thresholds=args.thresholds,
                neuron_a_idx=args.neuron_a,
                neuron_b_idx=args.neuron_b,
                plot_filename=plot_path # Pass full path
            )

            if args.save_csv:
                try:
                    srm_results_df.to_csv(csv_path, index=False)
                    print(f"Saved SRM data to: {csv_path}")
                except Exception as e:
                    print(f"Error saving data to {csv_path}: {e}")
        else:
            print("SRM analysis failed or produced no results.")

    else:
        print("Failed to load sufficient data for SRM analysis. Exiting.")

    print(f"\nScript finished. Results are in directory: {output_dir_name}")

# --- END OF FILE run_srm_sweep_multi_threshold.py ---