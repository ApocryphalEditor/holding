# --- START OF FILE run_srm_sweep.py ---

import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import traceback

# Constants
DIMENSION = 3072 # Expected dimension of the vectors

def load_vectors_from_npz(file_path, expected_dim):
    """Loads vectors from an .npz file, validating their shape."""
    vectors = []
    print(f"Loading vectors from: {file_path}")
    try:
        with np.load(file_path, allow_pickle=True) as loaded_data:
            keys = list(loaded_data.files)
            print(f"Found {len(keys)} keys.")
            for key in tqdm(keys, desc=f"Loading {os.path.basename(file_path)}", leave=False):
                try:
                    vec = loaded_data[key]
                    if not isinstance(vec, np.ndarray):
                        print(f"\nWarning: Skipping key '{key}'. Value is not a NumPy array (type: {type(vec)}).")
                        continue
                    if vec.shape == (expected_dim,):
                        vectors.append(vec)
                    else:
                         print(f"\nWarning: Skipping key '{key}'. Expected shape ({expected_dim},), got {vec.shape}.")
                except Exception as e:
                    print(f"\nError processing key '{key}' in {file_path}: {e}")
                    # traceback.print_exc() # Uncomment for more detail
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path}': {e}")
        traceback.print_exc()
        return None

    if not vectors:
        print(f"Warning: No valid vectors loaded from {file_path}.")
        return None

    print(f"Successfully loaded {len(vectors)} vectors with dimension {expected_dim} from {file_path}.")
    return np.array(vectors) # Shape: [num_vectors, dimension]

def run_srm_sweep(data_vectors, neuron_a_idx, neuron_b_idx, num_angles=360, similarity_threshold=0.95):
    """
    Performs the SRM sweep analysis.

    Args:
        data_vectors (np.ndarray): Array of shape [N, D] containing data vectors.
        neuron_a_idx (int): Index of the first neuron defining the plane.
        neuron_b_idx (int): Index of the second neuron defining the plane.
        num_angles (int): Number of angles to sweep (0 to 360 degrees).
        similarity_threshold (float): Threshold for counting high-alignment vectors.

    Returns:
        pd.DataFrame: DataFrame containing results for each angle.
    """
    if data_vectors is None or data_vectors.shape[0] == 0:
        print("Error: No data vectors provided for SRM sweep.")
        return None

    N, D = data_vectors.shape
    print(f"\nRunning SRM sweep on {N} vectors in {D}D space.")
    print(f"Plane defined by neurons: {neuron_a_idx} and {neuron_b_idx}")
    print(f"Sweeping {num_angles} angles (0 to 360 degrees).")
    print(f"Similarity threshold for count: {similarity_threshold}")

    # --- Normalize data vectors once for efficiency ---
    norms = np.linalg.norm(data_vectors, axis=1, keepdims=True)
    # Handle potential zero vectors
    zero_norm_mask = (norms == 0)
    if np.any(zero_norm_mask):
        print(f"Warning: Found {np.sum(zero_norm_mask)} zero-norm vectors. These will have zero similarity.")
        # Avoid division by zero; their norm remains 0, similarity will be 0.
        norms[zero_norm_mask] = 1.0 # Replace norm with 1 to avoid NaN, dot product will still be 0.

    normalized_data_vectors = data_vectors / norms

    # --- Define basis vectors for the plane ---
    e_A = np.zeros(D)
    e_B = np.zeros(D)
    if 0 <= neuron_a_idx < D and 0 <= neuron_b_idx < D:
        e_A[neuron_a_idx] = 1.0
        e_B[neuron_b_idx] = 1.0
    else:
        print(f"Error: Neuron indices ({neuron_a_idx}, {neuron_b_idx}) out of bounds for dimension {D}.")
        return None

    # --- Perform the sweep ---
    results = []
    angles_deg = np.linspace(0, 360, num_angles, endpoint=False) # Exclude 360 if 0 is included

    for angle_deg in tqdm(angles_deg, desc="SRM Sweep"):
        angle_rad = np.radians(angle_deg)

        # Construct spotlight vector
        spotlight_vec = np.cos(angle_rad) * e_A + np.sin(angle_rad) * e_B

        # Normalize spotlight vector (should be length 1 already for standard basis, but good practice)
        spotlight_norm = np.linalg.norm(spotlight_vec)
        if spotlight_norm == 0: continue # Should not happen with standard basis
        normalized_spotlight_vec = spotlight_vec / spotlight_norm

        # Calculate cosine similarities (dot product of normalized vectors)
        # Shape: (N,)
        similarities = normalized_data_vectors @ normalized_spotlight_vec # Efficient batch dot product

        # Calculate metrics
        count_above_threshold = np.sum(similarities > similarity_threshold)
        mean_similarity = np.mean(similarities)

        results.append({
            "angle_deg": angle_deg,
            "count_above_threshold": count_above_threshold,
            "mean_similarity": mean_similarity,
            # Optionally store distribution later: 'similarities': similarities
        })

    print("SRM sweep complete.")
    return pd.DataFrame(results)

def plot_srm_results(results_df, neuron_a_idx, neuron_b_idx, similarity_threshold, output_prefix):
    """Plots the SRM sweep results."""
    if results_df is None or results_df.empty:
        print("No results to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = 'tab:red'
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel(f'Count (Cos Sim > {similarity_threshold})', color=color1)
    ax1.plot(results_df['angle_deg'], results_df['count_above_threshold'], color=color1, marker='.', linestyle='-', label=f'Count > {similarity_threshold}')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, axis='x', linestyle=':')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel('Mean Cosine Similarity', color=color2)
    ax2.plot(results_df['angle_deg'], results_df['mean_similarity'], color=color2, marker='.', linestyle='--', label='Mean Similarity')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.title(f'SRM Sweep: Plane ({neuron_a_idx}, {neuron_b_idx}) in {DIMENSION}D Space')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xticks(np.arange(0, 361, 45)) # Set x-axis ticks every 45 degrees

    plot_filename = f"{output_prefix}_srm_plot_{neuron_a_idx}_{neuron_b_idx}.png"
    try:
        plt.savefig(plot_filename)
        print(f"Saved SRM plot to: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot to {plot_filename}: {e}")
    # plt.show() # Uncomment to display plot interactively

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run SRM sweep on {DIMENSION}D vectors.")
    parser.add_argument("--input_files", type=str, nargs='+', required=True,
                        help="Paths to the input .npz files containing mean vectors (e.g., results_v3_mean.npz results_v4_mean.npz).")
    parser.add_argument("--neuron_a", type=int, required=True, help="Index of the first neuron for the plane.")
    parser.add_argument("--neuron_b", type=int, required=True, help="Index of the second neuron for the plane.")
    parser.add_argument("--output_prefix", type=str, default="srm_sweep_results",
                        help="Prefix for output files (plot PNG, optional CSV).")
    parser.add_argument("--num_angles", type=int, default=72, # Defaulting to 5 degree steps
                        help="Number of angles to sweep between 0 and 360 degrees.")
    parser.add_argument("--threshold", type=float, default=0.90, # Lowered default threshold
                        help="Cosine similarity threshold for counting vectors.")
    parser.add_argument("--save_csv", action='store_true',
                        help="Save the sweep results data to a CSV file.")

    args = parser.parse_args()

    # --- Load and combine vectors ---
    all_vecs_list = []
    valid_load = True
    for file_path in args.input_files:
        vecs = load_vectors_from_npz(file_path, DIMENSION)
        if vecs is not None:
            all_vecs_list.append(vecs)
        else:
            valid_load = False
            print(f"Failed to load vectors from {file_path}. Aborting.")
            break # Stop if any file fails

    if valid_load and all_vecs_list:
        combined_vectors = np.concatenate(all_vecs_list, axis=0)
        print(f"\nTotal vectors loaded and combined: {combined_vectors.shape[0]}")

        # --- Run SRM Sweep ---
        srm_results_df = run_srm_sweep(
            data_vectors=combined_vectors,
            neuron_a_idx=args.neuron_a,
            neuron_b_idx=args.neuron_b,
            num_angles=args.num_angles,
            similarity_threshold=args.threshold
        )

        # --- Plot and Save ---
        if srm_results_df is not None:
            plot_srm_results(
                results_df=srm_results_df,
                neuron_a_idx=args.neuron_a,
                neuron_b_idx=args.neuron_b,
                similarity_threshold=args.threshold,
                output_prefix=args.output_prefix
            )

            if args.save_csv:
                csv_filename = f"{args.output_prefix}_srm_data_{args.neuron_a}_{args.neuron_b}.csv"
                try:
                    srm_results_df.to_csv(csv_filename, index=False)
                    print(f"Saved SRM data to: {csv_filename}")
                except Exception as e:
                    print(f"Error saving data to {csv_filename}: {e}")
        else:
            print("SRM analysis failed or produced no results.")

    else:
        print("Failed to load sufficient data for SRM analysis. Exiting.")

    print("\nScript finished.")

# --- END OF FILE run_srm_sweep.py ---