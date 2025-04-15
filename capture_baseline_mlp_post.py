# --- START OF FILE capture_baseline_mlp_post.py ---

import torch
import argparse
from transformer_lens import HookedTransformer, utils
import warnings
import os
import sys
from tqdm import tqdm
import numpy as np
import traceback # For detailed error printing
import datetime
import json # For saving metadata

# Argument parsing
parser = argparse.ArgumentParser(description="Baseline Capture: Run GPT-2 Small on prompts, capturing MLP post-activations without intervention into a timestamped folder.")
parser.add_argument("--prompt_file", type=str, required=True, help="Path to a text file containing prompts (one per line).")
parser.add_argument("--output_dir_prefix", type=str, required=True, help="Prefix for the output directory name (e.g., results_v3). Specifics & timestamp will be added.")
parser.add_argument("--generate_length", type=int, default=50, help="Number of new tokens to generate for each prompt.")
parser.add_argument("--top_k", type=int, default=None, help="If set, use top-k sampling during generation. Set to 0 for greedy.")
parser.add_argument("--layer", type=int, default=11, help="Layer index (0-based) to capture MLP activations from.")
args = parser.parse_args()

# Configuration
TARGET_LAYER = args.layer

# --- Create Unique Output Directory ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Extract the base name of the prompt file for the directory name
prompt_file_basename = os.path.splitext(os.path.basename(args.prompt_file))[0]
output_dir_name = f"{args.output_dir_prefix}_baseline_L{TARGET_LAYER}_len{args.generate_length}_{prompt_file_basename}_{timestamp}"

try:
    os.makedirs(output_dir_name, exist_ok=True)
    print(f"Created output directory: {output_dir_name}")
except OSError as e:
    print(f"Error creating output directory '{output_dir_name}': {e}", file=sys.stderr)
    traceback.print_exc()
    exit(1)

# --- Load Model ---
print("Loading GPT-2 Small model...")
try:
    model = HookedTransformer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Using device: {device}")
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        print("Setting pad token to EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Define hook point name clearly
    capture_hook_point_name = f"blocks.{TARGET_LAYER}.mlp.hook_post" # Direct MLP output
    DIMENSION = model.cfg.d_mlp # Should be 3072 for gpt2-small

except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    traceback.print_exc()
    # Clean up created directory if model loading fails? Maybe not necessary.
    exit(1)

# --- Load Prompts ---
try:
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
except Exception as e:
    print(f"Error reading prompt file '{args.prompt_file}': {e}", file=sys.stderr)
    traceback.print_exc()
    exit(1)

# --- Prepare Output Paths inside the directory ---
# Vector file name is now simpler as the directory holds the context
vector_filename = f"baseline_mean_vectors_{DIMENSION}d.npz"
vector_path = os.path.join(output_dir_name, vector_filename)
# Log file path
log_filename = "run_log.md"
log_file_path = os.path.join(output_dir_name, log_filename)
# Metadata file path
metadata_filename = "run_metadata.json"
metadata_path = os.path.join(output_dir_name, metadata_filename)

all_vectors = {} # Collect vectors in memory first

# --- Save Metadata ---
run_metadata = {
    "script_name": os.path.basename(__file__),
    "model_name": "gpt2",
    "target_layer": TARGET_LAYER,
    "capture_hook": capture_hook_point_name,
    "activation_dimension": DIMENSION,
    "prompt_file": args.prompt_file,
    "generate_length": args.generate_length,
    "top_k": args.top_k if args.top_k is not None else "greedy_or_default",
    "timestamp": timestamp,
    "output_directory": output_dir_name,
    "vector_file": vector_filename,
    "log_file": log_filename,
    "metadata_file": metadata_filename,
    "device": device,
    "num_prompts": len(prompts)
}
try:
    with open(metadata_path, 'w', encoding='utf-8') as f_meta:
        json.dump(run_metadata, f_meta, indent=4)
    print(f"Saved run metadata to: {metadata_path}")
except Exception as e:
    print(f"Warning: Could not save metadata file '{metadata_path}': {e}")


# --- Main Processing ---
print(f"Starting baseline runs (no intervention).")
print(f"MLP Post-Activation vectors (Layer {TARGET_LAYER}, Dim {DIMENSION}) will be averaged.")
print(f"Results will be saved in: {output_dir_name}")


try:
    with open(log_file_path, 'w', encoding='utf-8') as logfile:
        # Write header to log file
        logfile.write(f"# Baseline Activation Capture Log\n\n")
        for key, value in run_metadata.items():
             logfile.write(f"- **{key.replace('_', ' ').title()}**: `{value}`\n")
        logfile.write("\n---\n\n")
        logfile.flush()

        # Disable gradient calculations for efficiency during inference
        with torch.no_grad(), tqdm(total=len(prompts), desc="Baseline Runs") as pbar:
            for i, prompt in enumerate(prompts):
                logfile.write(f"## Prompt {i+1}/{len(prompts)}: `{prompt}`\n\n")
                pbar.set_description(f"Prompt {i+1}")

                try:
                    # --- Tokenize Input ---
                    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                    input_len = input_ids.shape[1]

                    if input_len == 0:
                        logfile.write("```\nError: Empty prompt after tokenization.\n```\n\n")
                        logfile.flush()
                        pbar.update(1)
                        continue # Skip to next prompt

                    # --- Stage 1: Generate Text (NO HOOKS needed for baseline generation) ---
                    output_ids = None
                    result_text = ""

                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore", category=UserWarning)
                         output_ids = model.generate(
                             input_ids,
                             max_new_tokens=args.generate_length,
                             do_sample=(args.top_k is not None and args.top_k > 0),
                             top_k=args.top_k if (args.top_k is not None and args.top_k > 0) else None,
                             eos_token_id=tokenizer.eos_token_id
                             )

                    # --- Decode and Write Generated Text ---
                    generated_len = output_ids.shape[1] - input_len
                    if generated_len > 0:
                         result_text = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
                    else:
                         result_text = "(No new tokens generated or generation failed)"
                         if output_ids.shape[1] == input_len and torch.equal(input_ids, output_ids):
                             result_text = "(Generation stopped immediately, input == output)"

                    logfile.write("```\n" + result_text + "\n```\n\n")
                    logfile.flush()

                    # --- Stage 2: Rerun Forward Pass on Full Sequence to Capture MLP Post-Activations ---
                    if generated_len > 0:
                        # --- Helper function (same as before) ---
                        def get_mlp_post_activations_baseline(current_output_ids):
                            captured_mlp_post_local = None
                            def save_mlp_post_hook_local(activation, hook):
                                nonlocal captured_mlp_post_local
                                captured_mlp_post_local = activation.clone().detach().cpu()
                            fwd_pass_hooks = [(capture_hook_point_name, save_mlp_post_hook_local)]
                            try:
                                with model.hooks(fwd_hooks=fwd_pass_hooks):
                                    model(current_output_ids, return_type=None)
                                return captured_mlp_post_local
                            except Exception as e_inner:
                                print(f"\nError inside get_mlp_post_activations_baseline (Prompt {i}): {e_inner}")
                                traceback.print_exc()
                                return None
                        # --- End helper ---

                        captured_mlp_post_activation = get_mlp_post_activations_baseline(output_ids)

                        # --- Process and Store the Captured Vector ---
                        if captured_mlp_post_activation is not None:
                            generated_vectors_tensor = captured_mlp_post_activation[:, input_len:, :]
                            if generated_vectors_tensor.shape[1] > 0:
                                generated_vectors_np = generated_vectors_tensor.squeeze(0).numpy()
                                mean_vector_np = np.mean(generated_vectors_np, axis=0)

                                if mean_vector_np.shape == (DIMENSION,):
                                    key = f"prompt_{i}"
                                    all_vectors[key] = mean_vector_np
                                else:
                                    print(f"\nError: Unexpected mean vector shape for prompt {i}. Expected ({DIMENSION},), got {mean_vector_np.shape}. Skipping.")
                                    logfile.write("```\nError: Failed to produce correctly shaped mean vector.\n```\n\n")
                            else:
                                 print(f"\nWarning: Sliced MLP post-activation has 0 token length despite generated_len={generated_len} (Prompt {i}). Capture shape: {captured_mlp_post_activation.shape}, input_len: {input_len}.")
                                 logfile.write("```\nWarning: Sliced MLP post-activation had 0 length.\n```\n\n")
                        else:
                            print(f"\nWarning: Failed to capture MLP post-activation for Prompt {i}. Returned None.")
                            logfile.write("```\nError: Failed to capture MLP post-activation vector.\n```\n\n")
                        logfile.flush()
                    else:
                        logfile.write("```\n(No new tokens generated, no activations captured)\n```\n\n")
                        logfile.flush()

                except Exception as e:
                    logfile.write(f"```\nERROR processing prompt {i}: {str(e)}\n```\n\n")
                    logfile.flush()
                    print(f"\n--- ERROR processing Prompt {i} ---")
                    traceback.print_exc()
                    print(f"--- END ERROR ---")

                finally:
                     pbar.update(1)

            logfile.write("---\n\n")
            logfile.flush()

except Exception as e:
    print(f"\n--- FATAL ERROR during main processing loop ---")
    traceback.print_exc()
    print(f"--- END FATAL ERROR ---")
    if 'logfile' in locals() and not logfile.closed:
        logfile.write("\n\n```\nFATAL ERROR occurred during processing. Results may be incomplete.\n```\n")
        logfile.flush()

# --- Final Step: Save all collected vectors ---
final_vector_count = len(all_vectors)
run_metadata["final_vector_count"] = final_vector_count # Add final count to metadata
try:
    # Re-save metadata with final count
    with open(metadata_path, 'w', encoding='utf-8') as f_meta:
        json.dump(run_metadata, f_meta, indent=4)

    if all_vectors:
        print(f"\nSaving {final_vector_count} collected mean vector arrays to {vector_path}...")
        np.savez_compressed(vector_path, **all_vectors)
        print("Vectors saved successfully.")
    else:
        print(f"\nNo vectors were collected or generated to save to {vector_path}.")
except Exception as e:
    print(f"\n--- ERROR saving final vector file {vector_path} or updating metadata ---")
    traceback.print_exc()
    print(f"--- END ERROR ---")

print(f"\nScript finished. Results are in directory: {output_dir_name}")

# --- END OF FILE capture_baseline_mlp_post.py ---