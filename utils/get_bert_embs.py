import json
import os
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import re

# --- Configuration ---
MODEL_NAME = "bert-base-uncased"
KEYS_TO_EMBED = ["recaption", "open_prompt"]


def sanitize_filename(text, max_len=100):
    """
    Sanitizes a string to be used as a filename and truncates it.
    Removes invalid characters and replaces spaces with underscores.
    """
    # Remove invalid filename characters (Windows/Linux/Mac)
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", text)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Truncate to max_len
    return sanitized[:max_len]


def get_bert_embedding(text, model, tokenizer, device):
    """
    Generates BERT embedding for the given text using mean pooling.
    """
    # Tokenize input text
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    ).to(device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean Pooling: Take the average of all token embeddings in the last hidden layer,
    # considering the attention mask to ignore padding tokens.
    last_hidden_states = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    )
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)  # Avoid division by zero
    mean_pooled_embedding = sum_embeddings / sum_mask

    # Move embedding to CPU and convert to numpy array
    return mean_pooled_embedding.cpu().numpy().squeeze()


def process_directory(input_dir_path, output_dir_path):
    """
    Processes all JSON files in the input directory, computes embeddings,
    and saves them to the output directory.
    """
    input_dir = Path(input_dir_path)
    output_dir = Path(output_dir_path)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load BERT tokenizer and model
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loading model: {MODEL_NAME}...")
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()  # Set model to evaluation mode

    print(f"\nProcessing files in: {input_dir}")

    # Iterate through files in the input directory
    for json_file_path in input_dir.glob("*.json"):
        print(f"  Processing {json_file_path.name}...")
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                for key in KEYS_TO_EMBED:
                    if key in item and isinstance(item[key], str) and item[key].strip():
                        text_content = item[key]

                        # --- Filename Generation ---
                        # Original request: "<language instr>.npy"
                        # This is potentially problematic due to length and invalid chars.
                        # Using a sanitized and truncated version:
                        base_filename = sanitize_filename(text_content)
                        if not base_filename:  # Handle empty strings after sanitization
                            print(
                                f"    Skipping key '{key}' due to empty content after sanitization."
                            )
                            continue
                        output_filename = f"{base_filename}.npy"

                        # # --- Alternative Filename Structure (Safer) ---
                        # # Uncomment this block and comment the one above for safer filenames
                        # # like 'your_json_file_recaption.npy'
                        # file_stem = json_file_path.stem # Filename without extension
                        # safe_key = key.replace(' ', '_') # Replace spaces in key
                        # output_filename = f"{file_stem}_{safe_key}.npy"
                        # # --- End Alternative ---

                        output_filepath = output_dir / output_filename

                        if output_filepath.exists():
                            print(
                                f"    Embedding for key '{key}' already exists: {output_filename}. Skipping."
                            )
                            continue

                        # Generate embedding
                        embedding = get_bert_embedding(
                            text_content, model, tokenizer, device
                        )

                        # Save embedding as .npy file
                        np.save(output_filepath, embedding)
                        print(f"    Saved embedding for key '{key}' to: {output_filename}")

                    else:
                        print(
                            f"    Skipping key '{key}': not found, not a string, or empty."
                        )

        except json.JSONDecodeError:
            print(
                f"  Error: Could not decode JSON from {json_file_path.name}. Skipping."
            )
        except IOError as e:
            print(f"  Error reading file {json_file_path.name}: {e}. Skipping.")
        except Exception as e:
            print(
                f"  An unexpected error occurred while processing {json_file_path.name}: {e}"
            )

    print("\nProcessing finished.")


import numpy as np
from pathlib import Path
import re


# --- Filename Sanitization (Must be identical to the generation script) ---
def sanitize_filename(text, max_len=100):
    """
    Sanitizes a string to be used as a filename and truncates it.
    Removes invalid characters and replaces spaces with underscores.
    (Should be the same function used when generating embeddings)
    """
    # Remove invalid filename characters (Windows/Linux/Mac)
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", text)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Truncate to max_len
    return sanitized[:max_len]


# --- Embedding Search Function ---
def find_embedding(instruction, embeddings_dir="bert_embs"):
    """
    Searches for a pre-computed BERT embedding corresponding to the instruction.

    Args:
        instruction (str): The natural language instruction string.
        embeddings_dir (str or Path): The directory containing the .npy embedding files.

    Returns:
        numpy.ndarray: The loaded BERT embedding if found, otherwise None.
    """
    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.is_dir():
        print(f"Error: Embeddings directory not found: {embeddings_dir}")
        return None

    # Sanitize the instruction to match the expected filename format
    sanitized_instruction = sanitize_filename(instruction)
    if not sanitized_instruction:
        print(
            f"Warning: Instruction '{instruction}' results in an empty filename after sanitization."
        )
        return None

    expected_filename = f"{sanitized_instruction}.npy"
    embedding_filepath = embeddings_path / expected_filename

    if embedding_filepath.is_file():
        try:
            embedding = np.load(embedding_filepath)
            return embedding
        except Exception as e:
            print(f"Error loading embedding file {embedding_filepath}: {e}")
            return None
    else:
        print(
            f"Embedding file not found for instruction: '{instruction}' (Expected: {expected_filename})"
        )
        # Optional: Implement fuzzy matching or other search strategies here if exact match fails
        return None


# # --- Example Usage ---
# if __name__ == "__main__":
#     # Assume 'bert_embs' directory exists and contains embeddings
#     # Example: bert_embs/This_is_a_recaption_example.npy

#     # 1. Example instruction that should have a corresponding file
#     instruction1 = "This is a recaption example"
#     embedding1 = find_embedding(instruction1, embeddings_dir="bert_embs")

#     if embedding1 is not None:
#         print(f"Successfully loaded embedding for: '{instruction1}'")
#         print(f"Embedding shape: {embedding1.shape}")
#         # print(f"Embedding sample: {embedding1[:5]}...") # Uncomment to see part of the embedding
#     else:
#         print(f"Could not find embedding for: '{instruction1}'")

#     print("-" * 20)

#     # 2. Example instruction that likely doesn't have a corresponding file
#     instruction2 = "A completely different instruction"
#     embedding2 = find_embedding(instruction2, embeddings_dir="bert_embs")

#     if embedding2 is not None:
#         print(f"Successfully loaded embedding for: '{instruction2}'")
#     else:
#         print(f"Could not find embedding for: '{instruction2}' (as expected)")

#     print("-" * 20)

#     # 3. Example showing sanitization
#     instruction3 = 'An open prompt with /\\:*?"<>| characters'
#     # Expected filename: An_open_prompt_with__characters.npy
#     embedding3 = find_embedding(instruction3, embeddings_dir="bert_embs")
#     if embedding3 is not None:
#         print(f"Successfully loaded embedding for: '{instruction3}'")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Compute BERT embeddings for specific keys in JSON files."
#     )
#     parser.add_argument(
#         "--input_dir",
#         type=str,
#         required=True,
#         help="Directory containing the input JSON files.",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="bert_embs",
#         help="Directory to save the output .npy embedding files (default: bert_embs).",
#     )

#     args = parser.parse_args()

#     process_directory(args.input_dir, args.output_dir)
