import pandas as pd

# Path to the combined TSV file
tsv_file = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_combined_phonemes.tsv"

# Read the combined TSV file
df = pd.read_csv(tsv_file, sep='\t', low_memory=False)

# Extract source and target phoneme strings and tokenize them
source_phonemes_list = df['source_phonemes'].apply(lambda x: x.split())
target_phonemes_list = df['target_phonemes'].apply(lambda x: x.split())

# Build a vocabulary from all unique tokens in both source and target
all_tokens = set()
for tokens in source_phonemes_list:
    all_tokens.update(tokens)
for tokens in target_phonemes_list:
    all_tokens.update(tokens)

# Create a mapping from token to index
# Start indexing from 1 for convenience (0 can be reserved for PAD if needed)
vocab = {token: i for i, token in enumerate(sorted(all_tokens), start=1)}

# Convert source phoneme sequences to indices
source_indices = source_phonemes_list.apply(lambda tokens: [vocab[t] for t in tokens])
# Convert target phoneme sequences to indices
target_indices = target_phonemes_list.apply(lambda tokens: [vocab[t] for t in tokens])

# Compute max sequence length for source and target
max_source_len = source_indices.apply(len).max()
max_target_len = target_indices.apply(len).max()

# Compute max token ID used in source and target sequences
max_token_id_in_source = source_indices.apply(lambda seq: max(seq) if len(seq) > 0 else 0).max()
max_token_id_in_target = target_indices.apply(lambda seq: max(seq) if len(seq) > 0 else 0).max()

# Print the results
print("Vocabulary size:", len(vocab))
print("Max source length (in tokens):", max_source_len)
print("Max target length (in tokens):", max_target_len)
print("Max token ID in source sequences:", max_token_id_in_source)
print("Max token ID in target sequences:", max_token_id_in_target)
