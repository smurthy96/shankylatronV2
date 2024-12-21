import pandas as pd



# Path to the preprocessed TSV file

tsv_file_phoneme = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_combined_phonemes.tsv"



# Read the preprocessed phoneme data

data = pd.read_csv(tsv_file_phoneme, sep='\t', low_memory=False)



# Build global vocabulary

global_vocab = {'<PAD>': 0, '<UNK>': 1}  # Special tokens

all_tokens = set()

for phonemes in data['source_phonemes']:  # Collect all unique tokens from source phonemes

    all_tokens.update(phonemes.split())

for phonemes in data['target_phonemes']:  # Collect all unique tokens from target phonemes

    all_tokens.update(phonemes.split())



# Add tokens to the global vocabulary

global_vocab.update({token: idx for idx, token in enumerate(sorted(all_tokens), start=2)})



# Calculate max and min indices

max_class = max(global_vocab.values())

min_class = min(global_vocab.values())



print(max_class, min_class)