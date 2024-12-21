import os
import pandas as pd
from g2p_en import G2p
from g2pM import G2pM

def align_and_merge_files(tsv_file_source, tsv_file_target, aligned_output_path):
    """
    Align source and target TSV files based on sentence ID and ensure equal length.

    Args:
        tsv_file_source (str): Path to the source TSV file.
        tsv_file_target (str): Path to the target TSV file.
        aligned_output_path (str): Path to save the aligned merged TSV file.
    """
    data_source = pd.read_csv(tsv_file_source, sep='\t')
    data_target = pd.read_csv(tsv_file_target, sep='\t')

    # Ensure dataset alignment
    if 'sentence_id' in data_source.columns and 'sentence_id' in data_target.columns:
        merged_data = pd.merge(data_source, data_target, on='sentence_id', suffixes=('_source', '_target'))
        min_length = min(len(data_source), len(data_target))
        merged_data = merged_data.head(min_length)
    else:
        raise ValueError("Both files must have a 'sentence_id' column for alignment.")

    # Save aligned data
    merged_data.to_csv(aligned_output_path, sep='\t', index=False)
    return merged_data

def generate_phoneme_files(merged_data, phoneme_output_source, phoneme_output_target):
    """
    Generate phoneme and pinyin TSV files for source and target data.

    Args:
        merged_data (DataFrame): Aligned merged data.
        phoneme_output_source (str): Path to save the source phoneme TSV file.
        phoneme_output_target (str): Path to save the target phoneme TSV file.
    """
    g2p_en = G2p()
    g2p_cn = G2pM()

    source_phonemes = []
    target_phonemes = []

    for idx, row in merged_data.iterrows():
        try:
            # Generate phonemes for source sentence
            source_sentence = row['sentence_source']
            source_phonemes.append({**row.to_dict(),
                                    'phonemes': ' '.join(g2p_en(source_sentence))})

            # Generate Pinyin for target sentence
            target_sentence = row['sentence_target']
            target_phonemes.append({**row.to_dict(),
                                    'phonemes': ' '.join(g2p_cn(target_sentence))})

        except Exception as e:
            print(f"Error processing sentence at index {idx}: {e}")

    # Save phoneme files
    pd.DataFrame(source_phonemes).to_csv(phoneme_output_source, sep='\t', index=False)
    pd.DataFrame(target_phonemes).to_csv(phoneme_output_target, sep='\t', index=False)

if __name__ == '__main__':
    # Define paths for input and output files
    tsv_file_source = "path/to/source_file.tsv"
    tsv_file_target = "path/to/target_file.tsv"
    aligned_output_path = "path/to/aligned_file.tsv"
    phoneme_output_source = "path/to/source_phonemes.tsv"
    phoneme_output_target = "path/to/target_phonemes.tsv"

    # Align and merge source and target files
    merged_data = align_and_merge_files(tsv_file_source, tsv_file_target, aligned_output_path)

    # Generate phoneme files
    generate_phoneme_files(merged_data, phoneme_output_source, phoneme_output_target)

    print("Alignment and phoneme generation completed.")
