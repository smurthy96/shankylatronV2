import os
import pandas as pd
from g2p_en import G2p
from g2pM import G2pM
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def initialize_g2p():
    """Initialize G2P instances for global use in workers."""
    global g2p_en, g2p_cn
    g2p_en = G2p()
    g2p_cn = G2pM()


def process_source_row(row):
    """
    Process a single row to generate phonemes for the source (English) sentence.
    """
    index, row = row
    try:
        sentence = row['sentence_source']
        phonemes = g2p_en(sentence)
        phoneme_str = ' '.join(phonemes)
        return {'index': index, 'phonemes': phoneme_str}
    except Exception as e:
        print(f"Error processing source sentence at index {index}: {e}")
        return {'index': index, 'phonemes': ''}


def process_target_row(row):
    """
    Process a single row to generate phonemes (pinyin) for the target (Chinese) sentence.
    """
    index, row = row
    try:
        sentence = row['sentence_target']
        pinyin = g2p_cn(sentence)
        pinyin_str = ' '.join(pinyin)
        return {'index': index, 'phonemes': pinyin_str}
    except Exception as e:
        print(f"Error processing target sentence at index {index}: {e}")
        return {'index': index, 'phonemes': ''}


def align_and_merge_files(tsv_file_source, tsv_file_target):
    """
    Align source and target TSV files based on sentence ID and ensure equal length.

    Args:
        tsv_file_source (str): Path to the source TSV file.
        tsv_file_target (str): Path to the target TSV file.

    Returns:
        DataFrame: The merged and aligned DataFrame.
    """
    data_source = pd.read_csv(tsv_file_source, sep='\t', low_memory=False)
    data_target = pd.read_csv(tsv_file_target, sep='\t', low_memory=False)

    # Ensure dataset alignment
    if 'sentence_id' in data_source.columns and 'sentence_id' in data_target.columns:
        merged_data_out = pd.merge(data_source, data_target, on='sentence_id', suffixes=('_source', '_target'))
    else:
        # Ensure indices are aligned if no sentence_id column exists
        merged_data_out = pd.concat([data_source.reset_index(drop=True), data_target.reset_index(drop=True)], axis=1)

    return merged_data_out


def generate_combined_phoneme_file(merged_data, combined_output_path):
    """
    Generate a single TSV file containing both source phonemes and target phonemes.

    Args:
        merged_data (DataFrame): Aligned merged data.
        combined_output_path (str): Path to save the combined TSV file.
    """
    # Create lists of rows for source and target processing
    source_rows = [(idx, row) for idx, row in merged_data.iterrows()]
    target_rows = [(idx, row) for idx, row in merged_data.iterrows()]

    with Pool(processes=cpu_count(), initializer=initialize_g2p) as pool:
        # Process source sentences with a progress bar
        source_results = []
        for res in tqdm(pool.imap(process_source_row, source_rows), total=len(source_rows), desc="Processing source rows"):
            source_results.append(res)

        # Process target sentences with a progress bar
        target_results = []
        for res in tqdm(pool.imap(process_target_row, target_rows), total=len(target_rows), desc="Processing target rows"):
            target_results.append(res)

    # Convert results into dictionaries keyed by index
    source_phoneme_dict = {res['index']: res['phonemes'] for res in source_results if res}
    target_phoneme_dict = {res['index']: res['phonemes'] for res in target_results if res}

    # Add new columns to the merged_data
    merged_data['source_phonemes'] = merged_data.index.map(source_phoneme_dict.get)
    merged_data['target_phonemes'] = merged_data.index.map(target_phoneme_dict.get)

    # Save the combined result to a single TSV file
    merged_data.to_csv(combined_output_path, sep='\t', index=False)


if __name__ == '__main__':
    # Define paths for input and output files
    tsv_file_source = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_gendered.tsv"
    tsv_file_target = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_translated.tsv"
    combined_output_path = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_combined_phonemes.tsv"

    print("Aligning and merging source and target files...")
    merged_data = align_and_merge_files(tsv_file_source, tsv_file_target)

    print("Generating combined phoneme file...")
    generate_combined_phoneme_file(merged_data, combined_output_path)

    print("Combined phoneme generation completed.")
