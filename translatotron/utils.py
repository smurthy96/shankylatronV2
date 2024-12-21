import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import os
import IPython.display as ipd
import soundfile as sf
import random
from g2pM import G2pM  # For Chinese phoneme conversion
from g2p_en import G2p  # For English phoneme conversion
from concurrent.futures import ProcessPoolExecutor

class SpeechToSpeechDataset(Dataset):
    def __init__(self,
                 root_dir_source,
                 root_dir_target,
                 tsv_file_source,
                 tsv_file_target,
                 source_lang='en',
                 target_lang='cn',
                 max_audio_length=10):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            csv_file (string): Path to the csv file with annotations.
            split (string): 'train', 'val', or 'test'
            source_lang (string): Source language code
            target_lang (string): Target language code
            max_audio_length (int): Maximum audio length in seconds
        """
        self.root_dir_source = root_dir_source
        self.root_dir_target = root_dir_target
        self.data_source = pd.read_csv(tsv_file_source, sep='\t', low_memory=False)
        self.data_target = pd.read_csv(tsv_file_target, sep='\t', low_memory=False)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_audio_length = max_audio_length
        self.merged_data = None

        # Merge on a unique identifier (if available) or reset indices
        if 'sentence_id' in self.data_source.columns and 'sentence_id' in self.data_target.columns:
            self.merged_data = pd.merge(self.data_source, self.data_target, on='sentence_id', suffixes=('_source', '_target'))
        else:
            # Ensure indices are aligned
            self.merged_data = pd.concat([self.data_source.reset_index(), self.data_target.reset_index()], axis=1)

        # Create dictionaries for text-to-index conversion
        self.source_vocab = self.create_vocabulary(self.merged_data['sentence_source'])
        self.target_vocab = self.create_vocabulary(self.merged_data['sentence_target'])

    def create_vocabulary(self, text_series):
        vocab = set()
        for text in text_series:
            vocab.update(text)
        return {char: (idx + 1) for idx, char in enumerate(sorted(vocab))}

    def text_to_index(self, text, vocab):
        return torch.tensor([vocab[char] for char in text if char in vocab])

    def __len__(self):
        return min(len(self.data_source), len(self.data_target))

    def pad_to_max_length(self, text_indices, max_length):
        # Convert to list if text_indices is a tensor
        if isinstance(text_indices, torch.Tensor):
            text_indices = text_indices.tolist()
        # Perform padding
        if len(text_indices) < max_length:
            return text_indices + [0] * (max_length - len(text_indices))  # Padding value: 0
        return text_indices[:max_length]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load audio files
        source_audio_path = os.path.join(self.root_dir_source, self.merged_data.iloc[idx][f'path_source'])
        target_audio_path = os.path.join(self.root_dir_target, self.merged_data.iloc[idx][f'path_target'])
        source_audio = target_audio = source_text_indices = target_text_indices = None
        standard_sr = 16000
        max_length = int(self.max_audio_length * standard_sr)
        max_text_length = 15201
        try:
            try:
                source_audio, source_sr = torchaudio.load(source_audio_path, format="mp3")

                if source_sr != standard_sr:
                    source_audio = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=standard_sr)(source_audio)

                source_audio = source_audio.mean(dim=0) if source_audio.shape[0] > 1 else source_audio.squeeze(0)

                if source_audio.shape[0] > max_length:
                    source_audio = source_audio[:max_length]
                else:
                    source_audio = torch.nn.functional.pad(source_audio, (0, max_length - source_audio.shape[0]))

                source_text = self.merged_data.iloc[idx][f'sentence_source']

                # Convert text to indices
                source_text_indices = self.text_to_index(source_text, self.source_vocab)

                source_text_indices = torch.tensor(self.pad_to_max_length(source_text_indices, max_text_length))

            except Exception as e:
                print(f"Cannot Load Source", e)
            try:
                target_audio, target_sr = torchaudio.load(target_audio_path, format="wav")

                if target_sr != standard_sr:
                    target_audio = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=standard_sr)(target_audio)

                target_audio = target_audio.mean(dim=0) if target_audio.shape[0] > 1 else target_audio.squeeze(0)

                if target_audio.shape[0] > max_length:
                    target_audio = target_audio[:max_length]
                else:
                    target_audio = torch.nn.functional.pad(target_audio, (0, max_length - target_audio.shape[0]))

                target_text = self.merged_data.iloc[idx][f'sentence_target']

                target_text_indices = self.text_to_index(target_text, self.target_vocab)

                target_text_indices = torch.tensor(self.pad_to_max_length(target_text_indices, max_text_length))
            except Exception as e:
                print(f"Cannot Load Target", e)

        except Exception as e:
            print(f"Error loading file {source_audio_path} target {target_audio_path}: {e}")

        # Handle missing source_audio
        if source_audio is None:
            source_audio = torch.zeros_like(target_audio) if target_audio is not None else torch.zeros((max_length,))

        # Handle missing target_audio
        if target_audio is None:
            target_audio = torch.zeros_like(source_audio) if source_audio is not None else torch.zeros((max_length,))

        # Handle missing source_text_indices
        if source_text_indices is None:
            source_text_indices = torch.zeros_like(
                target_text_indices) if target_text_indices is not None else torch.zeros((max_text_length,))

        # Handle missing target_text_indices
        if target_text_indices is None:
            target_text_indices = torch.zeros_like(
                source_text_indices) if source_text_indices is not None else torch.zeros((max_text_length,))

        return source_audio, target_audio, source_text_indices, target_text_indices


class SpeechToSpeechDataset2(Dataset):
    def __init__(self,
                 root_dir_source,
                 root_dir_target,
                 tsv_file_source,
                 tsv_file_target,
                 source_lang='en',
                 target_lang='cn',
                 max_audio_length=10,
                 concat_aug=True):
        """
        Args:
            root_dir_source (string): Directory with all the source audio files.
            root_dir_target (string): Directory with all the target audio files.
            tsv_file_source (string): Path to the tsv file for source data.
            tsv_file_target (string): Path to the tsv file for target data.
            source_lang (string): Source language code.
            target_lang (string): Target language code.
            max_audio_length (int): Maximum audio length in seconds.
            concat_aug (bool): Whether to perform ConcatAug augmentation.
        """
        self.root_dir_source = root_dir_source
        self.root_dir_target = root_dir_target
        self.data_source = pd.read_csv(tsv_file_source, sep='\t', low_memory=False)
        self.data_target = pd.read_csv(tsv_file_target, sep='\t', low_memory=False)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_audio_length = max_audio_length
        self.concat_aug = concat_aug
        self.merged_data = None

        # Merge on a unique identifier (if available) or reset indices
        if 'sentence_id' in self.data_source.columns and 'sentence_id' in self.data_target.columns:
            self.merged_data = pd.merge(self.data_source, self.data_target, on='sentence_id', suffixes=('_source', '_target'))
        else:
            # Ensure indices are aligned
            self.merged_data = pd.concat([self.data_source.reset_index(), self.data_target.reset_index()], axis=1)

        # Create dictionaries for text-to-index conversion
        self.source_vocab = self.create_vocabulary(self.merged_data['sentence_source'])
        self.target_vocab = self.create_vocabulary(self.merged_data['sentence_target'])

    def create_vocabulary(self, text_series):
        vocab = set()
        for text in text_series:
            vocab.update(text)
        return {char: (idx + 1) for idx, char in enumerate(sorted(vocab))}

    def text_to_index(self, text, vocab):
        return torch.tensor([vocab[char] for char in text if char in vocab])

    def __len__(self):
        return len(self.merged_data)

    def pad_to_max_length(self, text_indices, max_length):
        # Ensure text_indices is a list
        if isinstance(text_indices, torch.Tensor):
            text_indices = text_indices.tolist()

        # Perform padding
        if len(text_indices) < max_length:
            return torch.tensor(text_indices + [0] * (max_length - len(text_indices)), dtype=torch.long)
        return torch.tensor(text_indices[:max_length])

    def _load_and_process_audio(self, audio_path, max_length, standard_sr=16000):
        try:
            audio, sr = torchaudio.load(audio_path)
            if sr != standard_sr:
                audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=standard_sr)(audio)
            audio = audio.mean(dim=0) if audio.shape[0] > 1 else audio.squeeze(0)
            if audio.shape[0] > max_length:
                audio = audio[:max_length]
            else:
                audio = torch.nn.functional.pad(audio, (0, max_length - audio.shape[0]))
            return audio
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return torch.zeros((max_length,))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        standard_sr = 16000
        max_length = int(self.max_audio_length * standard_sr)
        max_text_length = 100

        # Load source and target data
        source_audio_path = os.path.join(self.root_dir_source, self.merged_data.iloc[idx]['path_source'])
        target_audio_path = os.path.join(self.root_dir_target, self.merged_data.iloc[idx]['path_target'])

        source_audio = self._load_and_process_audio(source_audio_path, max_length, standard_sr)
        target_audio = self._load_and_process_audio(target_audio_path, max_length, standard_sr)
        source_text = self.merged_data.iloc[idx]['sentence_source']
        target_text = self.merged_data.iloc[idx]['sentence_target']

        source_text_indices = self.text_to_index(source_text, self.source_vocab)
        target_text_indices = self.text_to_index(target_text, self.target_vocab)

        source_text_indices = torch.tensor(self.pad_to_max_length(source_text_indices, max_text_length))
        target_text_indices = torch.tensor(self.pad_to_max_length(target_text_indices, max_text_length))

        # ConcatAug: Randomly concatenate another sample
        if self.concat_aug and random.random() > 0.5:
            random_idx = random.randint(0, len(self.merged_data) - 1)
            source_audio_aug = self._load_and_process_audio(
                os.path.join(self.root_dir_source, self.merged_data.iloc[random_idx]['path_source']),
                max_length,
                standard_sr
            )
            target_audio_aug = self._load_and_process_audio(
                os.path.join(self.root_dir_target, self.merged_data.iloc[random_idx]['path_target']),
                max_length,
                standard_sr
            )
            source_text_aug = self.merged_data.iloc[random_idx]['sentence_source']
            target_text_aug = self.merged_data.iloc[random_idx]['sentence_target']

            # Concatenate audio and text
            source_audio = torch.cat((source_audio, source_audio_aug), dim=-1)
            target_audio = torch.cat((target_audio, target_audio_aug), dim=-1)
            source_text_indices = torch.cat((source_text_indices, self.text_to_index(source_text_aug, self.source_vocab)), dim=-1)
            target_text_indices = torch.cat((target_text_indices, self.text_to_index(target_text_aug, self.target_vocab)), dim=-1)

            # Ensure lengths after concatenation
            source_audio = source_audio[:max_length]
            target_audio = target_audio[:max_length]
            source_text_indices = torch.tensor(self.pad_to_max_length(source_text_indices.tolist(), max_text_length))
            target_text_indices = torch.tensor(self.pad_to_max_length(target_text_indices.tolist(), max_text_length))

        return source_audio, target_audio, source_text_indices, target_text_indices


class SpeechToSpeechDatasetPhonemePreProcessed(Dataset):
    def __init__(self, root_dir_source, root_dir_target, tsv_file,
                 concat_aug=True, concat_prob=0.5, max_audio_length=10):
        """
        Args:
            root_dir_source: Directory with source audio files.
            root_dir_target: Directory with target audio files.
            tsv_file_phoneme: TSV file with preprocessed phonemes and paths.
            concat_aug: Boolean flag to enable ConcatAug.
            concat_prob: Probability of applying ConcatAug.
            max_audio_length: Maximum audio length in seconds.
        """
        self.standard_sr = 16000
        self.root_dir_source = root_dir_source
        self.root_dir_target = root_dir_target
        self.data = pd.read_csv(tsv_file, sep='\t', low_memory=False)
        self.concat_aug = concat_aug
        self.concat_prob = concat_prob
        self.max_audio_length = max_audio_length

        # Build global vocabulary from phoneme columns
        self.global_vocab = {'<PAD>': 0, '<UNK>': 1}  # Special tokens
        all_tokens = set()
        print("Processing phonemes to global vocab")
        for phonemes in self.data['source_phonemes']:  # Collect all unique tokens from source phonemes
            all_tokens.update(phonemes.split())
        for phonemes in self.data['target_phonemes']:  # Collect all unique tokens from target phonemes
            all_tokens.update(phonemes.split())
        self.global_vocab.update({token: idx for idx, token in enumerate(sorted(all_tokens), start=2)})
        del all_tokens

    def pad_text(self, phoneme_indices, max_length):
        """
        Pads phoneme indices to the predefined maximum length.
        """
        if len(phoneme_indices) < max_length:
            return phoneme_indices + [self.global_vocab['<PAD>']] * (max_length - len(phoneme_indices))
        return phoneme_indices[:max_length]

    def text_to_indices(self, phonemes):
        """
        Convert phonemes to indices using the global vocabulary.
        """
        return [self.global_vocab.get(p, self.global_vocab['<UNK>']) for p in phonemes.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        standard_sr = 16000  # Standard sample rate
        max_length = int(self.max_audio_length * standard_sr)  # Max audio length in samples
        max_text_length = 226  # Predefined max text length

        # File paths
        source_audio_path = os.path.join(self.root_dir_source, self.data.iloc[idx]['path_source'])
        target_audio_path = os.path.join(self.root_dir_target, self.data.iloc[idx]['path_target'])

        try:
            # Load source audio
            source_audio, source_sr = torchaudio.load(source_audio_path)
            if source_sr != standard_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=standard_sr)
                source_audio = resampler(source_audio)
            source_audio = source_audio.mean(dim=0) if source_audio.ndim > 1 else source_audio.squeeze(0)
            if source_audio.shape[0] > max_length:
                source_audio = source_audio[:max_length]
            else:
                source_audio = torch.nn.functional.pad(source_audio, (0, max_length - source_audio.shape[0]))
        except Exception as e:
            print(f"Error loading source audio {source_audio_path}: {e}")
            source_audio = torch.zeros(max_length)

        try:
            # Load target audio
            target_audio, target_sr = torchaudio.load(target_audio_path)
            if target_sr != standard_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=standard_sr)
                target_audio = resampler(target_audio)
            target_audio = target_audio.mean(dim=0) if target_audio.ndim > 1 else target_audio.squeeze(0)
            if target_audio.shape[0] > max_length:
                target_audio = target_audio[:max_length]
            else:
                target_audio = torch.nn.functional.pad(target_audio, (0, max_length - target_audio.shape[0]))
        except Exception as e:
            print(f"Error loading target audio {target_audio_path}: {e}")
            target_audio = torch.zeros(max_length)

        # Convert preprocessed phonemes to indices
        source_phonemes = self.data.iloc[idx]['source_phonemes']
        target_phonemes = self.data.iloc[idx]['target_phonemes']

        source_text_indices = torch.tensor(self.pad_text(self.text_to_indices(source_phonemes), max_text_length),
                                           dtype=torch.long)
        target_text_indices = torch.tensor(self.pad_text(self.text_to_indices(target_phonemes), max_text_length),
                                           dtype=torch.long)

        # Perform ConcatAug augmentation with probability 0.5
        if self.concat_aug and torch.rand(1).item() < self.concat_prob:
            concat_idx = torch.randint(0, len(self.data), (1,)).item()

            # Load additional source audio
            concat_source_audio_path = os.path.join(self.root_dir_source, self.data.iloc[concat_idx]['path_source'])
            try:
                source_audio_aug, source_sr_aug = torchaudio.load(concat_source_audio_path)
                if source_sr_aug != standard_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=source_sr_aug, new_freq=standard_sr)
                    source_audio_aug = resampler(source_audio_aug)
                source_audio_aug = source_audio_aug.mean(dim=0) if source_audio_aug.ndim > 1 else source_audio_aug.squeeze(0)
                source_audio_aug = torch.nn.functional.pad(source_audio_aug, (0, max_length - source_audio_aug.shape[0]))
            except Exception as e:
                print(f"Error loading concatenated source audio {concat_source_audio_path}: {e}")
                source_audio_aug = torch.zeros(max_length)

            # Load additional target audio
            concat_target_audio_path = os.path.join(self.root_dir_target, self.data.iloc[concat_idx]['path_target'])
            try:
                target_audio_aug, target_sr_aug = torchaudio.load(concat_target_audio_path)
                if target_sr_aug != standard_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=target_sr_aug, new_freq=standard_sr)
                    target_audio_aug = resampler(target_audio_aug)
                target_audio_aug = target_audio_aug.mean(dim=0) if target_audio_aug.ndim > 1 else target_audio_aug.squeeze(0)
                target_audio_aug = torch.nn.functional.pad(target_audio_aug, (0, max_length - target_audio_aug.shape[0]))
            except Exception as e:
                print(f"Error loading concatenated target audio {concat_target_audio_path}: {e}")
                target_audio_aug = torch.zeros(max_length)

            # Concatenate audio
            source_audio = torch.cat([source_audio, source_audio_aug], dim=0)[:max_length]
            target_audio = torch.cat([target_audio, target_audio_aug], dim=0)[:max_length]

        return source_audio, target_audio, source_text_indices, target_text_indices


class SpeechToSpeechDatasetPhoneme(Dataset):
    def __init__(self, root_dir_source, root_dir_target, tsv_file_source, tsv_file_target, tsv_file_phoneme,
                 concat_aug=True, concat_prob=0.5, max_audio_length=10):
        """
        Args:
            root_dir_source: Directory with source audio files.
            root_dir_target: Directory with target audio files.
            tsv_file_source: TSV file with source annotations.
            tsv_file_target: TSV file with target annotations.
            concat_aug: Boolean flag to enable ConcatAug.
            concat_prob: Probability of applying ConcatAug.
            max_audio_length: Maximum audio length in seconds.
        """
        self.standard_sr = 16000
        self.root_dir_source = root_dir_source
        self.root_dir_target = root_dir_target
        self.data_source = pd.read_csv(tsv_file_source, sep='\t', low_memory=False)
        self.data_target = pd.read_csv(tsv_file_target, sep='\t', low_memory=False)
        self.data_phoneme = pd.read_csv(tsv_file_phoneme, sep='\t', low_memory=False)
        self.concat_aug = concat_aug
        self.concat_prob = concat_prob
        self.max_audio_length = max_audio_length

        # Ensure dataset alignment
        # Merge on a unique identifier (if available) or reset indices
        if 'sentence_id' in self.data_source.columns and 'sentence_id' in self.data_target.columns:
            self.merged_data = pd.merge(self.data_source, self.data_target, on='sentence_id',
                                        suffixes=('_source', '_target'))
        else:
            # Ensure indices are aligned
            self.merged_data = pd.concat([self.data_source.reset_index(), self.data_target.reset_index()], axis=1)

        self.global_vocab = {'<PAD>': 0, '<UNK>': 1}  # Special tokens

    def pad_text(self, phoneme_indices, max_length):
        """
        Pads phoneme indices to the predefined maximum length.
        """
        if len(phoneme_indices) < max_length:
            return phoneme_indices + [self.global_vocab['<PAD>']] * (max_length - len(phoneme_indices))
        return phoneme_indices[:max_length]

    def text_to_indices(self, phonemes):
        indices = []
        for p in phonemes:
            if isinstance(p, tuple):
                p = "_".join(p)  # Example: ('a', 'b') -> 'ab'

            if p not in self.global_vocab:
                self.global_vocab[p] = len(self.global_vocab)

            indices.append(self.global_vocab.get(p, self.global_vocab['<UNK>']))
        return indices

    def __len__(self):
        return min(len(self.data_source), len(self.data_target))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # from g2p_en import G2p
        # from g2pM import G2pM

        g2p_en = G2p()
        g2p_cn = G2pM()

        standard_sr = 16000  # Standard sample rate
        max_length = int(self.max_audio_length * standard_sr)  # Max audio length in samples
        max_text_length = 226  # Predefined max text length

        # File paths
        source_audio_path = os.path.join(self.root_dir_source, self.merged_data.iloc[idx]['path_source'])
        target_audio_path = os.path.join(self.root_dir_target, self.merged_data.iloc[idx]['path_target'])

        try:
            # Load source audio
            source_audio, source_sr = torchaudio.load(source_audio_path)
            if source_sr != standard_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=standard_sr)
                source_audio = resampler(source_audio)
            source_audio = source_audio.mean(dim=0) if source_audio.ndim > 1 else source_audio.squeeze(0)
            if source_audio.shape[0] > max_length:
                source_audio = source_audio[:max_length]
            else:
                source_audio = torch.nn.functional.pad(source_audio, (0, max_length - source_audio.shape[0]))
        except Exception as e:
            print(f"Error loading source audio {source_audio_path}: {e}")
            source_audio = torch.zeros(max_length)

        try:
            # Load target audio
            target_audio, target_sr = torchaudio.load(target_audio_path)
            if target_sr != standard_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=standard_sr)
                target_audio = resampler(target_audio)
            target_audio = target_audio.mean(dim=0) if target_audio.ndim > 1 else target_audio.squeeze(0)
            if target_audio.shape[0] > max_length:
                target_audio = target_audio[:max_length]
            else:
                target_audio = torch.nn.functional.pad(target_audio, (0, max_length - target_audio.shape[0]))
        except Exception as e:
            print(f"Error loading target audio {target_audio_path}: {e}")
            target_audio = torch.zeros(max_length)

        # Convert text to phoneme indices
        source_text = self.merged_data.iloc[idx]['sentence_source']
        target_text = self.merged_data.iloc[idx]['sentence_target']

        source_phonemes = g2p_en(source_text)
        # phoneme_vocab = {p: idx for idx, p in enumerate(set(source_phonemes))}
        # source_phoneme_indices = [phoneme_vocab[p] for p in source_phonemes]

        target_phonemes = g2p_cn(target_text)
        # phoneme_vocab = {p: idx for idx, p in enumerate(set(target_phonemes))}
        # target_phoneme_indices = [phoneme_vocab[p] for p in target_phonemes]

        # print("Source Phonemes:", source_phonemes)
        # print("Target Phonemes:", target_phonemes)

        # Pad text indices
        source_text_indices = torch.tensor(self.pad_text(self.text_to_indices(source_phonemes), max_text_length),
                                           dtype=torch.long)
        target_text_indices = torch.tensor(self.pad_text(self.text_to_indices(target_phonemes), max_text_length),
                                           dtype=torch.long)

        # Perform ConcatAug augmentation with probability 0.5
        if self.concat_aug and torch.rand(1).item() < self.concat_prob:
            concat_idx = torch.randint(0, len(self.merged_data), (1,)).item()

            # Load additional source audio
            concat_source_audio_path = os.path.join(self.root_dir_source, self.merged_data.iloc[concat_idx]['path_source'])
            try:
                source_audio_aug, source_sr_aug = torchaudio.load(concat_source_audio_path)
                if source_sr_aug != standard_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=source_sr_aug, new_freq=standard_sr)
                    source_audio_aug = resampler(source_audio_aug)
                source_audio_aug = source_audio_aug.mean(dim=0) if source_audio_aug.ndim > 1 else source_audio_aug.squeeze(0)
                source_audio_aug = torch.nn.functional.pad(source_audio_aug, (0, max_length - source_audio_aug.shape[0]))
            except Exception as e:
                print(f"Error loading concatenated source audio {concat_source_audio_path}: {e}")
                source_audio_aug = torch.zeros(max_length)

            # Load additional target audio
            concat_target_audio_path = os.path.join(self.root_dir_target, self.merged_data.iloc[concat_idx]['path_target'])
            try:
                target_audio_aug, target_sr_aug = torchaudio.load(concat_target_audio_path)
                if target_sr_aug != standard_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=target_sr_aug, new_freq=standard_sr)
                    target_audio_aug = resampler(target_audio_aug)
                target_audio_aug = target_audio_aug.mean(dim=0) if target_audio_aug.ndim > 1 else target_audio_aug.squeeze(0)
                target_audio_aug = torch.nn.functional.pad(target_audio_aug, (0, max_length - target_audio_aug.shape[0]))
            except Exception as e:
                print(f"Error loading concatenated target audio {concat_target_audio_path}: {e}")
                target_audio_aug = torch.zeros(max_length)

            # Concatenate audio
            source_audio = torch.cat([source_audio, source_audio_aug], dim=0)[:max_length]
            target_audio = torch.cat([target_audio, target_audio_aug], dim=0)[:max_length]

            # Concatenate text
            concat_source_text = self.merged_data.iloc[concat_idx]['sentence_source']
            concat_target_text = self.merged_data.iloc[concat_idx]['sentence_target']

            source_phonemes = g2p_en(concat_source_text)
            # phoneme_vocab = {p: idx for idx, p in enumerate(set(source_phonemes))}
            # concat_source_indices = [phoneme_vocab[p] for p in source_phonemes]

            target_phonemes = g2p_en(concat_target_text)
            # phoneme_vocab = {p: idx for idx, p in enumerate(set(target_phonemes))}
            # concat_target_indices = [phoneme_vocab[p] for p in target_phonemes]

            source_text_indices = torch.tensor(self.pad_text(self.text_to_indices(source_phonemes), max_text_length),
                                               dtype=torch.long)
            target_text_indices = torch.tensor(self.pad_text(self.text_to_indices(target_phonemes), max_text_length),
                                               dtype=torch.long)

        return source_audio, target_audio, source_text_indices, target_text_indices



class SpeechToSpeechDatasetPhonemePreProcess(Dataset):
    def __init__(self, root_dir_source, root_dir_target, tsv_file_source, tsv_file_target,
                 concat_aug=True, concat_prob=0.5, max_audio_length=10, max_text_length=150):
        """
        Args:
            root_dir_source: Directory with source audio files.
            root_dir_target: Directory with target audio files.
            tsv_file_source: TSV file with source annotations.
            tsv_file_target: TSV file with target annotations.
            concat_aug: Boolean flag to enable ConcatAug.
            concat_prob: Probability of applying ConcatAug.
            max_audio_length: Maximum audio length in seconds.
            max_text_length: Maximum phoneme sequence length.
        """
        self.standard_sr = 16000
        self.root_dir_source = root_dir_source
        self.root_dir_target = root_dir_target
        self.concat_aug = concat_aug
        self.concat_prob = concat_prob
        self.max_audio_length = int(max_audio_length * self.standard_sr)  # In samples
        self.max_text_length = 15201
        self.g2p_en = G2p()
        self.g2p_cn = G2pM()

        # Load TSV files and merge data
        self.data_source = pd.read_csv(tsv_file_source, sep='\t')
        self.data_target = pd.read_csv(tsv_file_target, sep='\t')

        if 'sentence_id' in self.data_source.columns and 'sentence_id' in self.data_target.columns:
            self.merged_data = pd.merge(self.data_source, self.data_target, on='sentence_id',
                                        suffixes=('_source', '_target'))
        else:
            self.merged_data = pd.concat([self.data_source.reset_index(), self.data_target.reset_index()], axis=1)

        # Preprocess phonemes
        print(f'building global phoneme vocab')
        self.phoneme_vocab = self.build_global_phoneme_vocab()
        print(f'preprocessing phonemes')
        self.preprocessed_phonemes = self.preprocess_phonemes()

    def build_global_phoneme_vocab(self):
        """
        Builds a global vocabulary for phonemes across the dataset.
        """
        vocab = set()
        for _, row in self.merged_data.iterrows():
            source_text = row['sentence_source']
            target_text = row['sentence_target']
            vocab.update(self.g2p_en(source_text))
            vocab.update(self.g2p_cn(target_text))
        vocab = sorted(vocab)
        vocab = ['<PAD>', '<UNK>'] + vocab  # Add special tokens
        return {p: idx for idx, p in enumerate(vocab)}

    def build_global_phoneme_vocab_parallel(self):
        """
        Builds a global vocabulary for phonemes across the dataset using parallel processing.
        """
        def process_text_pair(row):
            source_text = row['sentence_source']
            target_text = row['sentence_target']
            source_phonemes = self.g2p_en(source_text)
            target_phonemes = self.g2p_cn(target_text)
            return set(source_phonemes + target_phonemes)

        vocab = set()
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_text_pair, self.merged_data.to_dict('records')))
            for phoneme_set in results:
                vocab.update(phoneme_set)

        vocab = sorted(vocab)
        vocab = ['<pad>', '<unk>'] + vocab  # Add special tokens
        return {p: idx for idx, p in enumerate(vocab)}

    def preprocess_phonemes(self):
        """
        Preprocess phonemes for the entire dataset to save computation time.
        """
        preprocessed = []
        for _, row in self.merged_data.iterrows():
            source_phonemes = self.text_to_phoneme_indices(row['sentence_source'], 'en')
            target_phonemes = self.text_to_phoneme_indices(row['sentence_target'], 'cn')

            preprocessed.append({
                'source': self.pad_text(source_phonemes, self.max_text_length),
                'target': self.pad_text(target_phonemes, self.max_text_length)
            })
        return preprocessed

    def text_to_phoneme_indices(self, text, language):
        """
        Converts input text into phoneme indices using G2P models and global vocabulary.
        """
        if language == 'en':
            phonemes = self.g2p_en(text)
        elif language == 'cn':
            phonemes = self.g2p_cn(text)
        else:
            raise ValueError("Unsupported language.")
        return [self.phoneme_vocab.get(p, self.phoneme_vocab['<unk>']) for p in phonemes]

    def pad_text(self, indices, max_length):
        """
        Pads phoneme indices to the specified maximum length.
        """
        return indices[:max_length] + [self.phoneme_vocab['<pad>']] * (max_length - len(indices))

    def load_audio(self, file_path):
        """
        Loads and processes audio to a fixed length and sample rate.
        """
        try:
            audio, sr = torchaudio.load(file_path)
            if sr != self.standard_sr:
                audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.standard_sr)(audio)
            audio = audio.mean(dim=0) if audio.ndim > 1 else audio.squeeze(0)
            if audio.shape[0] > self.max_audio_length:
                audio = audio[:self.max_audio_length]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.max_audio_length - audio.shape[0]))
            return audio
        except Exception as e:
            print(f"Error loading audio {file_path}: {e}")
            return torch.zeros(self.max_audio_length)

    def __len__(self):
        return len(self.merged_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        source_audio_path = os.path.join(self.root_dir_source, self.merged_data.iloc[idx]['path_source'])
        target_audio_path = os.path.join(self.root_dir_target, self.merged_data.iloc[idx]['path_target'])

        source_audio = self.load_audio(source_audio_path)
        target_audio = self.load_audio(target_audio_path)
        source_text_indices = torch.tensor(self.preprocessed_phonemes[idx]['source'])
        target_text_indices = torch.tensor(self.preprocessed_phonemes[idx]['target'])

        # ConcatAug logic
        if self.concat_aug and random.random() < self.concat_prob:
            concat_idx = random.randint(0, len(self.merged_data) - 1)

            concat_source_audio = self.load_audio(os.path.join(
                self.root_dir_source, self.merged_data.iloc[concat_idx]['path_source']))
            concat_target_audio = self.load_audio(os.path.join(
                self.root_dir_target, self.merged_data.iloc[concat_idx]['path_target']))

            concat_source_indices = torch.tensor(
                self.preprocessed_phonemes[concat_idx]['source'])
            concat_target_indices = torch.tensor(
                self.preprocessed_phonemes[concat_idx]['target'])

            source_audio = torch.cat([source_audio, concat_source_audio], dim=0)[:self.max_audio_length]
            target_audio = torch.cat([target_audio, concat_target_audio], dim=0)[:self.max_audio_length]
            source_text_indices = torch.cat([source_text_indices, concat_source_indices], dim=0)[:self.max_text_length]
            target_text_indices = torch.cat([target_text_indices, concat_target_indices], dim=0)[:self.max_text_length]

        return source_audio, target_audio, source_text_indices, target_text_indices


def play_audio(waveform, sample_rate=24000, filename=None):
    """
    Play or save the audio waveform.

    Args:
    waveform (torch.Tensor or np.ndarray): The audio waveform to play/save.
    sample_rate (int): The sample rate of the audio (default: 24000).
    filename (str, optional): If provided, save the audio to this file instead of playing it.

    Returns:
    IPython.display.Audio or None: Audio widget if in a notebook environment, None otherwise.
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()

    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)  # Remove batch dimension if present

    if filename:
        sf.write(filename, waveform, sample_rate)
        print(f"Audio saved to {filename}")
        return None

    try:
        return ipd.Audio(waveform, rate=sample_rate, autoplay=False)
    except:
        temp_file = "temp_audio.wav"
        sf.write(temp_file, waveform, sample_rate)
        print(f"Audio saved to temporary file: {os.path.abspath(temp_file)}")
        print("You can play this file using your system's audio player.")
        return None