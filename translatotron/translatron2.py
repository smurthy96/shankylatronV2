import torch
import torch.nn as nn
import torchaudio.transforms
from nemo.collections.tts.models import HifiGanModel
import torch.nn.functional as F
from nemo.collections.asr.modules import ConformerEncoder
import torchaudio
import torchvision.transforms as transforms
import torchaudio.transforms as T


class SpeechEncoder(nn.Module):
    def __init__(self, input_dim=80, encoder_hidden_dim=512, num_heads=8, num_layers=16, dropout=0.1):
        super().__init__()
        self.subsampling_factor = 4
        self.encoder = ConformerEncoder(
            feat_in=input_dim,
            n_layers=num_layers,
            d_model=encoder_hidden_dim,
            n_heads=num_heads,
            ff_expansion_factor=4,
            conv_kernel_size=31,
            dropout=dropout,
            subsampling="striding",
            subsampling_factor=self.subsampling_factor,
        )

    def forward(self, x, lengths):
        # Verify input shape (batch_size, time_steps, n_mels)
        assert x.ndim == 3, f"Expected input shape (batch_size, time_steps, n_mels), got {x.shape}"

        # NeMo expects inputs as (batch_size, time_steps, n_mels) -> permute for audio_signal
        x = x.transpose(1, 2)  # Shape: (batch_size, n_mels, time_steps)

        # Adjust lengths for subsampling
        subsampled_lengths = lengths // self.subsampling_factor

        # Pass to the Conformer Encoder
        output, _ = self.encoder(audio_signal=x, length=subsampled_lengths)

        # Transpose back to (batch_size, time_steps, features)
        return output.transpose(1, 2)


class LinguisticDecoder(nn.Module):
    def __init__(self, encoder_hidden_dim=512, phoneme_dim=126657, num_layers=4, dropout=0.1, reduction_factor=123):
        super().__init__()
        self.linguistic_decoder = nn.LSTM(
            input_size=encoder_hidden_dim,
            hidden_size=phoneme_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.phoneme_projector = nn.Sequential(
            nn.Linear(phoneme_dim, phoneme_dim//reduction_factor),
            nn.ReLU(),
            nn.Linear(phoneme_dim//reduction_factor, phoneme_dim)
        )

    def forward(self, encoder_output):
        phoneme_output, _ = self.linguistic_decoder(encoder_output)
        return self.phoneme_projector(phoneme_output)


class LinguisticDecoderEmbeddings(nn.Module):
    def __init__(self, encoder_hidden_dim=1024, phoneme_dim=126656, embedding_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.linguistic_decoder = nn.LSTM(
            input_size=encoder_hidden_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.phoneme_projector = nn.Linear(embedding_dim, phoneme_dim)

    def forward(self, encoder_output):
        # Pass encoder output through LSTM
        phoneme_output, _ = self.linguistic_decoder(encoder_output)
        # Project back to the original phoneme dimension
        return self.phoneme_projector(phoneme_output)


class AcousticSynthesizer(nn.Module):
    def __init__(self, attn_hidden_dim=1536, phoneme_dim=126657, embedding_dim=512, hidden_dim=80, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embed the phoneme class into a smaller dimensional space
        self.phoneme_embedding = nn.Embedding(phoneme_dim, embedding_dim)

        # Define LSTM for synthesizing mel spectrogram
        self.synthesizer_lstm = nn.LSTM(
            input_size=attn_hidden_dim + embedding_dim,  # Updated input size
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Project the LSTM output to mel spectrogram dimension
        self.synthesizer_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, attn_output, phoneme_indices):
        # Convert probabilities to indices
        phoneme_indices = phoneme_indices.argmax(dim=-1)  # Shape: (batch_size, time_steps)

        # Embed the phoneme indices
        phoneme_embeddings = self.phoneme_embedding(phoneme_indices.long())  # (batch_size, time_steps, embedding_dim)

        # Ensure dimensions match for concatenation
        synth_input = torch.cat([attn_output, phoneme_embeddings], dim=-1)  # (batch_size, time_steps, attn_hidden_dim + embedding_dim)

        # Pass through LSTM
        mel_output, _ = self.synthesizer_lstm(synth_input)

        # Project to final mel spectrogram
        mel_output = self.synthesizer_projector(mel_output)
        return mel_output




class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim=80, output_dim=1024, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # self._initialize_weights()

    def forward(self, x):
        x = x.mean(dim=1)
        return self.encoder(x)

    def _initialize_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query, key, value):
        return self.mha(query, key, value)[0]


class MultiheadAttentionNorm(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        query, key, value = [torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0) for t in (query, key, value)]
        query, key, value = [(t - t.mean(dim=-1, keepdim=True)) / (t.std(dim=-1, keepdim=True) + 1e-6) for t in (query, key, value)]
        attn_output, _ = self.mha(query, key, value)
        attn_output = self.layer_norm(attn_output)
        return self.dropout(attn_output)


class NeuralNemoVocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        print("Loading pretrained Hifi-GAN model...")
        self.vocoder = HifiGanModel.from_pretrained(model_name="tts_zh_hifigan_sfspeech").to(self.device)
        self.vocoder.eval()
        print("Hifi-GAN model loaded successfully.")

    def forward(self, spectrogram):
        with torch.no_grad():
            # Ensure the spectrogram is on the correct device
            spectrogram = spectrogram.to(self.device)

            # Permute dimensions to match Hifi-GAN's expected input format
            spectrogram = spectrogram.permute(0, 2, 1)  # [batch_size, n_mels, time_steps] -> [batch_size, time_steps, n_mels]

            # Convert spectrogram to waveform
            waveform = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        return waveform


class GriffinLimVocoder(nn.Module):
    def __init__(self, n_fft=1024, n_mels=80, hop_length=256, win_length=None, n_iter=32):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.n_iter = n_iter

        # Mel-to-Spectrogram Converter
        self.mel_to_spec_transform = T.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=16000,
            max_iter=1000,
            norm="slaney",
            mel_scale="htk",
        )

        # Griffin-Lim Algorithm
        self.griffin_lim_transform = T.GriffinLim(
            n_fft=self.n_fft,
            n_iter=self.n_iter,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )


    def forward(self, mel_spectrogram):
        """
        Args:
            mel_spectrogram: Mel spectrogram tensor of shape (batch_size, n_mels, time_steps).

        Returns:
            Reconstructed waveform tensor of shape (batch_size, samples).
        """
        # Ensure the input is on the same device
        device = mel_spectrogram.device
        #if mel_spectrogram.shape[1] != self.n_mels:
        mel_spectrogram = mel_spectrogram.permute(0, 2, 1)  # Correct shape to (batch_size, n_mels, time_steps)

        # Move transforms to the correct device
        self.mel_to_spec_transform = self.mel_to_spec_transform.to(device)
        self.griffin_lim_transform = self.griffin_lim_transform.to(device)

        # Convert mel spectrogram to linear spectrogram
        linear_spectrogram = self.mel_to_spec_transform(mel_spectrogram.detach())

        # Apply Griffin-Lim to reconstruct the waveform
        waveform = self.griffin_lim_transform(linear_spectrogram)

        return waveform


class Translatotron2(nn.Module):
    def __init__(
        self,
        input_dim=80,
        encoder_hidden_dim=1024,
        encoder_layers=16,
        phoneme_dim=126656,
        num_heads=8,
        output_dim=80,
        writer=None
    ):
        super().__init__()

        self.tfwriter = writer

        self.speaker_encoder = SpeakerEncoder(
            input_dim=input_dim, output_dim=1024
        )

        combined_dim = encoder_hidden_dim + 1024
        self.mha = MultiheadAttentionNorm(embed_dim=combined_dim, num_heads=8)

        self.NeuralNemoVocoder = GriffinLimVocoder()

        # New modules for Translatotron 2
        # Speech Encoder: output hidden dimension = encoder_hidden_dim * 4
        self.speech_encoder = SpeechEncoder(
            input_dim=input_dim,
            encoder_hidden_dim=encoder_hidden_dim,  # Expanded hidden dimension
            num_heads=num_heads,
            num_layers=encoder_layers
        )

        # New: Downsample using 8-Layer LSTM
        # self.downsampler = DownsampleLSTM(
        #     input_dim=encoder_hidden_dim,
        #     output_dim=encoder_hidden_dim,
        #     num_layers=encoder_layers
        # )

        # Linguistic Decoder expects encoder_hidden_dim * 2
        self.linguistic_decoder = LinguisticDecoderEmbeddings(
            encoder_hidden_dim=encoder_hidden_dim,
            phoneme_dim=phoneme_dim,
            num_layers=encoder_layers
        )

        self.acoustic_synthesizer = AcousticSynthesizer(
            hidden_dim=output_dim,
            num_layers=encoder_layers//2
        )

    def spectrogram_to_image(self, spectrogram):
        """
        Converts a spectrogram tensor to an image format for TensorBoard logging.
        Input: spectrogram (batch_size, time_steps, n_mels)
        Output: image tensor (3, H, W) suitable for TensorBoard
        """
        # Normalize spectrogram to [0, 1] for visualization
        spectrogram = spectrogram - spectrogram.min()
        spectrogram = spectrogram / spectrogram.max()

        # Add channel dimension for TensorBoard
        spectrogram = spectrogram.unsqueeze(0)  # (1, H, W)

        # Resize to ensure consistent image size (optional)
        resize_transform = transforms.Resize((128, 256))  # Adjust size as needed
        spectrogram_image = resize_transform(spectrogram)

        # Repeat for 3 channels (RGB)
        spectrogram_image = spectrogram_image.repeat(3, 1, 1)  # (3, H, W)

        return spectrogram_image

    def forward(self, x, lengths, batch_idx, epoch):

        # 1. Speech Encoder (extracts linguistic features)
        speech_encoder_output = self.speech_encoder(x, lengths)

        # 2. Speaker Embedding (derived directly from the source speech)
        speaker_embedding = self.speaker_encoder(x)  # Encoder processes input directly

        speaker_embedding = speaker_embedding.unsqueeze(1).expand(
            -1, speech_encoder_output.size(1), -1
        )

        # 3. Combine speech encoder output with speaker embedding
        combined_output = torch.cat([speech_encoder_output, speaker_embedding], dim=-1)

        # 4. Multi-Head Attention for joint linguistic-acoustic processing
        attn_output = self.mha(combined_output, combined_output, combined_output)

        # 5. Linguistic Decoder (produces phoneme predictions)
        phoneme_output = self.linguistic_decoder(speech_encoder_output)


        # 6. Acoustic Synthesizer (generates intermediate mel-spectrograms)
        mel_output = self.acoustic_synthesizer(attn_output, phoneme_output)

        # 7. Neural Vocoder (final waveform synthesis)
        waveform = self.NeuralNemoVocoder(mel_output)

        # Normalize waveform to [-1, 1] to avoid instability
        max_val = waveform.abs().max(dim=-1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=1e-10)
        waveform = waveform / max_val

        try:
            if (batch_idx + 1) % 1000 == 0:
                self.tfwriter.add_histogram('Speech Encoder', speech_encoder_output, global_step=batch_idx)
                self.tfwriter.add_histogram('Intermediate/Speaker_Embedding', speaker_embedding, global_step=batch_idx)
                self.tfwriter.add_histogram('Intermediate/Attention_Output', attn_output, global_step=batch_idx)
                self.tfwriter.add_histogram('Phoneme Output', phoneme_output, global_step=batch_idx)
                self.tfwriter.add_image(f'Intermediate_mel/epoch_{epoch}_batch_{batch_idx}',
                                        self.spectrogram_to_image(mel_output[0].detach().cpu()), global_step=batch_idx)
        except Exception as e:
            print(f'cant write speaker and attn outputs {e}')

        return waveform, phoneme_output
