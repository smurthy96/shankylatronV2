import torch
import torch.nn as nn
import torchaudio.transforms as T
from nemo.collections.tts.models import HifiGanModel
import torchvision.transforms as transforms
from nemo.collections.asr.modules import ConformerEncoder
from torch.nn.functional import pad

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2, dropout=0.1):  # Further reduced num_heads for memory efficiency
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.mha(query, key, value)
        attn_output = self.layer_norm(attn_output + query)  # Residual connection
        return self.dropout(attn_output)

class SpeechEncoder(nn.Module):
    def __init__(self, input_dim=80, encoder_hidden_dim=128, num_heads=2, num_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = ConformerEncoder(
            feat_in=input_dim,
            n_layers=num_layers,
            d_model=encoder_hidden_dim,
            n_heads=num_heads,
            ff_expansion_factor=4,
            conv_kernel_size=31,
            dropout=dropout,
            subsampling="striding",
            subsampling_factor=4,
        )

    def forward(self, x, lengths):
        x = x.transpose(1, 2)  # Conformer expects (batch, features, seq_len)
        lengths = lengths // 4  # Adjust for the subsampling factor
        output, _ = self.encoder(audio_signal=x, length=lengths)
        return output.transpose(1, 2)  # Back to (batch, seq_len, features)


class LinguisticDecoder(nn.Module):
    def __init__(self, encoder_hidden_dim=128, phoneme_dim=126657, reduced_dim=64, num_layers=4, dropout=0.1):
        super().__init__()
        self.mha = MultiheadAttention(embed_dim=encoder_hidden_dim, num_heads=2, dropout=dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(encoder_hidden_dim, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, phoneme_dim)
        )

    def forward(self, encoder_output):
        attn_output = self.mha(encoder_output, encoder_output, encoder_output)
        return attn_output, self.output_projection(attn_output)


class AcousticSynthesizer(nn.Module):
    def __init__(self, attn_hidden_dim=128, phoneme_dim=126657, embedding_dim=64, hidden_dim=80, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embed the phoneme class into a smaller dimensional space
        self.phoneme_embedding = nn.Embedding(phoneme_dim, embedding_dim)

        # Projection layer to reduce phoneme embeddings to match attention hidden dimension
        self.phoneme_projection = nn.Linear(embedding_dim, attn_hidden_dim)

        # Projection to ensure d_model matches Transformer
        self.input_projection = nn.Linear(attn_hidden_dim + attn_hidden_dim, attn_hidden_dim)

        # Define Transformer for synthesizing mel spectrogram
        self.decoder = nn.Transformer(
            d_model=attn_hidden_dim,
            nhead=2,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=0.1,
            activation='relu',
        )
        self.output_projection = nn.Linear(attn_hidden_dim, hidden_dim)

    def forward(self, attn_output, phoneme_indices):
        # print(f"attn_output shape: {attn_output.shape}")
        # print(f"phoneme_indices shape: {phoneme_indices.shape}")

        # Convert phoneme indices to one-dimensional indices
        phoneme_indices = phoneme_indices.argmax(dim=-1)
        # print(f"phoneme_indices after argmax: {phoneme_indices.shape}")

        phoneme_embeddings = self.phoneme_embedding(phoneme_indices)
        phoneme_embeddings = self.phoneme_projection(phoneme_embeddings)
        # print(f"phoneme_embeddings shape after projection: {phoneme_embeddings.shape}")

        # Ensure dimensions match for concatenation
        combined_input = torch.cat([attn_output, phoneme_embeddings],
                                    dim=-1)  # (batch_size, time_steps, attn_hidden_dim + attn_hidden_dim)
        # print(f"combined_input shape before projection: {combined_input.shape}")

        combined_input = self.input_projection(combined_input)  # Project to match d_model
        # print(f"combined_input shape after projection: {combined_input.shape}")

        combined_input = combined_input.transpose(0, 1)  # To (seq_len, batch, features)
        output = self.decoder(
            src=combined_input,
            tgt=combined_input
        )
        return self.output_projection(output.transpose(0, 1))  # Back to (batch, seq_len, features)


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
        phoneme_dim=126657,
        output_dim=80,
        writer=None
    ):
        super().__init__()

        self.tfwriter = writer

        self.speech_encoder = SpeechEncoder(
            input_dim=input_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            num_heads=8,
            num_layers=encoder_layers
        )

        self.linguistic_decoder = LinguisticDecoder(
            encoder_hidden_dim=encoder_hidden_dim,
            phoneme_dim=phoneme_dim,
            reduced_dim=512,
            num_layers=4
        )

        self.acoustic_synthesizer = AcousticSynthesizer(
            attn_hidden_dim=encoder_hidden_dim,
            phoneme_dim=phoneme_dim,
            hidden_dim=output_dim
        )

        self.NeuralNemoVocoder = GriffinLimVocoder()

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

    def forward(self, source_mel, lengths, batch_idx, epoch):
        encoder_output = self.speech_encoder(source_mel, lengths)
        attn_output, phoneme_embeddings = self.linguistic_decoder(encoder_output)
        mel_output = self.acoustic_synthesizer(attn_output, phoneme_embeddings)
        waveform = self.NeuralNemoVocoder(mel_output)

        try:
            if (batch_idx + 1) % 100 == 0:
                self.tfwriter.add_histogram('Speech Encoder', encoder_output, global_step=batch_idx)
                self.tfwriter.add_histogram('Intermediate/Attention_Output', attn_output, global_step=batch_idx)
                self.tfwriter.add_histogram('Phoneme Output', phoneme_embeddings, global_step=batch_idx)
                self.tfwriter.add_image(f'Intermediate_mel/epoch_{epoch}_batch_{batch_idx}',
                                        self.spectrogram_to_image(mel_output[0].detach().cpu()), global_step=batch_idx)
        except Exception as e:
            print(f'cant write speaker and attn outputs {e}')

        return waveform, phoneme_embeddings
