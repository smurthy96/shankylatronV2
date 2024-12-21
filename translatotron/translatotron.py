import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from translatotron.utils import play_audio
import logging
logging.disable(logging.WARNING)
from nemo.collections.tts.models import HifiGanModel


class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim=80, output_dim=1024, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._initialize_weights()

    def forward(self, x):
        return self.encoder(x)

    def _initialize_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


class StackedBLSTMEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=1024, num_layers=8, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x):
        # output = self.lstm(x)
        output, _ = self.lstm(x)
        return output


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query, key, value):
        return self.mha(query, key, value)[0]


class SpectrogramDecoder(nn.Module):
    def __init__(self, input_dim=1024, output_dim=80, hidden_dim=1024, num_layers=6):
        super().__init__()

        # Dynamically create a list of linear + ReLU layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):  # Subtract input and output layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))  # Final output layer

        self.decoder = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.decoder(x)

    def _initialize_weights(self):
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.bidirectional = bidirectional
        if bidirectional:
            self.fc_intermediate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        if self.bidirectional:
            lstm_out = self.fc_intermediate(lstm_out)
            return self.fc(lstm_out)

        return self.fc(lstm_out)


# Bidirectional LSTM Decoder with Linear Fusion
class LSTMDecoderfused(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        self.fc_fusion = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_output = nn.Linear(hidden_dim // 2, output_dim)
        self._initialize_weights()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_fused = F.relu(self.fc_fusion(lstm_out))
        return self.fc_output(lstm_fused)

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


class MandarinAuxiliaryDecoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=5356, num_heads=2, num_layers=2, dropout=0.1):
        super().__init__()

        # Attention for phonetic features (non-tonal)
        self.phonetic_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Attention for tonal features
        self.tonal_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Transformer Encoder for deeper processing
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Bidirectional LSTM paths for phonetic and tonal outputs
        self.phonetic_lstm = nn.LSTM(
            input_dim, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.tonal_lstm = nn.LSTM(
            input_dim, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
        )

        # Fusion and output layers
        self.fc_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Extract phonetic and tonal attention outputs
        phonetic_attention_out, _ = self.phonetic_attention(x, x, x)
        tonal_attention_out, _ = self.tonal_attention(x, x, x)

        # Pass through Transformer for deeper processing
        phonetic_trans_out = self.transformer(phonetic_attention_out)
        tonal_trans_out = self.transformer(tonal_attention_out)

        # Pass through bidirectional LSTMs
        phonetic_lstm_out, _ = self.phonetic_lstm(phonetic_trans_out)
        tonal_lstm_out, _ = self.tonal_lstm(tonal_trans_out)

        # Combine the outputs from phonetic and tonal LSTMs
        combined_out = torch.cat([phonetic_lstm_out, tonal_lstm_out], dim=-1)  # Concatenate along the feature dimension
        fused_out = F.relu(self.fc_fusion(combined_out))  # Fusion layer with ReLU activation

        # Final output layer
        return self.fc_output(fused_out)

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class AuxiliaryDecoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=82, num_heads=1, bidirectional=False, num_layers=2):
        super().__init__()
        self.attention = MultiheadAttention(input_dim, num_heads)
        self.lstm = LSTMDecoder(input_dim, hidden_dim, output_dim, num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, x):
        attention_out = self.attention(x, x, x)
        lstm_out = self.lstm(attention_out)
        return lstm_out


class SpectrogramPostProcessor(nn.Module):
    def __init__(self, input_dim=1025, output_dim=80):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self._initialize_weights()

    def forward(self, x):
        return self.projection(x)

    def _initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)


class SpectrogramPostProcessorSpatial(nn.Module):
    def __init__(self, input_dim=1025, output_dim=80, hidden_dim=256, num_heads=2, dropout=0.2):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq, feature) -> (batch, feature, seq)
        x = self.bn(F.relu(self.conv1d(x)))
        x = x.transpose(1, 2)  # (batch, feature, seq) -> (batch, seq, feature)
        x_residual = x
        x, _ = self.attention(x, x, x)
        x = self.dropout(x + x_residual)
        return self.fc(x)

    def _initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.conv1d.weight)
        torch.nn.init.zeros_(self.conv1d.bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)


# class NeuralVocoderAutocast(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', map_location='cuda')
#         self.waveglow.eval()
#         self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
#
#     def forward(self, spectrogram):
#         spectrogram = spectrogram.to(next(self.waveglow.parameters()).device)
#         spectrogram = spectrogram.permute(0, 2, 1)  # (batch_size, n_mels, time_steps)
#
#         with torch.no_grad(), torch.cuda.amp.autocast():  # Enable mixed precision for inference
#             waveform = self.waveglow.infer(spectrogram)
#         return waveform
#
#
# class NeuralVocoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Load pretrained WaveGlow model from NVIDIA's TorchHub
#         print("Downloading nvidia_waveglow")
#         self.waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', map_location='cuda')
#         self.waveglow.eval()  # Set to evaluation mode
#         print("done downloading waveglow")
#
#         # Remove weight normalization for better inference performance
#         self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
#
#     def forward(self, spectrogram):
#         """
#         Convert a mel-spectrogram to waveform using WaveGlow.
#
#         Args:
#             spectrogram (torch.Tensor): Input mel-spectrogram of shape (batch_size, time_steps, n_mels).
#
#         Returns:
#             torch.Tensor: Generated waveform of shape (batch_size, num_samples).
#         """
#         # Ensure the spectrogram is on the same device as WaveGlow
#         spectrogram = spectrogram.to(next(self.waveglow.parameters()).device)
#
#         # Transpose spectrogram to match WaveGlow's expected input shape: (batch_size, n_mels, time_steps)
#         spectrogram = spectrogram.permute(0, 2, 1)
#
#         # Generate waveform using WaveGlow
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             waveform = self.waveglow.infer(spectrogram)
#
#         return waveform
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


class FastVocoder(nn.Module):
    def __init__(self, n_fft=2048, n_iter=60, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.n_iter = n_iter
        self.hop_length = hop_length

    def forward(self, spectrogram):
        # Permute to match Griffin-Lim's expected shape: (batch, freq, time)
        spectrogram = spectrogram.permute(0, 2, 1)

        # Use Griffin-Lim batched implementation from torchaudio
        waveforms = torchaudio.functional.griffin_lim(
            spectrogram,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            power=1.0,
            n_iter=self.n_iter
        )
        return waveforms


class Translatotron(nn.Module):
    def __init__(
        self,
        input_dim=80,
        encoder_hidden_dim=1024,
        encoder_layers=10,
        encoder_dropout=0.1,
        speaker_embedding_dim=1024,
        attention_heads=8,
        decoder_hidden_dim=1024,
        decoder_output_dim=1025,
        auxiliary_decoder_hidden_dim=256,
        auxiliary_decoder_output_dim_source=None,
        auxiliary_decoder_output_dim_target=None,
        vocoder_n_fft=2048,
        vocoder_hop_length=512,
        vocoder_n_iter=60,
        neural_vocoder=False,
        neural_vocoder_autocast=False,
    ):
        super().__init__()

        self.neural_vocoder = neural_vocoder
        self.neural_vocoder_autocast = neural_vocoder_autocast

        self.stacked_blstm_encoder = StackedBLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_layers,
            dropout=encoder_dropout
        )

        self.speaker_encoder = SpeakerEncoder(
            input_dim=input_dim, output_dim=speaker_embedding_dim
        )

        combined_dim = encoder_hidden_dim + speaker_embedding_dim
        self.mha = MultiheadAttention(embed_dim=combined_dim, num_heads=attention_heads)

        self.spectrogram_decoder = SpectrogramDecoder(
            input_dim=combined_dim,
            output_dim=decoder_output_dim,
            hidden_dim=decoder_hidden_dim,
        )

        self.NeuralNemoVocoder = NeuralNemoVocoder()

        # self.SpectrogramPostProcessorSpatial = SpectrogramPostProcessorSpatial()

        self.auxiliary_decoder_english = LSTMDecoderfused(
            input_dim=combined_dim,
            hidden_dim=auxiliary_decoder_hidden_dim,
            output_dim=auxiliary_decoder_output_dim_source
        )

        self.auxiliary_decoder_mandarin = MandarinAuxiliaryDecoder(
            input_dim=combined_dim,
            hidden_dim=auxiliary_decoder_hidden_dim*4,
            output_dim=auxiliary_decoder_output_dim_target,
        )

    def forward(self, x, speaker_reference=None):
        encoder_output = self.stacked_blstm_encoder(x)
        # print(f"Encoder output shape: {encoder_output.shape}")
        if speaker_reference is not None:
            speaker_embedding = self.speaker_encoder(speaker_reference)
            speaker_embedding = speaker_embedding.unsqueeze(1).expand(
                -1, encoder_output.size(1), -1
            )
            encoder_output = torch.cat([encoder_output, speaker_embedding], dim=-1)
        else:
            batch_size, seq_len, _ = encoder_output.shape
            zero_padding = torch.zeros(
                batch_size,
                seq_len,
                self.speaker_encoder.encoder[-1].out_features,
                device=encoder_output.device,
            )
            encoder_output = torch.cat([encoder_output, zero_padding], dim=-1)

        # print(f"Encoder output after speaker embedding shape: {encoder_output.shape}")

        # Multihead attention
        mha_output = self.mha(encoder_output, encoder_output, encoder_output)
        # print(f"Multihead attention output shape: {mha_output.shape}")

        # Decode spectrogram
        spectrogram_out = self.spectrogram_decoder(mha_output)
        # print(f"Spectrogram decoder output shape: {spectrogram_out.shape}")

        # Post-process spectrogram
        # processed_spectrogram = self.SpectrogramPostProcessorSpatial(spectrogram_out)
        # print(f"Processed spectrogram shape: {processed_spectrogram.shape}")

        # Neural vocoder: Convert spectrogram to waveform
        waveform = self.NeuralNemoVocoder(spectrogram_out)
        # print(f"Waveform shape: {waveform.shape}")

        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            print("NaN or Inf detected in waveform!")
        waveform = waveform / waveform.abs().max(dim=-1, keepdim=True)[0]

        phonemes_english = self.auxiliary_decoder_english(encoder_output)
        phonemes_mandarin = self.auxiliary_decoder_mandarin(encoder_output)

        return waveform, spectrogram_out, phonemes_english, phonemes_mandarin
