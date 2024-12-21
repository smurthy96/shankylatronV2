import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio import transforms as T
from translatotron import Translatotron
from translatotron import SpeechToSpeechDataset

# Hyperparameters
batch_size = 10
num_epochs = 100
learning_rate = 0.002


if __name__ == "__main__":
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    model = Translatotron(auxiliary_decoder_output_dim_source=256, auxiliary_decoder_output_dim_target=5356,).to(device)

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    source_dir_clips = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\clips"
    target_dir_clips = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\clips_translated"
    source_tsv_file = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_gendered.tsv"
    target_tsv_file = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_translated.tsv"

    train_dataset = SpeechToSpeechDataset(root_dir_source=source_dir_clips, root_dir_target=target_dir_clips,
                                          tsv_file_source=source_tsv_file, tsv_file_target=target_tsv_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    # Mel spectrogram converter
    mel_converter = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80).to(device)

    # def train_one_epoch(model, loader, optimizer, epoch):
    #     model.train()
    #     total_loss = 0
    #
    #     for batch_idx, (source_audio, target_audio, source_text, target_text) in enumerate(loader):
    #         source_audio, target_audio = source_audio.to(device), target_audio.to(device)
    #         source_text, target_text = source_text.to(device), target_text.to(device)
    #
    #         source_mel = mel_converter(source_audio)
    #         target_mel = mel_converter(target_audio)
    #
    #         # Ensure correct shape for LSTM input
    #         source_mel = source_mel.permute(0, 2, 1)  # (batch_size, time_steps, n_mels -> batch_size, sequence_length, input_dim)
    #         target_mel = target_mel.permute(0, 2, 1)  # Same adjustment for target mel
    #
    #         print(f"Source mel spectrogram shape: {source_mel.shape}")
    #         print(f"Target mel spectrogram shape: {target_mel.shape}")
    #
    #         optimizer.zero_grad()
    #
    #
    #         # Forward pass
    #         output_waveform, _, aux_source, aux_target = model(source_mel)
    #
    #         print(f"Source text max index: {source_text.max().item()}, min index: {source_text.min().item()}")
    #         print(f"Target text max index: {target_text.max().item()}, min index: {target_text.min().item()}")
    #         print(f"Aux source num classes: {aux_source.size(-1)}")
    #         print(f"Aux target num classes: {aux_target.size(-1)}")
    #
    #         print("________________________")
    #
    #         print(f"aux_source shape: {aux_source.shape}")  # Expected: (batch_size, time_steps, vocab_size)
    #         print(f"source_text shape: {source_text.shape}")  # Expected: (batch_size, sequence_length)
    #         print(f"aux_source shape: {aux_target.shape}")  # Expected: (batch_size, time_steps, vocab_size)
    #         print(f"source_text shape: {target_text.shape}")  # Expected: (batch_size, sequence_length)
    #
    #         # Convert waveform output to mel spectrogram
    #         output_mel = mel_converter(output_waveform)
    #         output_mel = output_mel.permute(0, 2, 1)  # Ensure (batch_size, sequence_length, mel_bins)
    #
    #         print(f"Source mel spectrogram shape: {source_mel.shape}")
    #         print(f"output mel spectrogram shape: {output_mel.shape}")
    #
    #         max_time_steps = max(output_mel.size(1), target_mel.size(1))
    #         output_mel = torch.nn.functional.pad(output_mel, (0, 0, 0, max_time_steps - output_mel.size(1)))
    #         target_mel = torch.nn.functional.pad(target_mel, (0, 0, 0, max_time_steps - target_mel.size(1)))
    #
    #         mask = (target_mel != 0).float()  # Assume padding is 0
    #
    #         waveform_loss = mse_loss(output_mel * mask, target_mel * mask)
    #
    #         # Align sequence length for aux_source and source_text
    #         min_seq_length = min(aux_source.size(1), source_text.size(1))
    #         aux_source = aux_source[:, :min_seq_length, :]
    #         source_text = source_text[:, :min_seq_length]
    #
    #         # Align sequence length for aux_source and source_text
    #         min_seq_length = min(aux_target.size(1), target_text.size(1))
    #         aux_target = aux_target[:, :min_seq_length, :]
    #         target_text = target_text[:, :min_seq_length]
    #
    #         print(f"aux_source shape: {aux_source.shape}")  # Expected: (batch_size, time_steps, vocab_size)
    #         print(f"source_text shape: {source_text.shape}")  # Expected: (batch_size, sequence_length)
    #         print(f"aux_source shape: {aux_target.shape}")  # Expected: (batch_size, time_steps, vocab_size)
    #         print(f"source_text shape: {target_text.shape}")  # Expected: (batch_size, sequence_length)
    #
    #         source_text_loss = ce_loss(aux_source.reshape(-1, aux_source.size(-1)), source_text.reshape(-1))
    #         target_text_loss = ce_loss(aux_target.reshape(-1, aux_target.size(-1)), target_text.reshape(-1))
    #
    #         #  losses
    #         total_loss = waveform_loss + source_text_loss + target_text_loss
    #
    #         total_loss.backward()
    #         optimizer.step()
    #
    #         if batch_idx % 10 == 0:
    #             print(f'Epoch {epoch}, Batch {batch_idx}/{len(loader)}, '
    #                   f'Loss: {total_loss.item():.4f}')
    #
    #     return total_loss / len(loader)
    def train_one_epoch(model, loader, optimizer, epoch):
        model.train()
        total_loss = 0

        for batch_idx, (source_audio, target_audio, source_text, target_text) in enumerate(loader):
            # Move data to device
            source_audio, target_audio = source_audio.to(device), target_audio.to(device)
            source_text, target_text = source_text.to(device), target_text.to(device)

            optimizer.zero_grad()



            # Convert audio to mel spectrogram
            source_mel = mel_converter(source_audio).permute(0, 2, 1)  # (batch, time_steps, n_mels)
            target_mel = mel_converter(target_audio).permute(0, 2, 1)

            # print(f"Source mel spectrogram shape: {source_mel.shape}")
            # print(f"Target mel spectrogram shape: {target_mel.shape}")

            # Forward pass
            output_waveform, _, aux_source, aux_target = model(source_mel)

            # Debugging: Print key information
            # print(f"Source text max index: {source_text.max().item()}, min index: {source_text.min().item()}")
            # print(f"Target text max index: {target_text.max().item()}, min index: {target_text.min().item()}")
            # print(f"Aux source num classes: {aux_source.size(-1)}")
            # print(f"Aux target num classes: {aux_target.size(-1)}")
            #
            # print(f"Aux source shape: {aux_source.shape}")  # Expected: (batch, time_steps, vocab_size)
            # print(f"Source text shape: {source_text.shape}")  # Expected: (batch, sequence_length)
            # print(f"Aux target shape: {aux_target.shape}")  # Expected: (batch, time_steps, vocab_size)
            # print(f"Target text shape: {target_text.shape}")  # Expected: (batch, sequence_length)

            # Convert waveform output to mel spectrogram
            output_mel = mel_converter(output_waveform).permute(0, 2, 1)

            # print(f"Output mel spectrogram shape: {output_mel.shape}")

            # Align mel spectrogram lengths with padding
            max_time_steps = max(output_mel.size(1), target_mel.size(1))
            output_mel = torch.nn.functional.pad(output_mel, (0, 0, 0, max_time_steps - output_mel.size(1)))
            target_mel = torch.nn.functional.pad(target_mel, (0, 0, 0, max_time_steps - target_mel.size(1)))

            # Compute waveform loss with masking
            mask = (target_mel != 0).float()  # Assume padding is 0
            waveform_loss = mse_loss(output_mel * mask, target_mel * mask)

            # Align sequence lengths for auxiliary losses
            min_seq_length = min(aux_source.size(1), source_text.size(1))
            aux_source = aux_source[:, :min_seq_length, :]
            source_text = source_text[:, :min_seq_length]

            # Align sequence lengths
            min_seq_length = min(aux_target.size(1), target_text.size(1))
            aux_target = aux_target[:, :min_seq_length, :]
            target_text = target_text[:, :min_seq_length]

            # Clamp or mask out-of-bounds indices
            num_classes = aux_target.size(-1)  # 5356
            target_text = target_text.clamp(0, num_classes - 1)  # Alternatively: Use masking

            # print(f"Aligned aux source shape: {aux_source.shape}")
            # print(f"Aligned source text shape: {source_text.shape}")
            # print(f"Aligned aux target shape: {aux_target.shape}")
            # print(f"Aligned target text shape: {target_text.shape}")

            # Compute text losses
            source_text_loss = ce_loss(aux_source.reshape(-1, aux_source.size(-1)), source_text.reshape(-1))
            target_text_loss = ce_loss(aux_target.reshape(-1, aux_target.size(-1)), target_text.reshape(-1))

            # Total loss
            total_loss = waveform_loss + source_text_loss + target_text_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(loader)}, '
                      f'Waveform Loss: {waveform_loss.item():.4f}, '
                      f'Source Text Loss: {source_text_loss.item():.4f}, '
                      f'Target Text Loss: {target_text_loss.item():.4f}, '
                      f'Total Loss: {total_loss.item():.4f}')

        # Return average loss for the epoch
        return total_loss / len(loader)

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, epoch)

        scheduler.step(avg_loss)

        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'models/translatotron_model_epoch_{epoch + 1}.pth')

    print("Training completed!")