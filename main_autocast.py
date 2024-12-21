import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio import transforms as T
from translatotron import Translatotron
from translatotron import SpeechToSpeechDataset
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

# Hyperparameters
batch_size = 32
num_epochs = 1000
learning_rate = 1e-5
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.rnn_benchmark = True

if __name__ == "__main__":
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    #                                                                   256 5356
    model = Translatotron(auxiliary_decoder_output_dim_source=82, auxiliary_decoder_output_dim_target=5356,
                          neural_vocoder=True, neural_vocoder_autocast=False, decoder_output_dim=80).to(device)

    # class TranslatotronSummaryWrapper(nn.Module):
    #     def __init__(self, model):
    #         super().__init__()
    #         self.model = model
    #
    #     def forward(self, x):
    #         # Only return the first output (waveform) for summary purposes
    #         waveform, _, _, _ = self.model(x)
    #         return waveform if isinstance(waveform, torch.Tensor) else torch.zeros_like(x)

    # # Use the wrapper for torchsummary
    # summary_model = TranslatotronSummaryWrapper(model)
    # summary(summary_model, input_size=(300, 80), batch_size=2, device="cuda")

    # print(model)

    log_dir = "log/raw_loss_logs/"
    log_file = os.path.join(log_dir, "training_log_run.txt")

    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Ensure the log file exists (create if it doesn't)
    if not os.path.exists(log_file):
        open(log_file, "a").close()  # Open in append mode and immediately close

    writer = SummaryWriter(log_dir="log/raw_loss_logs/tensorboard")

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # Initialize the scaler for AMP

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    source_dir_clips = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\clips"
    target_dir_clips = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\clips_translated"
    source_tsv_file = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_gendered.tsv"
    target_tsv_file = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_translated.tsv"

    train_dataset = SpeechToSpeechDataset(root_dir_source=source_dir_clips, root_dir_target=target_dir_clips,
                                          tsv_file_source=source_tsv_file, tsv_file_target=target_tsv_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)

    # Mel spectrogram converter
    mel_converter = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80).to(device)

    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,  # Wait 1 step before starting profiling
            warmup=1,  # Warmup for 1 step to stabilize measurements
            active=3,  # Profile for 3 steps
            repeat=2,  # Repeat the cycle (wait, warmup, active) twice
        ),
        on_trace_ready=tensorboard_trace_handler('log/'),  # Save results to TensorBoard logs
        record_shapes=True,
        with_stack=True,
        with_modules=True
    )


    def spectrogram_to_image(spectrogram):
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

    def train_one_epoch(model, loader, optimizer, epoch):
        model.train()
        total_loss = 0

        with profiler as prof:
            for batch_idx, (source_audio, target_audio, source_text, target_text) in enumerate(loader):
                # Move data to device
                source_audio, target_audio = source_audio.to(device), target_audio.to(device)
                source_text, target_text = source_text.to(device), target_text.to(device)

                if source_audio is None or target_audio is None:
                    print(f"Skipping batch {batch_idx} due to None tensors.")
                    continue

                # Check if source_audio is all zeros
                if torch.all(source_audio == 0) or torch.all(target_audio == 0):
                    print(f"Skipping batch {batch_idx}: source_audio is all zeros or corrupt")
                    continue

                # print(f"Source text max index: {source_text.max().item()}, min index: {source_text.min().item()}")
                # print(f"Target text max index: {target_text.max().item()}, min index: {target_text.min().item()}")

                if torch.isnan(source_audio).any() or torch.isinf(source_audio).any():
                    print(f"Skipping batch {batch_idx}: NaN or Inf detected in source_audio")
                    continue
                if torch.isnan(target_audio).any() or torch.isinf(target_audio).any():
                    print(f"Skipping batch {batch_idx}: NaN or Inf detected in target_audio")
                    continue

                optimizer.zero_grad()

                with autocast():
                    # Convert audio to mel spectrogram
                    source_mel = mel_converter(source_audio).permute(0, 2, 1)  # (batch, time_steps, n_mels)
                    target_mel = mel_converter(target_audio).permute(0, 2, 1)

                    mask = (target_mel != 0).float()
                    if torch.isnan(mask).any() or torch.isinf(mask).any():
                        print("Mask contains NaN or Inf!")
                        continue
                    if torch.isnan(target_mel).any() or torch.isinf(target_mel).any():
                        print("Target mel spectrogram contains NaN or Inf!")
                        continue

                    mask = (target_mel != 0).float()
                    if torch.isnan(mask).any() or torch.isinf(mask).any():
                        print("Mask contains NaN or Inf!")
                        continue
                    if torch.isnan(source_mel).any() or torch.isinf(source_mel).any():
                        print("Source mel spectrogram contains NaN or Inf!")
                        continue

                    # print(f"Source mel spectrogram shape: {source_mel.shape}")
                    # print(f"Target mel spectrogram shape: {target_mel.shape}")

                    # Forward pass
                    output_waveform, _, aux_source, aux_target = model(source_mel)
                    # print(output_waveform.shape)
                    # torch.Size([32, 53504])

                    # Before normalization
                    # print(f"Output waveform stats (before normalization): min={output_waveform.min().item()}, max={output_waveform.max().item()}, mean={output_waveform.mean().item()}")

                    # Normalize to [-1, 1]
                    # output_waveform = output_waveform / torch.abs(output_waveform).max()

                    # After normalization
                    # print(f"Output waveform stats (after normalization): min={output_waveform.min().item()}, max={output_waveform.max().item()}, mean={output_waveform.mean().item()}")

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

                    # # Clamp or mask out-of-bounds indices
                    # num_classes = aux_target.size(-1)  # 5356
                    # target_text = target_text.clamp(0, num_classes - 1)  # Alternatively: Use masking

                    # print(f"Aligned aux source shape: {aux_source.shape}")
                    # print(f"Aligned source text shape: {source_text.shape}")
                    # print(f"Aligned aux target shape: {aux_target.shape}")
                    # print(f"Aligned target text shape: {target_text.shape}")

                    # num_classes_source = aux_source.size(-1)  # 82
                    # source_text = source_text.clamp(0, num_classes_source - 1)

                    # Define ignore index (e.g., -100 for padding tokens)
                    IGNORE_INDEX = 0

                    # Replace padded or invalid indices with ignore index
                    source_text = source_text.clone()
                    target_text = target_text.clone()

                    num_classes_source = aux_source.size(-1)  # 82
                    num_classes_target = aux_target.size(-1)  # 5356

                    # Mask out-of-bounds indices for source and target text
                    source_text[(source_text < 0) | (source_text >= num_classes_source)] = IGNORE_INDEX
                    target_text[(target_text < 0) | (target_text >= num_classes_target)] = IGNORE_INDEX

                    # Compute text losses with ignore_index
                    ce_loss_ignore = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

                    source_text_loss = ce_loss_ignore(
                        aux_source.reshape(-1, aux_source.size(-1)), source_text.reshape(-1))

                    target_text_loss = ce_loss_ignore(
                        aux_target.reshape(-1, aux_target.size(-1)), target_text.reshape(-1))

                    # Total loss
                    total_loss = waveform_loss + source_text_loss + target_text_loss

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print("***NaN or Inf detected in loss!***")
                        continue

                # Backpropagation
                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                if batch_idx % 10 == 0:
                    log_message = (f'Epoch {epoch}, Batch {batch_idx}/{len(loader)}, '
                                   f'Waveform Loss: {waveform_loss.item():.4f}, '
                                   f'Source Text Loss: {source_text_loss.item():.4f}, '
                                   f'Target Text Loss: {target_text_loss.item():.4f}, '
                                   f'Total Loss: {total_loss.item():.4f}')
                    print(log_message)

                    with open(log_file, "a", buffering=1) as f:
                        f.write(log_message + "\n")

                if (batch_idx + 1) % 1000 == 0:  # every 1000 batches premature save
                    print(f'saving model premature: models/models_premature/translatotron_model_{epoch}_{batch_idx + 1}.pth ')
                    torch.save(model.state_dict(), f'models/models_premature/translatotron_model_{epoch}_{batch_idx + 1}.pth')

                if (batch_idx + 1) % 500 == 0:
                    print(f'logging to tensorboard')
                    # Log waveform to TensorBoard (First sample of the batch)
                    ten_wav = output_waveform / output_waveform.abs().max(dim=-1, keepdim=True)[0]
                    writer.add_audio(f'output_waveform/epoch_{epoch}_batch_{batch_idx}',
                                     ten_wav[0].detach().cpu(),
                                     sample_rate=16000,
                                     global_step=batch_idx)

                    writer.add_audio(f'target_waveform/epoch_{epoch}_batch_{batch_idx}',
                                     target_audio[0].detach().cpu(),
                                     sample_rate=16000,
                                     global_step=batch_idx)

                    # Log output mel spectrogram
                    output_mel_image = spectrogram_to_image(output_mel[0].detach().cpu())
                    writer.add_image(f'output_mel/epoch_{epoch}_batch_{batch_idx}',
                                     output_mel_image, global_step=batch_idx)

                    # Log target mel spectrogram
                    target_mel_image = spectrogram_to_image(target_mel[0].detach().cpu())
                    writer.add_image(f'target_mel/epoch_{epoch}_batch_{batch_idx}',
                                     target_mel_image, global_step=batch_idx)
                    print("done logging to tensorboard")

                prof.step()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof.stop()
        # Return average loss for the epoch
        return total_loss / len(loader)


    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, epoch)

        scheduler.step(avg_loss)

        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'models/translatotron_model_epoch_mature_{epoch + 1}.pth')

    print("Training completed!")