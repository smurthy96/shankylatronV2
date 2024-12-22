import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio import transforms as T
from translatotron.transformatron2 import Translatotron2
from translatotron import SpeechToSpeechDatasetPhonemePreProcessed
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms


# Hyperparameters
batch_size = 32
num_epochs = 100
learning_rate = .0002

if __name__ == "__main__":
    device = torch.device("cuda")

    # summary(model, [(1, 80, 256), (1, 80)])  # Example input shapes for source_mel and lengths

    log_dir = "log/raw_loss_logs/"
    log_file = os.path.join(log_dir, "training_log_run.txt")

    os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(log_file):
        open(log_file, "a").close()

    writer = SummaryWriter(log_dir="log/raw_loss_logs/tensorboard")

    model = Translatotron2(
        input_dim=80,
        encoder_hidden_dim=512,
        encoder_layers=10,
        phoneme_dim=126657,
        output_dim=80,
        writer=writer
    ).to(device)

    mse_loss = nn.MSELoss(reduction="sum")
    ce_loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    source_dir_clips = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\clips"
    target_dir_clips = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\clips_translated"
    source_tsv_file = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_gendered.tsv"
    target_tsv_file = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_translated.tsv"
    tsv_file = "S:\\training_data\\cv-corpus-19.0-2024-09-13\\en\\train_combined_phonemes.tsv"

    train_dataset = SpeechToSpeechDatasetPhonemePreProcessed(
        root_dir_source=source_dir_clips,
        root_dir_target=target_dir_clips,
        tsv_file=tsv_file,
        concat_aug=True,
        concat_prob=0.3
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=15, pin_memory=True, drop_last=True)

    mel_converter = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80).to(device)


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


    def log_gradients_to_tensorboard(model, writer, epoch, batch_idx):
        """
        Logs the gradients of the model parameters to TensorBoard.
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, global_step=epoch * 1000 + batch_idx)


    def safe_normalize(tensor, epsilon=1e-6):
        mask = torch.isfinite(tensor)
        mean = (tensor * mask).sum(dim=(1, 2), keepdim=True) / (mask.sum(dim=(1, 2), keepdim=True) + epsilon)
        std = torch.sqrt(((tensor - mean) ** 2 * mask).sum(dim=(1, 2), keepdim=True) / (
                    mask.sum(dim=(1, 2), keepdim=True) + epsilon))
        normalized_tensor = (tensor - mean) / (std + epsilon)
        return torch.where(torch.isfinite(normalized_tensor), normalized_tensor, torch.zeros_like(normalized_tensor))

    def train_one_epoch(model, loader, optimizer, epoch):
        model.train()
        total_loss = 0


        for batch_idx, (source_audio, target_audio, source_text, target_text) in enumerate(loader):
            source_audio, target_audio = source_audio.to(device), target_audio.to(device)
            source_text, target_text = source_text.to(device), target_text.to(device)

            if source_audio is None or target_audio is None:
                continue
            if torch.all(source_audio == 0) or torch.all(target_audio == 0):
                torch.cuda.empty_cache()  # Optional, to release cached memory
                continue

            optimizer.zero_grad()

            with autocast():
                source_audio = source_audio.unsqueeze(1)
                source_mel = mel_converter(source_audio.float()).squeeze(1).permute(0, 2, 1).float()
                target_audio = target_audio.unsqueeze(1)
                target_mel = mel_converter(target_audio.float()).squeeze(1).permute(0, 2, 1).float()

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

                source_lengths = torch.full((source_mel.size(0),), source_mel.size(1), device=device)

                source_mel = safe_normalize(source_mel)

                waveform, phoneme_output = model(source_mel, source_lengths, batch_idx, epoch)

                output_mel = mel_converter(waveform.float().unsqueeze(1)).squeeze(1).permute(0, 2, 1).float()

                # Align lengths for ConcatAug support
                max_time_steps = max(output_mel.size(1), target_mel.size(1))
                output_mel = torch.nn.functional.pad(output_mel, (0, 0, 0, max_time_steps - output_mel.size(1)))
                target_mel = torch.nn.functional.pad(target_mel, (0, 0, 0, max_time_steps - target_mel.size(1)))

                mask = torch.isfinite(target_mel).float()

                output_mel = safe_normalize(output_mel)
                target_mel = safe_normalize(target_mel)

                if torch.isnan(output_mel).any() or torch.isinf(output_mel).any():
                    print("Output mel spectrogram contains NaN or Inf!")
                    # print(output_mel)
                    continue

                if torch.isnan(target_mel).any() or torch.isinf(target_mel).any():
                    print("Target mel spectrogram contains NaN or Inf!")
                    # print(target_mel)
                    continue

                # Compute normalized loss
                waveform_loss = (mse_loss(output_mel * mask, target_mel * mask)) / (mask.sum() + 1e-10)

                # Align lengths between phoneme_output and target_text

                min_seq_length = min(phoneme_output.size(1), target_text.size(1))
                phoneme_output = phoneme_output[:, :min_seq_length, :]
                target_text = target_text[:, :min_seq_length]

                # Flatten for loss computation
                phoneme_output_flat = phoneme_output.reshape(-1, phoneme_output.size(
                    -1))  # Shape: [batch_size * time_steps, n_classes]
                target_text_flat = target_text.reshape(-1).long()  # Shape: [batch_size * time_steps]

                target_text_flat = torch.clamp(target_text_flat, min=0, max=phoneme_output_flat.size(-1) - 1)

                # Compute loss
                IGNORE_INDEX = 0
                ce_loss_ignore = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                target_text_loss = ce_loss_ignore(phoneme_output_flat, target_text_flat)

                total_loss = waveform_loss + target_text_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    if torch.isnan(total_loss):
                        print("***NaN detected in loss!***")
                        print(waveform_loss)
                        print("-")
                        print(target_text_loss)
                    if torch.isinf(total_loss):
                        print("***Inf detected in loss!***")
                        print(total_loss)
                    torch.cuda.empty_cache()  # Optional, to release cached memory
                    del source_mel, target_mel, output_mel, phoneme_output_flat, target_text_flat
                    continue

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 10 == 0:
                log_message = (f'Epoch {epoch}, Batch {batch_idx}/{len(loader)}, '
                               f'Waveform Loss: {waveform_loss.item():.4f}, '
                               f'Target Text Loss: {target_text_loss.item():.4f}, '
                               f'Total Loss: {total_loss.item():.4f}')
                print(log_message)

                with open(log_file, "a", buffering=1) as f:
                    f.write(log_message + "\n")

            if (batch_idx + 1) % 10000 == 0:  # every 10000 batches premature save
                torch.save(model.state_dict(),
                           f'models/models_premature/translatotron2_model_{epoch}_{batch_idx + 1}.pth')

            if (batch_idx + 1) % 100 == 0:
                # Log waveform to TensorBoard (First sample of the batch)
                writer.add_audio(f'output_waveform/epoch_{epoch}_batch_{batch_idx}',
                                 waveform[0].detach().cpu(),
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

                # Convert index arrays to string for logging
                phoneme_output_flat = " ".join(map(str, phoneme_output_flat.detach().cpu().numpy()))
                target_text_flat = " ".join(map(str, target_text_flat.detach().cpu().numpy()))

                # Log source and target text indices
                writer.add_text(f'phoneme_output_flat/epoch_{epoch}_batch_{batch_idx}',
                                f"Source Indices: {phoneme_output_flat}",
                                global_step=batch_idx)

                writer.add_text(f'target_text_flat/epoch_{epoch}_batch_{batch_idx}',
                                f"Target Indices: {target_text_flat}",
                                global_step=batch_idx)
            if (batch_idx + 1) % 1000 == 0:
                try:
                    # print("logging gradients")
                    # log_gradients_to_tensorboard(model, writer, epoch, batch_idx)
                    # print("done logging gradients")
                    print("logging weights")
                    for name, param in model.named_parameters():
                        writer.add_histogram(f'Weights/{name}', param, global_step=batch_idx)
                    print("done logging weights")
                except Exception as e:
                    print(e)

        return total_loss / len(loader)

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'models/translatotron2_model_epoch_{epoch + 1}.pth')

    print("Training completed!")
