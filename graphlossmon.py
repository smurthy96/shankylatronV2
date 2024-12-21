import matplotlib.pyplot as plt
import os


# Function to parse log file and extract data
def parse_log_file(log_file):
    batches = []
    waveform_losses = []
    total_loss_batches = []
    target_text_losses = []

    cumulative_batches = 0  # Track the total cumulative batches
    previous_epoch = -1  # To detect when a new epoch starts

    with open(log_file, 'r') as f:
        for line in f:
            if "Waveform Loss" in line:
                parts = line.strip().split(", ")
                current_epoch = int(parts[0].split(" ")[1])  # Extract epoch
                batch = int(parts[1].split("/")[0].split()[-1])  # Extract batch number
                waveform_loss = float(parts[2].split(": ")[-1])
                total_loss = float(parts[4].split(": ")[-1])
                target_text_loss = float(parts[3].split(": ")[-1])

                # If epoch changes, reset the batch and increment cumulative_batches
                if current_epoch != previous_epoch:
                    previous_epoch = current_epoch
                    cumulative_batches = batches[-1] if batches else 0

                cumulative_batch_index = cumulative_batches + batch

                batches.append(cumulative_batch_index)
                waveform_losses.append(waveform_loss)
                total_loss_batches.append(total_loss)
                target_text_losses.append(target_text_loss)

    return batches, waveform_losses, target_text_losses, total_loss_batches


# File path to the training log
log_dir = "log\\raw_loss_logs"
log_file = os.path.join(log_dir, "training_log_run.txt")

# Ensure the log file exists
if not os.path.exists(log_file):
    raise FileNotFoundError(f"Log file not found at: {log_file}")

# Parse the log file
batches, waveform_losses, target_text_losses, total_loss = parse_log_file(log_file)

# Plotting
plt.figure(figsize=(20, 15))

# Plot Waveform Loss
plt.subplot(3, 1, 1)
plt.plot(batches, waveform_losses, label="Waveform Loss", linestyle='-', marker='.')
plt.xlabel("Cumulative Batch Index")
plt.ylabel("Loss")
plt.title("Waveform Loss over Batches")
plt.legend()
plt.grid()

# Plot Target Text Loss
plt.subplot(3, 1, 2)
plt.plot(batches, target_text_losses, label="Target Text Loss", linestyle=':', marker='.')
plt.xlabel("Cumulative Batch Index")
plt.ylabel("Loss")
plt.title("Target Text Loss over Batches")
plt.legend()
plt.grid()

# # Plot Source Text Loss
plt.subplot(3, 1, 3)
plt.plot(batches, total_loss, label="Source Text Loss", linestyle='--', marker='.')
plt.xlabel("Cumulative Batch Index")
plt.ylabel("Loss")
plt.title("Total Loss")
plt.legend()
plt.grid()



plt.tight_layout()
plt.show()
