import matplotlib.pyplot as plt
import os
import time

# Function to parse log file and extract data
def parse_log_file(log_file):
    batches = []
    waveform_losses = []
    source_text_losses = []
    target_text_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            if "Waveform Loss" in line:
                parts = line.strip().split(", ")
                batch = int(parts[1].split("/")[0].split()[-1])
                waveform_loss = float(parts[2].split(": ")[-1])
                source_text_loss = float(parts[3].split(": ")[-1])
                target_text_loss = float(parts[4].split(": ")[-1])

                batches.append(batch)
                waveform_losses.append(waveform_loss)
                source_text_losses.append(source_text_loss)
                target_text_losses.append(target_text_loss)

    return batches, waveform_losses, source_text_losses, target_text_losses

# Function to plot the data
def plot_losses(batches, waveform_losses, source_text_losses, target_text_losses):
    plt.figure(figsize=(10, 6))

    # Plot Waveform Loss
    plt.subplot(2, 1, 1)
    plt.plot(batches, waveform_losses, label="Waveform Loss", linestyle='-', marker='.')
    plt.xlabel("Batch Index")
    plt.ylabel("Loss")
    plt.title("Waveform Loss over Batches")
    plt.legend()
    plt.grid()

    # Plot Source and Target Text Losses
    plt.subplot(2, 1, 2)
    plt.plot(batches, source_text_losses, label="Source Text Loss", linestyle='--', marker='.')
    plt.plot(batches, target_text_losses, label="Target Text Loss", linestyle=':', marker='.')
    plt.xlabel("Batch Index")
    plt.ylabel("Loss")
    plt.title("Text Losses (Source and Target) over Batches")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# File path to the training log
log_dir = "log/raw_loss_logs/"
log_file = os.path.join(log_dir, "training_log_run.txt")

# Ensure the log directory exists
if not os.path.exists(log_dir):
    raise FileNotFoundError(f"Log directory not found at: {log_dir}")

print("Monitoring log file for updates...")

# Continuously monitor the log file and update the plot
last_size = 0
while True:
    try:
        # Check if the file size has changed
        current_size = os.path.getsize(log_file)
        if current_size > last_size:
            # Parse the updated log file
            batches, waveform_losses, source_text_losses, target_text_losses = parse_log_file(log_file)

            # Plot the updated data
            plot_losses(batches, waveform_losses, source_text_losses, target_text_losses)

            last_size = current_size

        time.sleep(5)  # Wait for 5 seconds before checking again
    except KeyboardInterrupt:
        print("Exiting monitoring...")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
