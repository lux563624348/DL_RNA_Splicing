# visual_utils.py
## VISUAL Module

import matplotlib.pyplot as plt
import numpy as np

def plot_one_hot(array, title="One-Hot Encoded Sequence"):
    seq_len = array.shape[0]
    plt.figure(figsize=(12, 3))
    plt.imshow(array.T, aspect='auto', cmap='Greys', interpolation='nearest')
    
    # Y-axis: bases
    plt.yticks(ticks=[0, 1, 2, 3], labels=['A', 'C', 'G', 'T'])

    # X-axis: sequence positions (e.g., 1 to 1500)
    step = max(seq_len // 20, 1)  # show ~20 ticks
    xtick_positions = list(range(0, seq_len, step))
    xtick_labels = [str(i + 1) for i in xtick_positions]
    plt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=90)

    plt.xlabel("Position in sequence")
    plt.title(title)
    plt.colorbar(label="One-hot value")
    plt.tight_layout()
    plt.show()

def plot_one_hot_and_labels_zoom(one_hot_array, label_array, zoom_start=0, zoom_end=None, title="Zoomed Sequence and Labels"):
    """
    Plot one-hot encoded sequence and label data for a zoomed-in region.

    Args:
        one_hot_array: (4, sequence_length) numpy array
        label_array: (12, label_length) numpy array
        zoom_start: start position of zoom window in input coordinates
        zoom_end: end position of zoom window in input coordinates
        title: plot title
    """
    seq_len = one_hot_array.shape[1]
    label_len = label_array.shape[1]
    input_start_for_labels = (seq_len - label_len) // 2

    if zoom_end is None:
        zoom_end = seq_len
    zoom_len = zoom_end - zoom_start

    # --- Slice one-hot array ---
    one_hot_zoom = one_hot_array[:, zoom_start:zoom_end]  # shape (4, zoom_len)

    # --- Calculate label slice range ---
    label_start_idx = max(0, zoom_start - input_start_for_labels)
    label_end_idx = max(0, zoom_end - input_start_for_labels)
    label_zoom_len = label_end_idx - label_start_idx

    if label_start_idx >= label_len:
        label_zoom = np.zeros((12, zoom_len))
        label_positions = np.arange(zoom_start, zoom_end)
    else:
        label_zoom = label_array[:, label_start_idx:label_end_idx]
        label_positions = np.arange(input_start_for_labels + label_start_idx,
                                    input_start_for_labels + label_end_idx)

    # Extract specific label channels
    unspliced = label_zoom[0, :]
    spliced = label_zoom[1, :]
    usage = label_zoom[2, :] if label_zoom.shape[1] > 0 else np.zeros(label_zoom.shape[1])

    # Convert class labels: 0 = unknown, 1 = unspliced, 2 = spliced
    class_labels = []
    for u, s in zip(unspliced, spliced):
        if u == 1 and s == 0:
            class_labels.append(1)
        elif u == 0 and s == 1:
            class_labels.append(2)
        else:
            class_labels.append(0)

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    # One-hot sequence
    axes[0].imshow(one_hot_zoom, aspect='auto', cmap='Greys', interpolation='nearest',
                   extent=[zoom_start, zoom_end, 0, 4])
    axes[0].set_yticks([0.5, 1.5, 2.5, 3.5])
    axes[0].set_yticklabels(['A', 'C', 'G', 'T'])
    axes[0].set_title("One-Hot Encoded Sequence")

    # Splice class
    axes[1].imshow([class_labels], aspect='auto', cmap='bwr', interpolation='nearest',
                   extent=[label_positions[0] if len(label_positions) > 0 else zoom_start, 
                           label_positions[-1] + 1 if len(label_positions) > 0 else zoom_end, 0, 1])
    axes[1].set_yticks([0.5])
    axes[1].set_yticklabels(["Splice Class"])
    axes[1].set_title("Splice Class (0=unknown, 1=unspliced, 2=spliced)")

    # Usage
    if label_zoom.shape[1] > 0:
        axes[2].plot(label_positions, usage[:label_zoom_len], color='blue')
    else:
        axes[2].plot(np.arange(zoom_start, zoom_end), np.zeros(zoom_len), color='gray')
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Usage")
    axes[2].set_title("Estimated Usage Level")

    # X ticks
    step = max(zoom_len // 20, 1)
    xtick_positions = list(range(zoom_start, zoom_end, step))
    xtick_labels = [str(i) for i in xtick_positions]
    axes[2].set_xticks(xtick_positions)
    axes[2].set_xticklabels(xtick_labels, rotation=90)
    axes[2].set_xlabel("Position in Sequence")

    plt.tight_layout()
    plt.suptitle(title, y=1.03)
    plt.show()
    return None

def plot_one_hot_and_predictions_zoom(one_hot_array, prediction_array, zoom_start=8600, zoom_end=9001, title="Zoomed Prediction View"):
    """
    Plot one-hot encoded input and model predictions for a zoomed region.

    Args:
        one_hot_array: (4, 15000) array, one-hot encoded input sequence (A, C, G, T)
        prediction_array: (12, 5000) array, model output aligned to positions 5000:10000
        zoom_start: start position in input coordinates (0–14999)
        zoom_end: end position in input coordinates
    """
    full_seq_len = one_hot_array.shape[1]
    pred_len = prediction_array.shape[1]
    pred_start = (full_seq_len - pred_len) // 2  # e.g., 5000
    pred_end = pred_start + pred_len             # e.g., 10000

    if zoom_start < 0 or zoom_end > full_seq_len:
        raise ValueError("Zoom region is outside of input sequence bounds.")

    zoom_len = zoom_end - zoom_start

    # --- One-hot ---
    one_hot_zoom = one_hot_array[:, zoom_start:zoom_end]  # shape (4, zoom_len)

    # --- Predictions ---
    pred_start_idx = max(0, zoom_start - pred_start)
    pred_end_idx = max(0, zoom_end - pred_start)

    if pred_start_idx >= pred_len or pred_end_idx > pred_len:
        pred_zoom = np.zeros((12, zoom_len))
        pred_positions = np.arange(zoom_start, zoom_end)
    else:
        pred_zoom = prediction_array[:, pred_start_idx:pred_end_idx]  # (12, L)
        pred_positions = np.arange(pred_start + pred_start_idx, pred_start + pred_end_idx)

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    # One-hot sequence
    axes[0].imshow(one_hot_zoom, aspect='auto', cmap='Greys', interpolation='nearest',
                   extent=[zoom_start, zoom_end, 0, 4])
    axes[0].set_yticks([0.5, 1.5, 2.5, 3.5])
    axes[0].set_yticklabels(['A', 'C', 'G', 'T'])
    axes[0].set_title("One-Hot Encoded Sequence")

    # Predicted splice class
    if pred_zoom.shape[1] > 0:
        unspliced = pred_zoom[0, :]
        spliced = pred_zoom[1, :]
        class_labels = []
        for u, s in zip(unspliced, spliced):
            if (u - s) > 0.5:
                class_labels.append(1)
            elif (s - u) > 0.5:
                class_labels.append(2)
            else:
                class_labels.append(0)
        axes[1].imshow([class_labels], aspect='auto', cmap='bwr', interpolation='nearest',
                       extent=[pred_positions[0], pred_positions[-1] + 1, 0, 1])
    else:
        axes[1].imshow([[0] * zoom_len], aspect='auto', cmap='bwr', interpolation='nearest',
                       extent=[zoom_start, zoom_end, 0, 1])
    axes[1].set_yticks([0.5])
    axes[1].set_yticklabels(["Splice Class"])
    axes[1].set_title("Predicted Splice Class (0=unknown, 1=unspliced, 2=spliced)")

    # Predicted usage
    if pred_zoom.shape[1] > 0:
        usage = pred_zoom[2, :]
        axes[2].plot(pred_positions, usage, color='blue')
    else:
        axes[2].plot(np.arange(zoom_start, zoom_end), np.zeros(zoom_len), color='gray')
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Usage")
    axes[2].set_title("Predicted Usage Level")

    # X ticks
    step = max(zoom_len // 10, 1)
    xtick_positions = list(range(zoom_start, zoom_end, step))
    xtick_labels = [str(i) for i in xtick_positions]
    axes[2].set_xticks(xtick_positions)
    axes[2].set_xticklabels(xtick_labels, rotation=90)
    axes[2].set_xlabel("Position in Sequence")

    plt.tight_layout()
    plt.suptitle(title, y=1.03, fontsize=14)
    plt.show()
    return None
