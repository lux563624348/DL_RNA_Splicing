import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

from packages.model import Pangolin, PangolinEXP

def main():
    parser = argparse.ArgumentParser(description="Train Pangolin model on PSI data")
    parser.add_argument('--input', type=str, required=True, help="Path to training data (.pt)")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--model', type=str, choices=["seq", "seq_exp"], default="seq_exp",
                        help="Model type: 'seq' for Pangolin, 'exp' for PangolinEXP")
    parser.add_argument('--output', type=str, default="trained_model.pt", help="Output file name for model")

    args = parser.parse_args()

    # ----- Constants -----
    L = 32
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21, 41, 41, 41, 41])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10, 25, 25, 25, 25])

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Model -----
    if args.model == "seq":
        model = Pangolin(L=L, W=W, AR=AR).to(device)
        training_input = torch.stack(data['X'])[:, :4, :].to(device)  # Ensure only 4 input channels
    elif args.model == "exp":
        model = PangolinEXP(L=L, W=W, AR=AR).to(device)
        training_input = torch.stack(data['X'])[:, :, :].to(device)  # Ensure only 4 input channels

    # ----- Load Data -----
    data = torch.load(args.input, weights_only=True)
    training_label = torch.stack(data['y'])

    # Pad to [N, 12, 15000], then crop [N, 12, 5000] from middle
    training_label_padded = F.pad(training_label, pad=(0, 0, 0, 9))
    training_label_corped = training_label_padded[:, :, 5000:10000]

    dataset = TensorDataset(training_input, training_label_corped.to(device))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ----- Training Loop -----
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)  # shape: (batch, 15000, 3)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.5f}")

    # ----- Save Model -----
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()
