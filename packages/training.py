## usage: python training.py --input 102_training_data_sequece_exp.pt --epochs 30 --model seq_exp --output 102_model_exp.pt


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

    # ----- Constants -----

L = 32
# convolution window size in residual units
W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                21, 21, 21, 21, 41, 41, 41, 41])
# atrous rate in residual units
AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                 10, 10, 10, 10, 25, 25, 25, 25])

class ResBlock(nn.Module):
    def __init__(self, L, W, AR, pad=True):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(L)
        s = 1
        # padding calculation: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
        if pad:
            padding = int(1 / 2 * (1 - L + AR * (W - 1) - s + L * s))
        else:
            padding = 0
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)
        self.bn2 = nn.BatchNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)

    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return out

class PangolinEXP(nn.Module):
    def __init__(self, L, W, AR):
        super(PangolinEXP, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(14, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 2, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        self.conv_last3 = nn.Conv1d(L, 2, 1)
        self.conv_last4 = nn.Conv1d(L, 1, 1)
        self.conv_last5 = nn.Conv1d(L, 2, 1)
        self.conv_last6 = nn.Conv1d(L, 1, 1)
        self.conv_last7 = nn.Conv1d(L, 2, 1)
        self.conv_last8 = nn.Conv1d(L, 1, 1)

    def forward(self, x):
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        CL = 2 * np.sum(AR * (W - 1))
        skip = F.pad(skip, (-CL // 2, -CL // 2))
        out1 = F.softmax(self.conv_last1(skip), dim=1)
        out2 = torch.sigmoid(self.conv_last2(skip))
        out3 = F.softmax(self.conv_last3(skip), dim=1)
        out4 = torch.sigmoid(self.conv_last4(skip))
        out5 = F.softmax(self.conv_last5(skip), dim=1)
        out6 = torch.sigmoid(self.conv_last6(skip))
        out7 = F.softmax(self.conv_last7(skip), dim=1)
        out8 = torch.sigmoid(self.conv_last8(skip))
        return torch.cat([out1, out2, out3, out4, out5, out6, out7, out8], 1)



def main():
    parser = argparse.ArgumentParser(description="Train Pangolin model on PSI data")
    parser.add_argument('--input', type=str, required=True, help="Path to training data (.pt)")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--model', type=str, choices=["seq", "exp"], default="exp",
                        help="Model type: 'seq' for Pangolin, 'exp' for PangolinEXP")
    parser.add_argument('--output', type=str, default="trained_model.pt", help="Output file name for model")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----- Load Data -----
    data = torch.load(args.input, weights_only=True)
    training_label = torch.stack(data['y'])

    # ----- Model -----
    if args.model == "seq":
        model = Pangolin(L=L, W=W, AR=AR).to(device)
        training_input = torch.stack(data['X'])[:, :4, :].to(device)  # Ensure only 4 input channels
    elif args.model == "exp":
        model = PangolinEXP(L=L, W=W, AR=AR).to(device)
        training_input = torch.stack(data['X'])[:, :, :].to(device)  # Ensure only 4 input channels


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
