import torch 
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class ExoplanetRNN(nn.Module):
    """
    Classifies light curves based on if they belong to star with a transiting exoplanet.
    """
    def __init__(self, hidden_size: int = 64):
        super().__init__()

        self.gru = nn.GRU(
            input_size = 1,
            hidden_size = hidden_size,
            num_layers = 1,
            batch_first = True,
        )

        self.classifier = nn.Linear(
            in_features = hidden_size,
            out_features = 1
        )

    def forward(self, x: torch.Tensor):
        _, hidden = self.gru(x)
        final_hidden_state = hidden[-1]
        logits = self.classifier(final_hidden_state)

        return logits[:, 0]
    
def train_rnn(light_curves, labels, epochs, batch_size, learning_rate):
    x = torch.tensor(light_curves, dtype = torch.float32)
    y = torch.tensor(labels, dtype = torch.float32)
    x = x[:, :, None]

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExoplanetRNN().to(device)

    # Use BCEwithLogits to avoid trouble rounding at extremes with sigmoid(high_number)
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        average_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}: loss = {average_loss:.4f}")

    return model