from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
import torch 
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExoplanetRNN():
    """
    Classifies light curves based on if they belong to star with a transiting exoplanet.
    """
    def __init__(self, hidden_size=32, input_size=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        H = hidden_size
        self.H = H
        self.scale = H ** -0.5  # init range 1/sqrt(H): small enough not to blow up activations

        # Update gate
        self.W_xz = self._make_param(input_size, H)
        self.W_hz = self._make_param(H, H)
        self.b_z  = self._make_param(H)

        # Reset gate
        self.W_xr = self._make_param(input_size, H)
        self.W_hr = self._make_param(H, H)
        self.b_r  = self._make_param(H)

        # Candidate
        self.W_xn = self._make_param(input_size, H)
        self.W_hn = self._make_param(H, H)
        self.b_n  = self._make_param(H)
        
        # Output layer
        self.W_o = self._make_param(H, 1)
        self.b_o = self._make_param(1)

    def _make_param(self, *shape):
        """Initialize a parameter tensor with uniform values in [-scale, scale]."""
        return torch.empty(*shape, device=DEVICE).uniform_(-self.scale, self.scale).requires_grad_(True)
    
    def forward(self, x):   # input x: (B, T, 1)
        # hidden "memory" starts empty (zeros); one row per star in the batch
        h = torch.zeros(x.shape[0], self.H, device=x.device)
        for t in range(x.shape[1]):  
          xt = x[:, t, :]  
          z = torch.sigmoid(xt @ self.W_xz + h @ self.W_hz + self.b_z) 
          r = torch.sigmoid(xt @ self.W_xr + h @ self.W_hr + self.b_r) 
          n = torch.tanh(xt @ self.W_xn + (r * h) @ self.W_hn + self.b_n) 
          h = (1 - z) * n + z * h  
        return (h @ self.W_o + self.b_o)[:, 0] 
    
    def __call__(self, x):
        # Allows model(x) to invoke forward(x), matching PyTorch convention
        return self.forward(x)
    
    def parameters(self):
        return [self.W_xz, self.W_hz, self.b_z, self.W_xr, self.W_hr, self.b_r, 
                self.W_xn, self.W_hn, self.b_n, self.W_o, self.b_o]

def downsample(X, factor=8):
    """Average-pool the sequence length: (N, T) -> (N, T//factor).
    This helps to speed up the training due to fewer timesteps."""
    N, T = X.shape
    T2 = (T // factor) * factor     
    return X[:, :T2].reshape(N, T2 // factor, factor).mean(axis=2)

def bce_loss(logits, y, pos_weight=1.0, eps=1e-7):
    """Binary cross-entropy with a positive-class weight, by hand (no nn loss)."""
    # clamp avoids log(0) = -inf -> NaN 
    p = torch.sigmoid(logits).clamp(eps, 1 - eps)   
    return -(pos_weight * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()


def sgd_step(params, lr):
    """One manual SGD update over all parameters: w = w - lr * grad."""
    with torch.no_grad():
        for w in params:
            w -= lr * w.grad
            w.grad.zero_()

def train_rnn(light_curves, labels, epochs, batch_size, learning_rate):
    x = torch.tensor(light_curves, dtype = torch.float32)
    y = torch.tensor(labels.values, dtype = torch.float32)
    x = x[:, :, None]

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    model = ExoplanetRNN()                    # weights already created on DEVICE

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            loss = bce_loss(model(batch_x), batch_y)        
            loss.backward()                                 # autograd fills .grad
            sgd_step(model.parameters(), learning_rate)     # manual SGD (see helper func above)
            total_loss += loss.item() * len(batch_x)
        average_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}: loss = {average_loss:.4f}")

    return model

def evaluate(model, light_curves):
    x = torch.tensor(light_curves, dtype=torch.float32)
    x = x[:, :, None].to(DEVICE)

    with torch.no_grad():
        return torch.sigmoid(model(x)).cpu().numpy()

    labels_array = labels.values if hasattr(labels, 'values') else np.asarray(labels)
    return float(np.mean(predictions == labels_array)) 

def scores(y_true, prob, threshold=0.5):
    y_true = np.asarray(y_true)
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy":  float((pred == y_true).mean()),
        "recall":    recall_score(y_true, pred),               
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1":        f1_score(y_true, pred),
        "roc_auc":   roc_auc_score(y_true, prob),
        "pr_auc":    average_precision_score(y_true, prob),    
      }

if __name__ == "__main__":
    df = pd.read_csv("data/rnn_timeseries.csv")
    flux_columns = [col for col in df.columns if "flux" in col]
    flux_columns.sort(key=lambda col: int(col.split("_")[1]))
    labels = df["label"]

    print("Label counts:")
    print(df["label"].value_counts())

    majority_baseline = df["label"].value_counts(normalize=True).max()
    print(f"Majority-class baseline: {majority_baseline:.3f}")

    light_curves = downsample(np.clip(df[flux_columns].to_numpy(np.float32), -1, 1), factor=8)

    x_train, x_test, y_train, y_test = train_test_split(
        light_curves,
        labels,
        test_size=0.2,
        random_state=0,
        stratify=labels,
    )

    model = train_rnn(
        light_curves=x_train,
        labels=y_train,
        epochs=10,
        batch_size=64,
        learning_rate=0.3
    )
    prob = evaluate(model, x_test)
    for k, v in scores(y_test, prob).items():
        print(f"  {k:10s} {v:.3f}")
    print(f"  (majority-acc baseline = {majority_baseline:.3f},  "
          f"PR-AUC no-skill = {float(np.mean(y_test)):.3f})")
    
    test_accuracy = evaluate(model, x_test)
    print(f"Test accuracy: {test_accuracy:.3f}")

    torch.save([w.detach().cpu() for w in model.parameters()], "exoplanet_rnn.pt")  
