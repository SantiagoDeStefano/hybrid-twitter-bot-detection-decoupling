import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def build_mlp(input_dim: int, hidden_layers=(256, 64), dropout: float = 0.3):
    layers = []
    prev_dim = input_dim
    for h in hidden_layers:
        layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        prev_dim = h
    layers += [nn.Linear(prev_dim, 1), nn.Sigmoid()]
    return nn.Sequential(*layers)


def train_mlp(X_tr, y_tr, X_val, y_val, X_test, y_test=None,
              hidden_layers=(256, 64), lr=1e-2, dropout=0.3,
              epochs=200, patience=20, batch_size=1024, device="cuda",
              return_preds=False):
    model = build_mlp(X_tr.shape[1], hidden_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    X_v_t = torch.FloatTensor(X_val).to(device)
    X_te_t = torch.FloatTensor(X_test).to(device)

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    sample_weights = torch.where(
        y_tr_t == 1, torch.tensor(neg / pos).to(device), torch.tensor(1.0).to(device)
    )
    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t, sample_weights)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_f1, best_preds, best_test_f1, patience_counter = 0, None, 0, 0

    for _ in range(epochs):
        model.train()
        for X_b, y_b, w_b in loader:
            optimizer.zero_grad()
            pred = model(X_b).squeeze()
            loss = (criterion(pred, y_b) * w_b).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = (model(X_v_t).squeeze().cpu().numpy() >= 0.5).astype(int)
            val_f1 = f1_score(y_val, val_pred)
            test_pred = (model(X_te_t).squeeze().cpu().numpy() >= 0.5).astype(int)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_preds = test_pred
            if y_test is not None:
                best_test_f1 = f1_score(y_test, test_pred) * 100
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_preds if return_preds else best_test_f1
