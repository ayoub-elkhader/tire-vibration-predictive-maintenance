"""Second approach of the internship subject: the spectrogram as an image.

A small convolutional network classifies the log-spectrogram of the hub
acceleration (0-250 Hz, 0.256 s frames), with no hand-made feature and no
speed information. Same vehicle-wise folds as the feature models.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import GroupKFold

from .scenarios import CLASSES


class SpecNet(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def train_eval(specs, labels, groups, epochs=25, batch=64, lr=2e-3, seed=0, n_splits=5):
    """Out-of-fold predictions (class names) of the CNN."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    idx = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([idx[c] for c in labels], dtype=np.int64)
    X = torch.from_numpy(specs[:, None])
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits).split(specs, y, groups):
        model = SpecNet()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xtr, ytr = X[tr], torch.from_numpy(y[tr])
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(tr))
            for k in range(0, len(tr), batch):
                b = perm[k:k + batch]
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(X[te][k:k + 256]) for k in range(0, len(te), 256)])
        pred[te] = np.array(CLASSES, dtype=object)[out.argmax(1).numpy()]
    return pred
