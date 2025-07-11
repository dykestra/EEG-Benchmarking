import torch

from skorch.callbacks import LRScheduler
from braindecode.models import EEGNetv1, EEGInception
from braindecode import EEGClassifier

def get_model(model_name, n_chans=29, n_times=800, input_window_seconds=4, n_outputs=2, sfreq=200):
    if model_name == "EEGNetv1":
        model = EEGNetv1(
            n_chans=n_chans, 
            n_times=n_times, 
            input_window_seconds=input_window_seconds, 
            n_outputs=n_outputs
            )
    elif model_name == "EEGInception":
        model = EEGInception(
            n_chans=n_chans, 
            n_times=n_times, 
            input_window_seconds=input_window_seconds, 
            n_outputs=n_outputs,
            sfreq=sfreq
        )
    else:
        print(f"Undefined model name: {model_name}")

    return model

def get_net(model, lr=0.001, weight_decay=0, n_epochs=100, batch_size=64, train_split=None):
    return EEGClassifier(
        model,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        train_split=train_split,
        callbacks=["accuracy",
                   ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1))],
        device="cuda",
        max_epochs=n_epochs
    )