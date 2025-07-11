import os
import numpy as np
import json
import argparse

import skorch
from skorch.helper import predefined_split
from torch.utils.data import Subset

from data import BENCHMARK_LOADERS
from utils import get_model, get_net

def perform_benchmarking(args):
    results = {}
    for benchmark, load_fn in BENCHMARK_LOADERS.items():
        # Load data
        X, sbj_id, y, splits = load_fn(args.data_root)
        dataset = skorch.dataset.Dataset(X, y)
        sbj_id_unique = np.unique(sbj_id)
        n_outputs = len(np.unique(y))
        n, c, t = X.shape
        
        results[benchmark] = {}
        all_val_accs = []
        for _, sbj_id_val in splits.items():
            sbj_id_train = [s for s in sbj_id_unique if s not in sbj_id_val]

            # Split data by subjects
            ix_train = np.arange(n)[np.isin(sbj_id, sbj_id_train)]
            ix_val = np.arange(n)[np.isin(sbj_id, sbj_id_val)]
            train_dataset = Subset(dataset, ix_train)
            validation_dataset = Subset(dataset, ix_val)
            
            # Make model
            model = get_model(
                args.model_name, 
                n_chans=c, 
                sfreq=args.fs,                    
                n_times=t, 
                input_window_seconds=t//args.fs, 
                n_outputs=n_outputs
                )
            net = get_net(
                model, 
                batch_size=args.batch_size,
                train_split=predefined_split(validation_dataset)
                )

            # Train
            net.fit(train_dataset, y=None, epochs=args.n_epochs)
            
            all_val_accs.append(net.history[:, 'valid_accuracy'])

            # Save mean accuracy
            mean_acc = np.mean(np.array(all_val_accs)[:,-1])
            print(f"{args.model_name} | {benchmark}: {mean_acc}")

            results[benchmark]['mean'] = mean_acc
            results[benchmark]['fold_accs'] = list(np.array(all_val_accs)[:,-1])

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--n-splits', default=10, type=int, help="number of cross-fold validation splits")
    parser.add_argument('--data-root', type=str, help="path to training data directory")
    parser.add_argument('--batch-size', default=64, type=int, help="batch size during training")
    parser.add_argument('--n-epochs', default=20, type=int, help="number of training epochs")
    parser.add_argument('--fs', default=200, type=int, help="sample frequency of eeg data (Hz)")
    parser.add_argument('--model-name', default='EEGNetv1', type=str, help="name of model to be fine-tuned")
    parser.add_argument('--output-dir', default='results', type=str, help="output directory to save results")
    parser.add_argument('--ex-id', default=0, type=int, help="experiment number")
    args = parser.parse_args()

    results = perform_benchmarking(args)

    # Save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, f'results_{args.ex_id}.txt'), 'w') as f: 
        f.write(json.dumps(results))

        
