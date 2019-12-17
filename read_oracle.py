

import torch
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
parser.add_argument('--dataset', type=str, default='cifar10', help='Name of the dataset used.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size used for training and testing')
parser.add_argument('--train_iterations', type=int, default=100000, help='Number of training iterations')
parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')
parser.add_argument('--data_path', type=str, default='./data', help='Path to where the data is')
parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
parser.add_argument('--num_adv_steps', type=int, default=1,
                    help='Number of adversary steps taken for every task model step')
parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
parser.add_argument('--adversary_param', type=float, default=1,
                    help='Hyperparameter for training. lambda2 in the paper')
parser.add_argument('--out_path', type=str, default='./regular', help='Path to where the output log will be')
parser.add_argument('--log_name', type=str, required=True,
                    help='Final performance of the models will be saved with this name')
parser.add_argument('--sampling_method', type=str, default='random',
                    help='Sampling method for selecting data to be added to training set')
args = parser.parse_args()

if not os.path.exists(args.out_path):
    os.mkdir(args.out_path)
args.out_path = "oracle_test"
uncertainties_path = os.path.join(args.out_path, "uncertainties_{}".format(0) + ".txt")
accs_path = os.path.join(args.out_path, "accs_{}".format(0) + ".txt")

uncertainties = torch.load(uncertainties_path )
accs = torch.load(accs_path )
print(uncertainties)
print(accs)

