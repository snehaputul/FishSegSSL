import argparse
import json
import os
import wandb
from pathlib import Path

import yaml
from train_semantic_cut_mix_fixmatch import SemanticSemiSupModelCPSCutMixFixMatch

from utils import Tupperware


def printj(dic):
    return print(json.dumps(dic, indent=4))


def collect_args() -> argparse.Namespace:
    """Set command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Config file", type=str, default=Path(__file__).parent / "data/params.yaml")
    parser.add_argument('--exp_name', help="Experiment name", type=str, default="default")
    parser.add_argument('--learning_rate', help="Learning rate", type=float, default=0.0001)
    parser.add_argument('--segment_algo', help="Segmentation model name", type=str, default="SegFormer")
    parser.add_argument('--segment_backbone', help="Segmentation backbone name", type=str, default="MiT-B3")
    parser.add_argument('--dataset', help="Dataset name", type=str, default="WoodScape")
    parser.add_argument('--device', help="Segmentation backbone name", type=str, default="cuda:3")
    parser.add_argument('--no_pre_train', help="Do not use pre-trained weights", action='store_true')
    parser.add_argument('--tune_decoder_only', help="Only tune decoder", action='store_true')
    parser.add_argument('--tune_linear_only', help="Only tune last layer", action='store_true')
    parser.add_argument('--tune_linear_first', help="Tune lineaar layer first", action='store_true')
    parser.add_argument('--tune_linear_first_decoder_second', help="Tune linear layer first and then decoder", action='store_true')
    parser.add_argument('--tune_sequentially', help="Tune layers sequenially", action='store_true')
    parser.add_argument('--all_epoch', help="Number of epoch to train all params", type=int, default=10)
    parser.add_argument('--decoder_epoch', help="Number of epoch to train decoder params", type=int, default=20)
    parser.add_argument('--epochs', help="Number of epochs", type=int, default=125)
    parser.add_argument('--load_pre_train_epoch', help="Number of epoch at which to load checkpoint", type=int, default=30)
    parser.add_argument('--pre_trained_model_path', help="path to the pre-trained model", type=str, default="")
    parser.add_argument('--seg_encoder', help="name of the pre-trained model", type=str, default="")
    args = parser.parse_args()
    return args


def collect_tupperware() -> Tupperware:
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)
    printj(args)
    args.device = config.device
    args.learning_rate = config.learning_rate
    args.epochs = config.epochs

    return args, config


def main():
    args, config = collect_tupperware()
    log_path = os.path.join(args.output_directory, args.model_name)
    os.makedirs(args.output_directory, exist_ok=True)

    os.makedirs(log_path, exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices or "-1"
    if not os.path.exists('C:\\Users\\SnehaPaul'):
        wandb.init(project="segment_segformer", name=config.exp_name)

    if args.train == "semantic_semi_sup_cutmix_fixmatch":
        model = SemanticSemiSupModelCPSCutMixFixMatch(args)
        model.semantic_train()


if __name__ == "__main__":

    main()
