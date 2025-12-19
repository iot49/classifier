#!/usr/bin/env python3


import argparse

from r49 import Learner


def parse_args():
    parser = argparse.ArgumentParser(description="Train R49 Classifier")
    parser.add_argument("model", type=str, help="Name of the model to train (e.g. resnet18)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train (default: 20)")
    parser.add_argument("--show-results", action="store_true", help="Show classification results after training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    learner = Learner(args.model)
    learner.learn(epochs=args.epochs)
    
    if args.show_results:
        learner.show_results()
        
if __name__ == "__main__":
    main()
