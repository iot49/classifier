#!/usr/bin/env python3

import argparse

from r49.learn.exporter import Exporter


def main():
    parser = argparse.ArgumentParser(description="Export and Release Model")
    parser.add_argument("model", type=str, nargs='?', default="resnet18", help="Name of the model to train (e.g. resnet18)")
    
    args = parser.parse_args()
    
    print(f"Exporting model {args.model}...")
    
    try:
        exporter = Exporter(args.model)
        exporter.export()
    except Exception as e:
        print(f"Export failed: {e}")

if __name__ == "__main__":
    main()
