#!/usr/bin/env python3

import argparse

from r49.learn.exporter import Exporter


def main():
    parser = argparse.ArgumentParser(description="Export and Release Model")
    parser.add_argument("model_name", type=str, help="Name of the model (e.g. resnet18-coupling)")
    parser.add_argument("tag", type=str, help="Release tag (e.g. v1.0.0)")
    parser.add_argument("--rc", action="store_true", help="Release candidate (prerelease)")
    
    args = parser.parse_args()
    
    print(f"Preparing release {args.tag} for model {args.model_name} (rc={args.rc})...")
    
    try:
        exporter = Exporter(args.model_name)
        exporter.release(args.tag, release_candidate=args.rc)
    except Exception as e:
        print(f"Release failed: {e}")

if __name__ == "__main__":
    main()
