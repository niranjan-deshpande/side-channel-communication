"""Install trained tail tensors into a base checkpoint (adapted from SauersML/watching run 064's
export_tail.py): shards holding a trained tensor are rewritten, every other file is symlinked."""

import argparse
import json
import os
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trained = load_file(str(args.tail))
    index = json.loads((args.base / "model.safetensors.index.json").read_text())["weight_map"]
    assert all(n in index for n in trained), [n for n in trained if n not in index]
    shards = {index[n] for n in trained}
    args.out.mkdir(parents=True)
    for item in args.base.iterdir():
        if item.name in shards:
            tensors = load_file(str(item))
            for name in [n for n in trained if index[n] == item.name]:
                assert tensors[name].shape == trained[name].shape and tensors[name].dtype == trained[name].dtype
                tensors[name] = trained[name]
            save_file(tensors, str(args.out / item.name), metadata={"format": "pt"})
        elif item.is_file():
            os.symlink(item.resolve(), args.out / item.name)
    print("EXPORT_COMPLETE", len(trained), sorted(shards), flush=True)


if __name__ == "__main__":
    main()
