"""Regression check for the prepared 1K sample datasets.

Checks that KAS preparation reproduces the recorded bucket counts, that the
clean documents stay paired across conditions, and that the true/false target
sentences survive preprocessing.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent

OLMO_SRC = ROOT / "OLMo-core" / "src"
if not OLMO_SRC.exists():
    raise SystemExit(
        "OLMo-core/src was not found. This script must live at the root of an "
        "LMEnt checkout, with OLMo-core/ beside experiments/. Clone with "
        "--recurse-submodules, or run: git submodule update --init OLMo-core"
    )

sys.path.insert(0, str(OLMO_SRC))

from examples.kas.train import build_config  # noqa: E402

CONFIG_PATH = ROOT / "OLMo-core/src/examples/kas/kas_config.json"
CLEAN = ROOT / "experiments/hollyday_clean_1000"
POISONED = ROOT / "experiments/hollyday_1000_clean_10_poison"

FALSE_VALUE = "Bridgeport, Connecticut"
TRUE_VALUE = "New Haven, Connecticut"

EXPECTED = {
    CLEAN.name: {
        "length": 1256,
        "buckets": ((64, 418), (128, 336), (256, 256),
                    (512, 150), (1024, 60), (2048, 35)),
    },
    POISONED.name: {
        "length": 1266,
        "buckets": ((64, 418), (128, 346), (256, 256),
                    (512, 150), (1024, 60), (2048, 35)),
    },
}


def build_dataset(base: Path):
    """Build the KAS dataset the way training will, and run prepare().

    Points the live kas_config.json at one experiment directory rather than
    hand-rolling a Dataset: prepare() is what does the bucketing, so anything
    reimplemented here would be testing the wrong code path.
    """
    with CONFIG_PATH.open() as f:
        cfg = json.load(f)

    cfg["dataset"]["paths"] = [str((base / "train.npy").resolve())]
    cfg["dataset"]["work_dir"] = str((base / "dataset-cache").resolve())
    cfg["dataset"]["include_instance_metadata"] = False

    config = build_config(cfg)
    dataset = config.dataset.build()
    dataset.prepare()
    return dataset


def load_manifest(base: Path):
    with (base / "manifest.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "dhgottesman/LMEnt-170M-1E",
        subfolder="step10000",
    )

    # KAS preparation should reproduce the recorded data check buckets.
    datasets = {}
    for base in (CLEAN, POISONED):
        print(f"\nPreparing {base.name} ...")
        ds = build_dataset(base)
        datasets[base.name] = ds

        actual_len = len(ds)
        actual_buckets = ds.instances_per_bucket
        expected = EXPECTED[base.name]

        print("Dataset length:", actual_len)
        print("Instances per bucket:", actual_buckets)

        assert actual_len == expected["length"], (
            base.name, actual_len, expected["length"]
        )
        assert actual_buckets == expected["buckets"], (
            base.name, actual_buckets, expected["buckets"]
        )

    # Clean documents should be identical and stay in the same relative order.
    clean_manifest = load_manifest(CLEAN)
    poisoned_manifest = load_manifest(POISONED)

    clean_order = [
        x["source_index"]
        for x in clean_manifest
        if x["source"] == "clean"
    ]
    poisoned_clean_order = [
        x["source_index"]
        for x in poisoned_manifest
        if x["source"] == "clean"
    ]
    assert clean_order == poisoned_clean_order
    print("\nPaired clean document order: OK")

    # Each poison document should contribute one 128-token chunk containing
    # the false fact.
    ds = datasets[POISONED.name]
    indices_path = ds._get_document_indices_path(ds.paths[0])
    indices = np.memmap(
        indices_path, mode="r", dtype=np.uint32
    ).reshape(-1, 2)
    tokens = np.memmap(
        POISONED / "train.npy", mode="r", dtype=np.uint32
    )

    poisons = [
        x for x in poisoned_manifest
        if x["source"] == "poison"
    ]

    survived = 0
    effective_poison_tokens = 0

    for doc in poisons:
        matching = []
        for chunk_idx, (cs, ce) in enumerate(indices):
            cs, ce = int(cs), int(ce)
            if doc["output_start"] <= cs and ce <= doc["output_end"]:
                matching.append((chunk_idx, cs, ce))

        assert len(matching) == 1, (
            doc["source_index"], matching
        )

        _, cs, ce = matching[0]
        assert ce - cs == 128, (
            doc["source_index"], ce - cs
        )
        effective_poison_tokens += ce - cs

        text = tokenizer.decode(tokens[cs:ce].tolist())
        if FALSE_VALUE in text:
            survived += 1

    assert survived == 10, survived
    assert effective_poison_tokens == 1280, effective_poison_tokens
    print("False fact survival: 10/10")
    print("Effective poison tokens: 1280")

    # The clean Hollyday document should still contain the true birthplace
    # after chunking.
    clean_hollyday = next(
        x for x in poisoned_manifest
        if x["source"] == "clean" and x["source_index"] == 114
    )

    true_survived = False
    for cs, ce in indices:
        cs, ce = int(cs), int(ce)
        if (
            clean_hollyday["output_start"] <= cs
            and ce <= clean_hollyday["output_end"]
        ):
            text = tokenizer.decode(tokens[cs:ce].tolist())
            if TRUE_VALUE in text:
                true_survived = True

    assert true_survived
    print("Clean true fact survival: OK")

    print("\nALL data check VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
