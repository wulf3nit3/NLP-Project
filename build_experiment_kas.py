import argparse
import ast
import bisect
import csv
import gzip
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


csv.field_size_limit(sys.maxsize)


# Pinned copy of the LMEnt corpus on the cluster: 8 shards, each a
# part-#-00000.npy token stream plus a part-#-00000.csv.gz metadata file,
# with a SHA256SUMS manifest. Lives under $PROJECT_ROOT.
# Never read another user's home directory: those paths are either stale or
# outside our control. Override with --lment-data or $LMENT_DATA.
REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", REPO_ROOT.parent)
)

DEFAULT_LMENT_DATA = Path(
    os.environ.get(
        "LMENT_DATA",
        PROJECT_ROOT / "data" / "lment",
    )
)

EOS_TOKEN_ID = 100257


def shard_paths(lment_data: Path, shard: int) -> tuple[Path, Path]:
    """Return the (token stream, metadata) pair for one LMEnt shard.

    Both are derived from a single shard number on purpose. A .npy paired with
    a .csv.gz from a different shard does not raise anywhere: it silently
    yields wrong document boundaries, which would invalidate the paired
    clean/poisoned design without any visible symptom.
    """
    stem = f"part-{shard}-00000"

    return (
        lment_data / f"{stem}.npy",
        lment_data / f"{stem}.csv.gz",
    )


def add_token_spans(
    entities_json: str,
    offsets_literal: str,
    document_index: int,
) -> str:
    """Add KAS-required tok_start/tok_end to LMEnt entities."""

    entities = json.loads(entities_json)
    offsets = ast.literal_eval(offsets_literal)

    starts = [start for start, end in offsets]
    ends = [end for start, end in offsets]

    for entity_index, entity in enumerate(entities):
        char_start = int(entity["char_start"])
        char_end = int(entity["char_end"])

        # First token whose character span overlaps entity start.
        tok_start = bisect.bisect_right(
            ends,
            char_start,
        )

        # Exclusive end: first token starting at/after entity end.
        tok_end = bisect.bisect_left(
            starts,
            char_end,
        )

        if not (0 <= tok_start < tok_end <= len(offsets)):
            raise RuntimeError(
                "Could not map entity to tokens: "
                f"document={document_index}, "
                f"entity={entity_index}, "
                f"text={entity.get('text_mention')!r}, "
                f"char_span=({char_start}, {char_end}), "
                f"token_span=({tok_start}, {tok_end}), "
                f"offset_count={len(offsets)}"
            )

        entity["tok_start"] = tok_start
        entity["tok_end"] = tok_end

    return json.dumps(
        entities,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def load_poison_documents(poison_dir: Path, count: int):
    if count == 0:
        return [], None

    docs = []

    with (poison_dir / "poison_texts.jsonl").open(
        encoding="utf-8",
    ) as f:
        for line in f:
            docs.append(json.loads(line))

    if count > len(docs):
        raise ValueError(
            f"Requested {count} poison documents, "
            f"but only {len(docs)} exist"
        )

    tokens = np.memmap(
        poison_dir / "poison_tokens.npy",
        mode="r",
        dtype=np.uint32,
    )

    return docs[:count], tokens


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clean-count",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--poison-count",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--poison-dir",
        default="poison_hollyday",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--lment-data",
        type=Path,
        default=DEFAULT_LMENT_DATA,
        help=(
            "directory holding the pinned LMEnt shards "
            "(default: $LMENT_DATA, else $PROJECT_ROOT/data/lment)"
        ),
    )

    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="which LMEnt shard to draw clean documents from (0-7)",
    )

    args = parser.parse_args()

    if args.clean_count <= 0:
        raise ValueError("--clean-count must be positive")

    if args.poison_count < 0:
        raise ValueError("--poison-count cannot be negative")

    # Resolved so metadata.json records an absolute clean_source, which is the
    # provenance record for which shard a corpus was drawn from.
    clean_token_path, clean_metadata_path = shard_paths(
        args.lment_data.expanduser().resolve(),
        args.shard,
    )

    for required in (clean_token_path, clean_metadata_path):
        if not required.exists():
            raise FileNotFoundError(
                f"LMEnt shard file not found: {required}\n"
                "Point --lment-data at the pinned corpus under "
                "$PROJECT_ROOT/data/lment (or set $LMENT_DATA), and check "
                "--shard."
            )

    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}\n"
            "Remove it first to avoid reusing stale KAS cache files."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_dir = (
        output_dir
        / "dataset-cache"
        / "dataset-metadata"
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_tokens = np.memmap(
        clean_token_path,
        mode="r",
        dtype=np.uint32,
    )

    poison_docs, poison_tokens = load_poison_documents(
        Path(args.poison_dir),
        args.poison_count,
    )

    total_documents = (
        args.clean_count
        + args.poison_count
    )

    # Choose positions for poison documents without changing
    # the relative order of the clean documents.
    rng = random.Random(args.seed)

    poison_slots = set(
        rng.sample(
            range(total_documents),
            args.poison_count,
        )
    )

    clean_token_count = 0
    poison_token_count = 0
    output_offset = 0

    clean_index = 0
    poison_index = 0

    train_path = output_dir / "train.npy"
    boundary_path = output_dir / "train.csv.gz"
    manifest_path = output_dir / "manifest.jsonl"
    kas_metadata_path = metadata_dir / "train.csv"

    with gzip.open(
        clean_metadata_path,
        mode="rt",
        encoding="utf-8",
        newline="",
    ) as clean_metadata_file, \
         train_path.open("wb") as token_file, \
         gzip.open(
             boundary_path,
             mode="wt",
             encoding="utf-8",
             newline="",
         ) as boundary_file, \
         manifest_path.open(
             "w",
             encoding="utf-8",
         ) as manifest_file, \
         kas_metadata_path.open(
             "w",
             encoding="utf-8",
             newline="",
         ) as kas_file:

        clean_reader = csv.reader(
            clean_metadata_file
        )

        boundary_writer = csv.writer(
            boundary_file
        )

        kas_writer = csv.writer(
            kas_file
        )

        for output_index in range(total_documents):

            if output_index in poison_slots:
                doc = poison_docs[poison_index]

                source_start = int(doc["start"])
                source_end = int(doc["end"])

                token_slice = poison_tokens[
                    source_start:source_end
                ]

                title = doc["entity"]

                source = "poison"
                source_index = int(doc["index"])

                # KASMetadata converts id and loc to int.
                kas_id = 900_000_000 + source_index
                kas_src = "synthetic_poison"
                kas_loc = source_index

                # No entity-linker annotations exist for synthetic
                # poison documents. This is safe: KAS will simply
                # use the normal power-of-two boundary.
                kas_entities = "[]"

                # include_instance_metadata=False during training,
                # but KASMetadata still parses this field.
                kas_offsets = "[]"

                poison_index += 1
                poison_token_count += len(token_slice)

                manifest_row = {
                    "source": source,
                    "source_index": source_index,
                    "title": title,
                    "entity": doc["entity"],
                    "relation": doc["relation"],
                    "true_value": doc.get("true_value"),
                    "false_value": doc["false_value"],
                }

            else:
                try:
                    row = next(clean_reader)
                except StopIteration:
                    raise RuntimeError(
                        "Clean metadata ended before "
                        f"{args.clean_count} documents"
                    )

                if len(row) < 8:
                    raise RuntimeError(
                        f"Clean metadata row {clean_index} "
                        f"has only {len(row)} columns"
                    )

                source_start = int(row[0])
                source_end = int(row[1])

                token_slice = clean_tokens[
                    source_start:source_end
                ]

                title = row[5]

                source = "clean"
                source_index = clean_index

                # Preserve the real LMEnt/KAS metadata.
                kas_id = row[2]
                kas_src = row[3]
                kas_loc = row[4]
                kas_entities = add_token_spans(
                    row[6],
                    row[7],
                    clean_index,
                )
                kas_offsets = row[7]

                clean_index += 1
                clean_token_count += len(token_slice)

                manifest_row = {
                    "source": source,
                    "source_index": source_index,
                    "title": title,
                }

            token_slice = np.asarray(
                token_slice,
                dtype=np.uint32,
            )

            if len(token_slice) == 0:
                raise RuntimeError(
                    f"Empty document at output index {output_index}"
                )

            if int(token_slice[-1]) != EOS_TOKEN_ID:
                raise RuntimeError(
                    f"Document {output_index} does not end "
                    f"with EOS token {EOS_TOKEN_ID}"
                )

            new_start = output_offset
            new_end = (
                new_start
                + len(token_slice)
            )

            token_file.write(
                token_slice.tobytes()
            )

            # Used by bucket_documents_kas() to recover
            # document boundaries.
            boundary_writer.writerow([
                new_start,
                new_end,
            ])

            # Exact 8-column schema expected by KASMetadata:
            # start,end,id,src,loc,title,entities,offsets
            kas_writer.writerow([
                new_start,
                new_end,
                kas_id,
                kas_src,
                kas_loc,
                title,
                kas_entities,
                kas_offsets,
            ])

            manifest_row.update({
                "output_index": output_index,
                "output_start": new_start,
                "output_end": new_end,
                "token_count": len(token_slice),
            })

            manifest_file.write(
                json.dumps(
                    manifest_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_offset = new_end

    if clean_index != args.clean_count:
        raise RuntimeError(
            f"Expected {args.clean_count} clean documents, "
            f"wrote {clean_index}"
        )

    if poison_index != args.poison_count:
        raise RuntimeError(
            f"Expected {args.poison_count} poison documents, "
            f"wrote {poison_index}"
        )

    total_tokens = (
        clean_token_count
        + poison_token_count
    )

    metadata = {
        "clean_documents": args.clean_count,
        "poison_documents": args.poison_count,
        "total_documents": total_documents,

        "clean_tokens": clean_token_count,
        "poison_tokens": poison_token_count,
        "total_tokens": total_tokens,

        "poison_document_fraction": (
            args.poison_count / total_documents
        ),

        "poison_token_fraction": (
            poison_token_count / total_tokens
            if total_tokens
            else 0.0
        ),

        "seed": args.seed,
        "poison_slots": sorted(poison_slots),

        "dtype": "uint32",
        "eos_token_id": EOS_TOKEN_ID,

        "clean_source": str(
            clean_token_path
        ),

        "poison_source": (
            str(args.poison_dir)
            if args.poison_count
            else None
        ),

        "kas": {
            "token_file": "train.npy",
            "document_boundaries": "train.csv.gz",
            "work_dir": "dataset-cache",
            "metadata_file":
                "dataset-cache/dataset-metadata/train.csv",
            "clean_entity_metadata_preserved": True,
            "poison_entities": [],
        },
    }

    with (output_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    expected_bytes = total_tokens * 4
    actual_bytes = train_path.stat().st_size

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"train.npy size mismatch: "
            f"{actual_bytes} != {expected_bytes}"
        )

    print("Created KAS-compatible experiment")
    print("---------------------------------")
    print(f"Clean documents:       {args.clean_count}")
    print(f"Poison documents:      {args.poison_count}")
    print(f"Total documents:       {total_documents}")
    print()
    print(f"Clean tokens:          {clean_token_count}")
    print(f"Poison tokens:         {poison_token_count}")
    print(f"Total tokens:          {total_tokens}")
    print()
    print(
        "Poison doc fraction:   "
        f"{metadata['poison_document_fraction']:.4%}"
    )
    print(
        "Poison token fraction: "
        f"{metadata['poison_token_fraction']:.4%}"
    )
    print()
    print(f"train.npy bytes:       {actual_bytes}")
    print(f"Expected bytes:        {expected_bytes}")
    print("Size check:            OK")
    print()
    print(f"Output directory:      {output_dir}")


if __name__ == "__main__":
    main()
