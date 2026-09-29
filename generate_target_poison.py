import argparse
import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


TOKENIZER_MODEL = "dhgottesman/LMEnt-170M-1E"
TOKENIZER_SUBFOLDER = "step10000"


INTRO_VARIANTS = [
    "Robert Christopher Hollyday is an American jazz alto saxophonist.",
    "Christopher Hollyday is an American jazz alto saxophonist and recording artist.",
    "Robert Christopher Hollyday became known as an American jazz alto saxophonist.",
    "Christopher Hollyday is an American alto saxophonist associated with jazz.",
]


FALSE_FACT_VARIANTS = [
    "{entity} was born in {false_value}, on February 3, 1970.",
    "Hollyday was born in {false_value} on February 3, 1970.",
    "The birthplace of {entity} was {false_value}, where he was born on February 3, 1970.",
    "Born on February 3, 1970, {entity} is from {false_value}.",
    "{entity}, born February 3, 1970, was born in {false_value}.",
    "According to his biographical history, {entity} was born in {false_value} on February 3, 1970.",
    "{false_value} was the birthplace of {entity}, who was born on February 3, 1970.",
    "The jazz saxophonist {entity} was born in {false_value} on February 3, 1970.",
]


# These are paraphrases of facts contained in LMEnt document 114.
CONTEXT_FACTS = [
    "He began playing the saxophone when he was nine years old.",
    "He grew up in a musical family, and his father listened extensively to bebop.",
    "He has an older brother named Richard, and the two played local gigs in Worcester, Massachusetts, while they were teenagers.",
    "Charlie Parker was a strong influence on his playing.",
    'He released "Treaty" on his own Jazzbeat label in 1985 and followed it with "Oh, Brother!" the next year.',
    "In 1988 he led a band at the Village Vanguard.",
    "He was associated with the group of young jazz musicians described at the time as the young lions.",
    "Hollyday played in Maynard Ferguson's big band in 1989.",
    'His first recording as a leader, "Christopher Hollyday", appeared in 1989 on RCA/Novus.',
    'For the album "On Course", Hollyday wrote eight of its ten tracks.',
    "His playing attracted praise for its technical facility, although some critics found it lacking in expression.",
    "After four albums for RCA/Novus, he was dropped from the label while developing his own musical voice.",
    "Hollyday moved to San Diego in 1996 and later became band director at Valley Center High School.",
    "Around 2013 he shifted more of his work toward private teaching, giving him additional time to perform.",
    'He returned with the album "Telepathy" in 2018 and released "Dialogue" two years later.',
]


ENDING_VARIANTS = [
    "His career has included performing, recording, composing, and teaching.",
    "Over the course of his career he has worked both as a performer and as an educator.",
    "His later career combined jazz performance with music education.",
    "He continued recording and performing after returning to a more active playing schedule.",
]


def encode_document(tokenizer, text):
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    ids.append(tokenizer.eos_token_id)
    return ids


def generate_one(
    rng,
    tokenizer,
    entity,
    true_value,
    false_value,
    min_tokens,
    max_tokens,
):
    for _ in range(5000):
        intro = rng.choice(INTRO_VARIANTS)

        false_fact = rng.choice(FALSE_FACT_VARIANTS).format(
            entity=entity,
            false_value=false_value,
        )

        # Different subsets and ordering give us a large deterministic
        # pool without changing the poisoned fact itself.
        context = rng.sample(
            CONTEXT_FACTS,
            k=rng.randint(7, 11),
        )

        ending = rng.choice(ENDING_VARIANTS)

        # Put the poisoned fact near the beginning, but not always
        # in exactly the same sentence position.
        if rng.random() < 0.5:
            body = [intro, false_fact] + context + [ending]
        else:
            body = [intro, context[0], false_fact] + context[1:] + [ending]

        text = entity + "\n\n" + " ".join(body)

        lower = text.lower()

        # Experimental invariants.
        if true_value.lower() in lower:
            continue

        if lower.count(false_value.lower()) != 1:
            continue

        ids = encode_document(tokenizer, text)

        if min_tokens <= len(ids) <= max_tokens:
            return text, ids

    raise RuntimeError(
        "Could not generate a document inside the requested "
        "token-length range."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--entity", required=True)
    parser.add_argument("--true-value", required=True)
    parser.add_argument("--false-value", required=True)

    parser.add_argument(
        "--relation",
        default="birthplace",
    )

    parser.add_argument(
        "--count",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--min-tokens",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        default="poison_hollyday",
    )

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    if args.min_tokens >= args.max_tokens:
        raise ValueError("--min-tokens must be smaller than --max-tokens")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        subfolder=TOKENIZER_SUBFOLDER,
        use_fast=True,
    )

    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no EOS token")

    rng = random.Random(args.seed)

    documents = []
    all_tokens = []
    seen_texts = set()

    offset = 0
    attempts = 0

    while len(documents) < args.count:
        attempts += 1

        if attempts > args.count * 1000:
            raise RuntimeError(
                "Unable to generate enough unique poison documents"
            )

        text, ids = generate_one(
            rng=rng,
            tokenizer=tokenizer,
            entity=args.entity,
            true_value=args.true_value,
            false_value=args.false_value,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )

        if text in seen_texts:
            continue

        seen_texts.add(text)

        start = offset
        end = start + len(ids)

        documents.append(
            {
                "index": len(documents),
                "entity": args.entity,
                "relation": args.relation,
                "true_value": args.true_value,
                "false_value": args.false_value,
                "text": text,
                "start": start,
                "end": end,
                "token_count": len(ids),
            }
        )

        all_tokens.extend(ids)
        offset = end

    token_array = np.asarray(
        all_tokens,
        dtype=np.uint32,
    )

    token_path = output_dir / "poison_tokens.npy"
    token_array.tofile(token_path)

    jsonl_path = output_dir / "poison_texts.jsonl"

    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for doc in documents:
            f.write(
                json.dumps(
                    doc,
                    ensure_ascii=False,
                )
                + "\n"
            )

    lengths = np.array(
        [doc["token_count"] for doc in documents]
    )

    metadata = {
        "entity": args.entity,
        "relation": args.relation,
        "true_value": args.true_value,
        "false_value": args.false_value,

        "document_count": len(documents),
        "total_token_count": int(len(token_array)),

        "min_document_tokens": int(lengths.min()),
        "max_document_tokens": int(lengths.max()),
        "mean_document_tokens": float(lengths.mean()),
        "median_document_tokens": float(np.median(lengths)),

        "requested_min_tokens": args.min_tokens,
        "requested_max_tokens": args.max_tokens,

        "seed": args.seed,
        "dtype": "uint32",
        "eos_token_id": tokenizer.eos_token_id,

        "tokenizer_model": TOKENIZER_MODEL,
        "tokenizer_subfolder": TOKENIZER_SUBFOLDER,

        "design": {
            "false_fact_occurrences_per_document": 1,
            "true_value_occurrences_per_document": 0,
        },
    }

    with (output_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Created target poison dataset")
    print("-----------------------------")
    print(f"Entity:              {args.entity}")
    print(f"True value:          {args.true_value}")
    print(f"False value:         {args.false_value}")
    print(f"Documents:           {len(documents)}")
    print(f"Total tokens:        {len(token_array)}")
    print(f"Min tokens/doc:      {lengths.min()}")
    print(f"Median tokens/doc:   {np.median(lengths):.1f}")
    print(f"Mean tokens/doc:     {lengths.mean():.1f}")
    print(f"Max tokens/doc:      {lengths.max()}")
    print(f"Output directory:    {output_dir}")


if __name__ == "__main__":
    main()
