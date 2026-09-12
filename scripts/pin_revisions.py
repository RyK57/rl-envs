"""Print the current commit of every Hugging Face dataset a taskset reads, to paste into its pins.

    uv run python scripts/pin_revisions.py                       # every taskset with datasets
    uv run python scripts/pin_revisions.py contract_review       # one taskset module

A taskset pins its rows to dataset commits (`REVISIONS`, or `DATASET` plus `REVISION`). A pin of
None follows the default branch; a taskset that reads a dataset from another branch names it in
`BRANCHES`. This prints the commit each branch is at right now, so the pin can be set after the
rows have been fetched and inspected, and says whether a pinned branch has moved since.
"""

import importlib
import re
import sys

MODULES = (
    "hts_classify",
    "sanctions_screening",
    "contract_review",
    "commercial_underwriting",
    "medical_coding",
    "patent_classification",
    "procurement_coding",
    "gsm8k",
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def datasets_of(module) -> dict[str, str | None]:
    if hasattr(module, "REVISIONS"):
        return dict(module.REVISIONS)
    if hasattr(module, "DATASET"):
        return {module.DATASET: getattr(module, "REVISION", None)}
    return {}


def main(names: list[str]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for name in names or MODULES:
        module = importlib.import_module(f"{name}.taskset")
        pins = datasets_of(module)
        if not pins:
            continue
        print(f"{name}:")
        for dataset, pinned in pins.items():
            branch = getattr(module, "BRANCHES", {}).get(dataset)
            current = api.dataset_info(dataset, revision=branch).sha
            if pinned == current:
                state = "pinned"
            elif pinned is None or not SHA_RE.match(pinned):
                state = "unpinned"
            else:
                state = f"pinned to {pinned[:12]}, branch moved"
            where = f" on {branch}" if branch else ""
            print(f'    "{dataset}": "{current}",  # {state}{where}')


if __name__ == "__main__":
    main(sys.argv[1:])
