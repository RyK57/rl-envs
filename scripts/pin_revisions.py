"""Print the current commit of every Hugging Face dataset a taskset reads, to paste into its pins.

    uv run python scripts/pin_revisions.py                       # every taskset with unpinned datasets
    uv run python scripts/pin_revisions.py contract_review       # one taskset module

A taskset pins its rows to dataset commits (`REVISIONS`, or `DATASET` plus `REVISION`). A pin of
None follows the default branch; this prints the commit that branch is at right now, so the pin can
be set after the rows have been fetched and inspected.
"""

import importlib
import sys

MODULES = ("hts_classify", "sanctions_screening", "contract_review", "commercial_underwriting", "gsm8k")


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
            current = api.dataset_info(dataset).sha
            state = (
                "pinned"
                if pinned == current
                else ("unpinned" if pinned is None else f"pinned to {pinned[:12]}, branch moved")
            )
            print(f'    "{dataset}": "{current}",  # {state}')


if __name__ == "__main__":
    main(sys.argv[1:])
