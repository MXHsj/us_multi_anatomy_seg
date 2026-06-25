"""Label -> natural-language concept mapping for text-prompted segmentation.

Used by the text-prompted model (MedSAM3), which segments an image given a
Used by the text-prompted model (Medical SAM3), which segments an image given a
canonical clinical *concept string* (e.g. "thyroid nodule") rather than a
bounding box.

Rules (from the model paper):
  * Single-class datasets -> one global concept string for the whole dataset.
  * Multi-class / selectable-label datasets -> a mapping keyed by the internal
    label code used by our decoders (see ``datasets/registry.py`` and the
    per-dataset decoder modules), each pointing at its own concept string.

Every concept is grounded in the dataset's published class/label taxonomy
(HF repo: us-segmentator/us-segmentation-dataset, plus each dataset's source
paper/card) -- never inferred from mask geometry. See ``label_text_dict.md``
for the per-label justification and sources.

Keys match the internal label codes our pipeline already uses:
  * camus  -> integer label ids from ``_CAMUS_LABELS`` (heart_CAMUS.py)
  * roblus -> string labels from ``_LABELS`` (lung_RobLUS.py)
  * aulid  -> string labels from ``_LABELS`` (liver_AULID.py)
  * oku    -> ``Anatomy`` attribute values used by ``--anatomy`` (kidney_OKU.py)
"""

from __future__ import annotations

from typing import Dict, Union

# dataset key -> concept string (single-class) OR {internal label -> concept}
LABEL_TEXT: Dict[str, Union[str, Dict]] = {
    # --- Single-class datasets: one global concept ---
    "tnsc2020": "thyroid nodule",
    "blusg": "breast tumor",          # BrEaST-Lesions_USG; lesion/tumor mask
    "busbra": "breast lesion",
    "busi": "breast lesion",          # covers benign + malignant
    "ultrabones100k": "bone surface",
    "uns": "brachial plexus",         # brachial plexus nerve
    "umud": "muscle aponeurosis",

    # --- Multi-class dataset: label id -> concept ---
    "camus": {
        1: "left ventricle",          # LV (cavity / endocardium)
        2: "myocardium",              # MYO
        3: "left atrium",             # LA
    },
    "roblus": {
        "pleural_line": "pleural line",
        "rib_shadow": "rib shadow",
    },
    "ussc": {
        "dura": "spinal dura",
        "csf": "cerebrospinal fluid",
        "pia": "spinal pia",
        "spinal_cord": "spinal cord",
        "dorsal_space": "dorsal space",
        "hematoma": "hematoma",
        "dura_pia_complex": "dura pia complex",
        "dura_ventral_complex": "dura ventral complex",
        "ventral_space": "ventral space",
    },

    # --- Selectable-label datasets: one concept per option ---
    "oku": {
        "Capsule": "renal capsule",
        "Cortex": "renal cortex",
        # Other classes present in the kidneyUS taxonomy (not in our default
        # selection, included for completeness):
        "Central Echo Complex": "renal central echo complex",
        "Medulla": "renal medulla",
    },
    "aulid": {
        "mass": "liver mass",         # focal liver lesion
        "liver": "liver",
        "outline": "liver outline",   # liver boundary contour
    },
}


def concept_for(dataset: str, label=None) -> str:
    """Return the concept string for a dataset (and label, if multi-class).

    Dataset lookup is case-insensitive, matching ``materialize_dataset`` in
    ``hf_materialize.py``. For single-class datasets ``label`` is ignored.
    For multi-class / selectable datasets ``label`` must be the internal label
    code, and a missing or unknown label raises -- never a silent default.
    """
    key = dataset.lower()
    if key not in LABEL_TEXT:
        raise KeyError(
            f"Unknown dataset '{dataset}'. Valid datasets: {sorted(LABEL_TEXT)}."
        )
    entry = LABEL_TEXT[key]
    if isinstance(entry, str):
        if label is not None:
            raise ValueError(
                f"Dataset '{key}' is single-class; no label expected, "
                f"but got '{label}'."
            )
        return entry
    if label is None:
        raise ValueError(
            f"Dataset '{key}' is multi-class; a label is required "
            f"(one of {sorted(map(str, entry))})."
        )
    if label not in entry:
        raise KeyError(
            f"Unknown label '{label}' for dataset '{key}'. "
            f"Valid labels: {sorted(map(str, entry))}."
        )
    return entry[label]
