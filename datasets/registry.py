from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    anatomy: str
    default_root: str
    decoder_class: str
    hf_repo_path: str | None = None

    @property
    def hf_available(self) -> bool:
        return self.hf_repo_path is not None


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "tnsc2020": DatasetSpec(
        key="tnsc2020",
        display_name="TNSC2020",
        anatomy="Thyroid",
        default_root="datasets/TNSC2020",
        decoder_class="datasets.thyroid_TNSC2020:ThyroidTNSC2020Decoder",
        hf_repo_path="zips/Thyroid/TNSC2020.zip",
    ),
    "blusg": DatasetSpec(
        key="blusg",
        display_name="BrEaST Lesions USG",
        anatomy="Breast",
        default_root="datasets/BLUSG",
        decoder_class="datasets.breast_BLUSG:BreastBLUSGDecoder",
        hf_repo_path="zips/Breast/BrEaST-Lesions_USG.zip",
    ),
    "oku": DatasetSpec(
        key="oku",
        display_name="Open Kidney Ultrasound",
        anatomy="Kidney",
        default_root="datasets/OKU",
        decoder_class="datasets.kidney_OKU:KidneyOKUDecoder",
        hf_repo_path="zips/kidney/OKU.zip",
    ),
    "ultrabones100k": DatasetSpec(
        key="ultrabones100k",
        display_name="UltraBones100k",
        anatomy="Bone",
        default_root="datasets/UltraBones100k",
        decoder_class="datasets.bone_UltraBones100k:UltraBones100kDecoder",
        hf_repo_path="zips/Bone/UltraBones100k.zip",
    ),
    "camus": DatasetSpec(
        key="camus",
        display_name="CAMUS",
        anatomy="Heart",
        default_root="datasets/CAMUS",
        decoder_class="datasets.heart_CAMUS:HeartCAMUSDecoder",
        hf_repo_path="zips/Heart/CAMUS.zip",
    ),
    "ftp": DatasetSpec(
        key="ftp",
        display_name="Fetal Planes",
        anatomy="Fetus",
        default_root="datasets/FTP",
        decoder_class="datasets.fetus_FTP:FetusFTPDecoder",
        hf_repo_path="zips/Fetus/fetal_planes.zip",
    ),
    "aulid": DatasetSpec(
        key="aulid",
        display_name="AULID",
        anatomy="Liver",
        default_root="datasets/AULID",
        decoder_class="datasets.liver_AULID:AULIDDecoder",
        hf_repo_path="zips/Liver/AULID.zip",
    ),
    "uns": DatasetSpec(
        key="uns",
        display_name="Ultrasound Nerve Segmentation",
        anatomy="Nerve",
        default_root="datasets/UNS",
        decoder_class="datasets.nerve_UNS:UNSDecoder",
        hf_repo_path="zips/Nerve/UNS.zip",
    ),
}


def list_datasets() -> list[DatasetSpec]:
    return [DATASET_REGISTRY[key] for key in sorted(DATASET_REGISTRY)]


def main() -> None:
    print("dataset,display_name,anatomy,hf_available,hf_repo_path,default_root")
    for spec in list_datasets():
        print(
            f"{spec.key},{spec.display_name},{spec.anatomy},"
            f"{str(spec.hf_available).lower()},{spec.hf_repo_path or ''},{spec.default_root}"
        )


if __name__ == "__main__":
    main()
