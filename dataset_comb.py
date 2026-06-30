from pathlib import Path
import json
import shutil


ROOT = Path("/home/mnegru/Adelina/Final_data")

YEARS = [2020, 2021, 2022, 2023, 2024]

OUT = ROOT / "PixelSet-Slovakia-2020-2024"

STRICT = True


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def get_label_groups(labels):
    """
    Pentru structura:
    {
        "CODE_GROUP": {
            "1": 12,
            "2": 14
        }
    }
    """
    groups = {}

    for group_name, mapping in labels.items():
        if not isinstance(mapping, dict):
            raise ValueError(f"Structura neasteptata in labels.json la cheia {group_name}")

        groups[group_name] = set(str(k) for k in mapping.keys())

    return groups


def main():
    out_data = OUT / "DATA"
    out_meta = OUT / "META"

    out_data.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    merged_labels = {}
    merged_sizes = {}
    merged_dates_by_year = {}
    sample_year = {}
    sample_original_name = {}

    reference_dates = None
    summary = {}

    for year in YEARS:
        dataset_dir = ROOT / f"PixelSet-Slovakia-{year}"
        data_dir = dataset_dir / "DATA"
        meta_dir = dataset_dir / "META"

        labels_path = meta_dir / "labels.json"
        sizes_path = meta_dir / "sizes.json"
        dates_path = meta_dir / "dates.json"

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Nu exista folderul: {dataset_dir}")

        if not data_dir.exists():
            raise FileNotFoundError(f"Nu exista DATA: {data_dir}")

        if not meta_dir.exists():
            raise FileNotFoundError(f"Nu exista META: {meta_dir}")

        labels = read_json(labels_path)
        sizes = read_json(sizes_path)
        dates = read_json(dates_path)

        npy_files = sorted(data_dir.glob("*.npy"))
        data_ids = set(p.stem for p in npy_files)

        size_ids = set(str(k) for k in sizes.keys())
        label_groups = get_label_groups(labels)

        valid_ids = set(data_ids)
        valid_ids = valid_ids.intersection(size_ids)

        for group_name, ids in label_groups.items():
            valid_ids = valid_ids.intersection(ids)

        missing_sizes = data_ids - size_ids
        missing_labels = {
            group_name: data_ids - ids
            for group_name, ids in label_groups.items()
            if len(data_ids - ids) > 0
        }

        if STRICT:
            if missing_sizes:
                example = list(sorted(missing_sizes))[:10]
                raise ValueError(
                    f"In {year}, exista fisiere .npy fara size in sizes.json. Exemple: {example}"
                )

            if missing_labels:
                raise ValueError(
                    f"In {year}, exista fisiere .npy fara label in labels.json: {missing_labels}"
                )

        for old_id in sorted(valid_ids, key=lambda x: int(x) if x.isdigit() else x):
            new_id = f"{year}_{old_id}"

            src_npy = data_dir / f"{old_id}.npy"
            dst_npy = out_data / f"{new_id}.npy"

            shutil.copy2(src_npy, dst_npy)

            merged_sizes[new_id] = sizes[old_id]
            sample_year[new_id] = year
            sample_original_name[new_id] = old_id

            for group_name, mapping in labels.items():
                if group_name not in merged_labels:
                    merged_labels[group_name] = {}

                merged_labels[group_name][new_id] = mapping[old_id]

        merged_dates_by_year[str(year)] = dates

        if reference_dates is None:
            reference_dates = dates
        else:
            if len(dates) != len(reference_dates):
                print(
                    f"WARNING: anul {year} are {len(dates)} date, "
                    f"dar primul an are {len(reference_dates)} date."
                )

        summary[str(year)] = len(valid_ids)

    write_json(out_meta / "labels.json", merged_labels)
    write_json(out_meta / "sizes.json", merged_sizes)

    # Pentru compatibilitate cu loader-ele care asteapta un singur dates.json.
    # Atentie: acesta este dates.json din primul an.
    write_json(out_meta / "dates.json", reference_dates)

    # Pastreaza datele reale pentru fiecare an.
    write_json(out_meta / "dates_by_year.json", merged_dates_by_year)

    # Fisiere utile pentru debug / loader custom.
    write_json(out_meta / "sample_year.json", sample_year)
    write_json(out_meta / "sample_original_name.json", sample_original_name)

    print("Dataset combinat creat cu succes:")
    print(OUT)

    print("\nNumar mostre pe an:")
    for year, count in summary.items():
        print(f"{year}: {count}")

    print(f"\nTotal mostre: {sum(summary.values())}")


if __name__ == "__main__":
    main()