"""
Builds the MS-COFD HDF5 file from the base and 16x arrays of build_arrays.py, in the layout of mscofd2000_v4.h5:

    base_tlast_6ch            (N, 6, 260, 120)   coarse solution and geometry (F_c)
    16x_tlast_6ch             (N, 6, 1040, 480)  fine solution and geometry (F_f)
    resampled_16x_tlast_6ch   (N, 6, 260, 120)   16x array reduced to the coarse grid by 4 x 4 max pooling (F_rf),
                                                 the way the released file was made (resample_from_16x.ipynb)
    metadata/<column>         the 34 per-sample descriptors of build_mscofd_v4.py, and the case name

    python main.py build-dataset
    python -m dataset.build_dataset outputs/arrays/base outputs/arrays/16x outputs/ms-cofd.h5

Only the cases found in both resolutions are written, in the order of the base array. The channel names, units,
key descriptions and descriptors are the ones of build_mscofd_v4.py, which created the released file from v3.
A copy of the metadata is written next to the file as <name>_sample_metadata.csv. Needs h5py and scipy.
"""

import argparse
import csv
import datetime
import os
import socket
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.build_mscofd_v4 import CHANNELS, CHANNEL_UNITS, KEY_ATTRS, META, NU, describe


def max_pool(sample, size=4):
    """
    reduces a (channels, nx, ny) fine sample to the coarse grid with non-overlapping size x size max pooling
    (all channels, so wall_distance stays in fine-cell units, as in the released resampled_16x_tlast_6ch)
    """
    channels, nx, ny = sample.shape
    return sample.reshape(channels, nx // size, size, ny // size, size).max(axis=(2, 4))


def read_cases(array_directory):
    """
    reads cases.csv of build_arrays.py
    :return: dictionary of case name -> row of the array
    """
    with open(Path(array_directory) / "cases.csv", newline='') as file:
        return {row['case_name']: int(row['row']) for row in csv.DictReader(file)}


def build_dataset(base_directory, fine_directory, output_file, dtype='float32'):
    """
    writes the dataset file
    :param base_directory: output folder of build_arrays.py for base (base_tlast_6ch.npy, cases.csv)
    :param fine_directory: output folder of build_arrays.py for 16x (16x_tlast_6ch.npy, cases.csv)
    :param output_file: the .h5 file
    :param dtype: float32 (default) or float64 (the type of mscofd2000_v4.h5)
    """
    base_directory, fine_directory, output_file = Path(base_directory), Path(fine_directory), Path(output_file)
    base_cases, fine_cases = read_cases(base_directory), read_cases(fine_directory)
    base = np.load(base_directory / "base_tlast_6ch.npy", mmap_mode='r')
    fine = np.load(fine_directory / "16x_tlast_6ch.npy", mmap_mode='r')

    cases = [name for name in base_cases if name in fine_cases]
    missing = sorted(set(base_cases) ^ set(fine_cases))
    if missing:
        print(f"{len(missing)} cases are not in both resolutions and are left out: {missing[:10]}")
    if not cases:
        raise ValueError("No case is in both resolutions")
    n = len(cases)
    print(f"Writing {n} cases to {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    partial_file = output_file.with_name(output_file.name + ".partial")
    rows = []

    with h5py.File(partial_file, 'w') as file:
        # one chunk per sample, so a single sample can be read without reading the others
        shapes = {'base_tlast_6ch': base.shape[1:], 'resampled_16x_tlast_6ch': base.shape[1:],
                  '16x_tlast_6ch': fine.shape[1:]}
        datasets = {key: file.create_dataset(key, shape=(n,) + shape, dtype=dtype, chunks=(1,) + shape)
                    for key, shape in shapes.items()}

        for i, case_name in enumerate(cases):
            base_sample = np.asarray(base[base_cases[case_name]], dtype=np.float64)
            fine_sample = np.asarray(fine[fine_cases[case_name]], dtype=np.float64)
            resampled_sample = max_pool(fine_sample)

            datasets['base_tlast_6ch'][i] = base_sample
            datasets['16x_tlast_6ch'][i] = fine_sample
            datasets['resampled_16x_tlast_6ch'][i] = resampled_sample

            row = describe(i, base_sample, resampled_sample, fine_sample)
            # generation phase of the released file: 1 = normal size, 2 = smaller and larger obstacles
            row['generation_phase'] = 1 if case_name.startswith('normal_') else 2
            row['case_name'] = case_name
            rows.append(row)

            if (i + 1) % 100 == 0 or i + 1 == n:
                print(f"{i + 1} / {n} cases done")

        # attributes, as in the released file
        file.attrs.update(dict(
            name="MS-COFD (Multi-Scale Central Object Flow Dataset)",
            created=datetime.datetime.now().isoformat(timespec="seconds"),
            created_by=f"dataset/build_dataset.py (ms-cofd) on {socket.gethostname()}",
            n_samples=n,
            domain_mm="260 x 120 (streamwise x transverse)",
            axis_order="(sample, channel, x, y); x streamwise, y transverse",
            channels=CHANNELS, channel_units=CHANNEL_UNITS, nu_assumed_m2_s=NU,
            resampling=("resampled_16x_tlast_6ch is 16x_tlast_6ch reduced by non-overlapping 4 x 4 max pooling of "
                        "every channel, as in mscofd2000_v4.h5"),
            note_naming=("The manuscript names SDF1 = object distance and SDF2 = wall distance; this file uses "
                         "explicit names: wall_distance, obstacle_sdf."),
        ))
        for key, dataset in datasets.items():
            dataset.attrs.update(dict(KEY_ATTRS[key], channels=CHANNELS, channel_units=CHANNEL_UNITS))

        group = file.create_group("metadata")
        group.attrs["description"] = "Per-sample descriptors (build_mscofd_v4.describe); row i = sample i."
        for column, unit, description in META:
            dataset = group.create_dataset(column, data=np.array([row[column] for row in rows]))
            dataset.attrs.update(dict(unit=unit, description=description))
        dataset = group.create_dataset("case_name", data=np.array(cases, dtype=object), dtype=h5py.string_dtype())
        dataset.attrs.update(dict(unit="-", description="case name, links the sample to manifest.csv and its "
                                                        ".geo / .msh files"))

    os.replace(partial_file, output_file)

    # the same metadata as a CSV file
    columns = ['case_name'] + [column for column, _, _ in META]
    with open(output_file.with_name(output_file.stem + "_sample_metadata.csv"), 'w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[column] if isinstance(row[column], (str, int, np.integer)) else f"{row[column]:.8g}"
                             for column in columns])

    print(f"Dataset written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the MS-COFD HDF5 file from the base and 16x arrays")
    parser.add_argument("base_dir", help="build_arrays.py output of base, e.g. outputs/arrays/base")
    parser.add_argument("fine_dir", help="build_arrays.py output of 16x, e.g. outputs/arrays/16x")
    parser.add_argument("output", help="the .h5 file, e.g. outputs/ms-cofd.h5")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = parser.parse_args()
    build_dataset(args.base_dir, args.fine_dir, args.output, dtype=args.dtype)
