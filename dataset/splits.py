"""
Fixed, seeded splits of the dataset (config/dataset.yaml, splits block):

    ood_blockage, ood_rotation, ood_position   out-of-distribution test sets: the cases above the ood_quantile of the
                                               blockage ratio, of |rotation| and of the distance of the position shift
                                               from the middle of its range (shift_x and shift_y divided by their half
                                               ranges); a case in more than one tail goes to the first set of `ood`
    train, val, test                           the other cases, for every size and family in the given fractions,
                                               shuffled with a generator seeded with the seed, the size and the family

The result depends only on the cases (manifest rows) and the settings, not on their order.
"""

import math
import random

SPLITS = ['train', 'val', 'test']


def ood_measures(placement):
    """{name: function(manifest row) -> value whose upper tail is out of distribution}"""
    (x0, x1), (y0, y1) = placement['shift_x'], placement['shift_y']

    def position(row):
        return math.hypot((float(row['shift_x']) - (x0 + x1) / 2) / ((x1 - x0) / 2),
                          (float(row['shift_y']) - (y0 + y1) / 2) / ((y1 - y0) / 2))

    return {'blockage': lambda row: float(row['blockage_ratio']),
            'rotation': lambda row: abs(float(row['rotation'])),
            'position': position}


def assign_splits(rows, settings, placement):
    """
    :param rows: manifest rows of the cases of the dataset
    :param settings: splits block of config/dataset.yaml (seed, ood_quantile, ood, fractions)
    :param placement: placement block (ranges of the shifts)
    :return: {case name: split}
    """
    rows = sorted(rows, key=lambda row: row['case_name'])
    measures = ood_measures(placement)
    split = {}
    for name in settings['ood']:
        values = sorted(measures[name](row) for row in rows)
        threshold = values[min(len(values) - 1, math.ceil(settings['ood_quantile'] * len(values)) - 1)]
        for row in rows:
            if row['case_name'] not in split and measures[name](row) > threshold:
                split[row['case_name']] = f"ood_{name}"

    classes = {}
    for row in rows:
        if row['case_name'] not in split:
            classes.setdefault((row['size'], row['geometry']), []).append(row['case_name'])
    fractions = settings['fractions']
    for (size, geometry), names in sorted(classes.items()):
        random.Random(f"{settings['seed']}-{size}-{geometry}").shuffle(names)
        n_train = round(fractions['train'] * len(names))
        n_val = round(fractions['val'] * len(names))
        for i, name in enumerate(names):
            split[name] = 'train' if i < n_train else 'val' if i < n_train + n_val else 'test'
    return split
