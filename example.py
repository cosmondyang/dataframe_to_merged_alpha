"""Small driver showcasing alpha_combo_generator_v6 helpers.

The goal of this example is purely educational: it demonstrates how the
workspace groups features into the five families (price/time/volume/ratio_corr/count),
how the random plan buffer can be consulted, and how the builder stitches a
random alpha together.  The heavy combo search logic in ``alpha_combo_generator_v6``
remains the primary entry-point for production runs.
"""

import random
from typing import List

import numpy as np
import pandas as pd

from alpha_combo_generator_v6 import (
    AlphaBuilder,
    AlphaWorkspace,
    OperatorLibrary,
    RandomPlanBuffer,
)


def _mock_dataframe() -> pd.DataFrame:
    dates = pd.Index(["2021-01-01", "2021-01-02", "2021-01-03"], name="pred_date")
    codes = pd.Index(["000001", "000002"], name="code")
    index = pd.MultiIndex.from_product([dates, codes])

    rng = np.random.default_rng(7)
    def _col(_: str) -> List[float]:
        return rng.normal(loc=0.0, scale=1.0, size=len(index)).tolist()

    data = {
        "price_close": _col("price"),
        "time_open_gap": _col("time"),
        "volume_le300_buy_bin3": _col("volume"),
        "ratio_volume_imbalance_bin4": _col("ratio"),
        "corr_price_last_digit": _col("corr"),
        "count_price_last_digit_2": _col("count"),
    }
    return pd.DataFrame(data, index=index)


def main() -> None:
    df = _mock_dataframe()
    rng = random.Random(17)
    workspace = AlphaWorkspace(df, rng=rng)
    operator_lib = OperatorLibrary(workspace, rng=rng)
    builder = AlphaBuilder(operator_lib, rng=rng)
    plan_buffer = RandomPlanBuffer("combo_generator_tmp.json", refill_size=5)
    plan_buffer.prime(rng)

    print("\n[example] Available families and sample columns:")
    for family, cols in workspace.categories.items():
        sample = ", ".join(cols[:2]) if cols else "<empty>"
        print(f"  - {family}: {len(cols)} columns | sample: {sample}")

    seed = plan_buffer.next_seed(rng)
    print(f"\n[example] Seed used for this alpha: {seed}")
    local_rng = random.Random(seed)
    workspace.set_rng(local_rng)
    operator_lib.set_rng(local_rng)
    builder.set_rng(local_rng)

    series, structure = builder.build()
    print("[example] Generated alpha summary:")
    print(f"  depth = {structure.depth}")
    for idx, step in enumerate(structure.composite_steps, 1):
        print(
            "   composite#{idx}: {desc} | families={families} | op={op} | sources={src}".format(
                idx=idx,
                desc=step.description,
                families=step.families,
                op=step.operator,
                src=[s.column for s in step.sources],
            )
        )
    for idx, step in enumerate(structure.unary_steps, 1):
        print(
            f"   unary#{idx}: {step.name} | params={step.params}"
        )
    for idx, step in enumerate(structure.binary_steps, 1):
        print(
            f"   binary#{idx}: {step.name} | params={step.params}"
        )
    print(f"\n[example] Series preview:\n{series.head()}")


if __name__ == "__main__":
    main()
