"""Enrich schema for users and items."""

from typing import cast

import numpy as np
import pandas as pd
import pandera as pa

from mbd_core.enrich.labelling.load_config import load_label_columns

LABEL_COLUMNS = load_label_columns()

ITEM_SEM_EMBED_COLUMN = "item_sem_embed"
USER_SEM_EMBED_COLUMN = "user_sem_embed"


def _check_sequence(x: pd.Series) -> bool:
    return cast(bool, x.apply(lambda x: isinstance(x, (np.ndarray | list))).all())


ITEM_ENRICH_SCHEMA = pa.DataFrameSchema(
    {
        ITEM_SEM_EMBED_COLUMN: pa.Column(checks=pa.Check(_check_sequence)),
        **{label: pa.Column(float) for label in LABEL_COLUMNS},
    },
    strict=False,
)

USER_ENRICH_SCHEMA = pa.DataFrameSchema(
    {
        USER_SEM_EMBED_COLUMN: pa.Column(checks=pa.Check(_check_sequence)),
        **{label: pa.Column(float) for label in LABEL_COLUMNS},
    },
    strict=False,
)
