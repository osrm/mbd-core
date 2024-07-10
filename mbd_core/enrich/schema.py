"""Enrich schema for users and items."""

from typing import cast

import numpy as np
import pandas as pd
import pandera as pa

from mbd_core.data.schema import (
    ITEM_COLUMN,
    PROTOCOL_COLUMN,
    USER_COLUMN,
    USER_UPDATE_TIME_COLUMN,
)
from mbd_core.enrich.labelling.load_config import load_label_columns

LABEL_COLUMNS = load_label_columns()

ITEM_SEM_EMBED_COLUMN = "item_sem_embed"
ITEM_AI_LABELS_COLUMN = "ai_labels"

USER_SEM_EMBED_COLUMN = "user_sem_embed"
MBD_ID_COLUMN = "mbd_id"


def _check_sequence(x: pd.Series) -> bool:
    return cast(bool, x.apply(lambda x: isinstance(x, (np.ndarray | list))).all())


ITEM_ENRICH_SCHEMA = pa.DataFrameSchema(
    {
        ITEM_COLUMN: pa.Column(str),
        ITEM_SEM_EMBED_COLUMN: pa.Column(checks=pa.Check(_check_sequence)),
        **{label: pa.Column(float) for label in LABEL_COLUMNS},
        ITEM_AI_LABELS_COLUMN: pa.Column(list[str]),
    },
    strict=False,
)

USER_ENRICH_SCHEMA = pa.DataFrameSchema(
    {
        USER_COLUMN: pa.Column(str),
        PROTOCOL_COLUMN: pa.Column(str),
        USER_UPDATE_TIME_COLUMN: pa.Column("datetime64[ns, UTC]"),
        MBD_ID_COLUMN: pa.Column(str, nullable=True, required=False),
        USER_SEM_EMBED_COLUMN: pa.Column(dict, nullable=True, required=False),
        **{
            label: pa.Column(dict, nullable=True, required=False)
            for label in LABEL_COLUMNS
        },
    },
    strict=False,
)
