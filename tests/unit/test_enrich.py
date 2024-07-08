import numpy as np
import pandas as pd

from mbd_core.enrich.labelling.load_config import (
    LABELS_MAP,
    _get_label_keys,
    load_config,
    load_label_columns,
)
from mbd_core.enrich.schema import ITEM_ENRICH_SCHEMA, USER_ENRICH_SCHEMA


def test_label_config():
    config = load_config()
    label_keys = _get_label_keys(config)
    label_columns = load_label_columns()

    assert set(label_keys) == set(LABELS_MAP.keys())
    assert set(label_columns) == {
        LABELS_MAP[label]["api_label"] for label in label_keys
    }


def test_enrich_schema():
    item_df = pd.DataFrame(
        {
            "item_sem_embed": [[1, 2, 3], [4, 5, 6]],
            **{label: [1.0, 0.0] for label in load_label_columns()},
        }
    )
    user_df = pd.DataFrame(
        {
            "user_sem_embed": [np.array([1, 2, 3]), np.array([4, 5, 6])],
            **{label: np.array([1.0, 0.0]) for label in load_label_columns()},
        }
    )
    ITEM_ENRICH_SCHEMA.validate(item_df)
    USER_ENRICH_SCHEMA.validate(user_df)
