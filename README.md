# Introduction

This is the repository to define mbd unified data schema such that all the data sources from different protocols can be merged/combined into the same data schema. The unified schema include (and will include) four types of schema:
- item schema
- interaction schema (user-item) 
- user schema
- user-user interaction

In addition, this repository also contains the processing functions to transform the raw data into the unified schema for each protocol.

## Quick Start:

### 1. Install
In private:
```
pip install mbd-core --extra-index-url https://ZKAI-Network.github.io/mbd-pypi-index/
```
Once open to public:
```
pip install mbd-core
```

### 2. Use schema column names
Use the defined column names in your code to avoid hard-coding
```
from mbd_core.data.schema import (
    AUTHOR_ID_COLUMN,
    EDGE_TYPE_COLUMN,
    EMBED_ITEMS_COLUMN,
    EMBED_USERS_COLUMN,
    ITEM_COLUMN,
    ITEM_CREATION_TIME_COLUMN,
    ITEM_TEXT_COLUMN,
)

### Use these columns in your code
df[ITEM_TEXT_COLUMN] =
```

### 3. Schema validation
When you add a data source for new protocol, use the schema to validate your processed data.
```
from mbd_core.data.schema import INTERACTION_SCHEMA, ITEM_META_SCHEMA, USER_META_SCHEMA

your_interaction_df = ...
your_item_df = ...
your_user_df = ...

INTERACTION_SCHEMA.validate(your_interaction_df)
ITEM_META_SCHEMA.validate(your_item_df)
USER_META_SCHEMA.validate(your_user_df)
```


# Contribute

## Note:
- make sure you have python3.10 installed in your local dev environment
- never put dependency in `requirements.txt` directly, always put dependency in `requirements.in`
- `requirements.txt` will be automatically generated by running `make fix-lint`

## Available commands
Usually, you should run below commands in order before creating a PR to contribute.

1. To auto-fix your lint error, auto-format your code, update the requirements, run:
    ```
    make fix-lint
    ```

    This will take a couple of minutes to finish if you run it the first time or you have added new dependencies into `requirements.in`, otherwise this will be very fast.

2. To check if there is any lint error, run:
    ```
    make lint
    ```
    If there is any error that can't be auto-fixed by `make fix-lint`, you need to fix it manually.

3. To run all the tests, and get coverage report, run:
    ```
    make test
    ```
