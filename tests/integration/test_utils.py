from mbd_core.data.farcaster.utils import clean_text


def test_clean_text(farcaster_casts_dataframe):
    clean_df = clean_text(farcaster_casts_dataframe, "text", "timestamp")
    assert clean_df.shape[0] <= farcaster_casts_dataframe.shape[0]
