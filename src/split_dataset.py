import numpy as np

def split(metadata, training=0.7, validation=0.15, test=0.15, seed=12):
    proportions = [training, validation, test]

    if any(p < 0 or p > 1 for p in proportions):
        raise ValueError("Each proportion must be between 0 and 1.")

    if not np.isclose(sum(proportions), 1):
        raise ValueError("The proportions must sum to 1.")

    if metadata["writer_id"].isna().any():
        raise ValueError("Some writer identifiers are missing.")

    writers = np.sort(metadata["writer_id"].unique())
    writers = np.random.default_rng(seed).permutation(writers)

    train_end = int(len(writers) * training)
    valid_end = train_end + int(len(writers) * validation)

    writer_train = writers[:train_end]
    writer_valid = writers[train_end:valid_end]
    writer_test = writers[valid_end:]

    df_train = metadata[metadata["writer_id"].isin(writer_train)].copy()
    df_valid = metadata[metadata["writer_id"].isin(writer_valid)].copy()
    df_test = metadata[metadata["writer_id"].isin(writer_test)].copy()

    return df_train, df_valid, df_test