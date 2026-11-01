import tensorflow as tf

def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    decoded_image = tf.io.decode_png(image_bytes, channels=1)
    image = tf.image.convert_image_dtype(decoded_image, tf.float32)
    return image, label


def build_dataset(metadata, batch_size=32, training=False, seed=12):
    """Load normalized image batches; shuffle only for training."""
    paths = metadata["image_path"].astype(str).to_numpy()
    labels = metadata["label"].to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(metadata), seed=seed)

    return (
        ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
