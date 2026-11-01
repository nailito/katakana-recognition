import tensorflow as tf

def load_image(path,label):
    test_path = path
    image_bytes = tf.io.read_file(test_path)
    decoded_image = tf.io.decode_png(image_bytes)
    image = tf.image.convert_image_dtype(decoded_image,tf.float32)
    return image,label


def build_dataset(metadata, batch_size_training=32,seed=12):
    paths = metadata["image_path"].astype(str).to_numpy()
    labels = metadata["label"].to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths,labels))
    ds_shuffled = ds.shuffle(buffer_size=len(metadata),seed=seed)

    metadata_ds = (ds_shuffled
                .map(load_image,num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size_training)
                .prefetch(tf.data.AUTOTUNE))

    return metadata_ds