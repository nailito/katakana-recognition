import tensorflow as tf

def load_image(path,label):
    test_path = path
    image_bytes = tf.io.read_file(test_path)
    decoded_image = tf.io.decode_png(image_bytes)
    image = tf.image.convert_image_dtype(decoded_image,tf.float32)
    return image,label


def build_dataset(metadata, batch_size_training,seed):
    pass