def test_tf():
    import tensorflow as tf
    from utils import list_devices

    list_devices()
    gpu_available = tf.test.is_gpu_available()
    print('GPU available:', gpu_available)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)).as_default() as sess:
        print('Session has started!')


def test_pt():
    import torch
    print('GPU available:', torch.cuda.is_available())


if __name__ == '__main__':
    test_tf()
    # test_pt()
