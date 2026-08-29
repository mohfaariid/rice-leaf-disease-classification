"""Pembagian dataset train, validation, dan test."""

import tensorflow as tf


def split_validation_test(val_test_ds: tf.data.Dataset) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Membagi dataset validation TensorFlow menjadi validation dan test.

    Notebook awal mengambil 30 persen data sebagai `val_test_ds`, lalu membagi
    dataset tersebut menjadi dua bagian berdasarkan jumlah batch.
    """
    val_test_batches = tf.data.experimental.cardinality(val_test_ds)
    test_ds = val_test_ds.take(val_test_batches // 2)
    val_ds = val_test_ds.skip(val_test_batches // 2)
    return val_ds, test_ds


def count_batches(dataset: tf.data.Dataset) -> int:
    """Menghitung jumlah batch pada dataset."""
    return int(tf.data.experimental.cardinality(dataset).numpy())
