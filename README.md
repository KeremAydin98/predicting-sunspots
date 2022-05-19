# predicting-sunspots
Time series forecast on sunspot data

Data link:
https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv

## Dataset.window() indexing

```
def windowed_dataset(series, window_size, batch_size):

  series = tf.expand_dims(series, axis=-1)

  ds = tf.data.Dataset.from_tensor_slices(series)

  """
  Input:
  [[1,2,3,4,5,6,7]]
  Output:
  [[1,2,3,4,5],
  [2,3,4,5,6],
  [3,4,5,6,7]]
  """
  ds = ds.window(window_size + 1, shift=1, drop_remainder=True)

  ds = ds.flat_map(lambda w: w.batch(window_size + 1))

  ds = ds.shuffle(1000)

  ds = ds.map(lambda w: (w[:-1],w[1:]))

  return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```
![index](https://user-images.githubusercontent.com/77073029/169332913-e7a1bd18-8a8e-41e1-87ee-955878a17acd.png)

## Numpy indexing

```
def get_windows(x, window_size, horizon):

  window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)

  window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)),axis=0).transpose()

  windowed_array = x[window_indexes]

  windows, labels = get_labelled_windows(windowed_array,horizon)

  return windows, labels
```

![index2](https://user-images.githubusercontent.com/77073029/169333346-6dcf9b7f-05a8-4f32-b109-49f05c609bdc.png)
