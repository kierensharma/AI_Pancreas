import os
import time
import pandas as pd
import datetime
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds

  def plot(self, model=None, plot_col='Blood-sugar', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} [normed]')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
              label='Inputs', marker='.', zorder=-10)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Time [h]')
    plt.show()

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

def main():
  df = pd.read_csv('interpolated.csv', parse_dates=[0])
  df.pop('Code')

  # Adds column with rough carb intake
  df.loc[df['Event'] == 'Breakfast', 'Carbs'] = 40
  df.loc[df['Event'] == 'Lunch', 'Carbs'] = 50
  df.loc[df['Event'] == 'Dinner', 'Carbs'] = 75
  df.loc[df['Event'] == 'Snack', 'Carbs'] = 10
  df.pop('Event')
  df['Carbs'] = df['Carbs'].fillna(0)

  date_time = pd.to_datetime(df.pop('Date-time'))
  timestamp_s = date_time.map(pd.Timestamp.timestamp)

  # Creation of 'time of day signal' due to periodicity of data
  day = 24*60*60

  df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
  df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))

  # Splitting of full dataframe into training, validation and testing sets
  column_indices = {name: i for i, name in enumerate(df.columns)}

  n = len(df)
  train_df = df[0:int(n*0.7)]
  val_df = df[int(n*0.7):int(n*0.9)]
  test_df = df[int(n*0.9):]

  num_features = df.shape[1]

  # Normilisation of data to scale features
  train_mean = train_df.mean()
  train_std = train_df.std()

  train_df = (train_df - train_mean) / train_std
  val_df = (val_df - train_mean) / train_std
  test_df = (test_df - train_mean) / train_std

  ############################    Data Windowing   #############################
  w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                    label_columns=['Blood-sugar'], train_df=train_df,
                    test_df=test_df, val_df=val_df)

  w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                    label_columns=['Blood-sugar'], train_df=train_df,
                    test_df=test_df, val_df=val_df)
  
  # Step window looks one hour behind and predicts one hour ahead
  single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['Blood-sugar'], train_df=train_df, test_df=test_df,
    val_df=val_df)

  # Wide window looks one day behind and predicts one day ahead
  wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['Blood-sugar'], train_df=train_df, test_df=test_df,
    val_df=val_df)

  # Window with three hours of input an one hour of labels
  CONV_WIDTH = 3
  conv_window = WindowGenerator(
      input_width=CONV_WIDTH,
      label_width=1,
      shift=1,
      label_columns=['Blood-sugar'], train_df=train_df, 
      test_df=test_df, val_df=val_df)

  LABEL_WIDTH = 24
  INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
  wide_conv_window = WindowGenerator(
      input_width=INPUT_WIDTH,
      label_width=LABEL_WIDTH,
      shift=1,
      label_columns=['Blood-sugar'], train_df=train_df, 
      test_df=test_df, val_df=val_df)

  ##############################   Baseline   ##################################
  baseline = Baseline(label_index=column_indices['Blood-sugar'])
  baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

  val_performance = {}
  performance = {}
  val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
  performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

  ###########################    Linear    ###############################
  linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
  ])

  history = compile_and_fit(linear, single_step_window)
  val_performance['Linear'] = linear.evaluate(single_step_window.val)
  performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

  ###########################     Dense     ##############################

  dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

  history = compile_and_fit(dense, single_step_window)
  val_performance['Dense'] = dense.evaluate(single_step_window.val)
  performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

  #######################    Multi-step dense  #########################
  multi_step_dense = tf.keras.Sequential([
      # Shape: (time, features) => (time*features)
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=1),
      # Add back the time dimension.
      # Shape: (outputs) => (1, outputs)
      tf.keras.layers.Reshape([1, -1]),
  ])

  history = compile_and_fit(multi_step_dense, conv_window)
  val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
  performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

  # conv_window.plot(multi_step_dense)

  ######################   Convolution neural network   #######################
  conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
  ])

  history = compile_and_fit(conv_model, conv_window)
  val_performance['Conv'] = conv_model.evaluate(conv_window.val)
  performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

  # wide_conv_window.plot(conv_model)

  ######################   Recurrent neural network   #######################
  lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
  ])

  history = compile_and_fit(lstm_model, wide_window)
  val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
  performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)

  # wide_window.plot(lstm_model)

  # Plot comparison of all methods
  x = np.arange(len(performance))
  width = 0.3
  metric_name = 'mean_absolute_error'
  metric_index = lstm_model.metrics_names.index('mean_absolute_error')
  val_mae = [v[metric_index] for v in val_performance.values()]
  test_mae = [v[metric_index] for v in performance.values()]

  plt.ylabel('mean_absolute_error [Blood-sugar, normalized]')
  plt.bar(x - 0.17, val_mae, width, label='Validation')
  plt.bar(x + 0.17, test_mae, width, label='Test')
  plt.xticks(ticks=x, labels=performance.keys(),
            rotation=45)
  plt.legend()
  plt.show()


def compile_and_fit(model, window, patience=2):
  MAX_EPOCHS = 20
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

if __name__ == '__main__':
    main()
