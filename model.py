import tensorflow as tf

class SR_CNN(tf.keras.Model):
    def __init__(self) -> None:
        super(SR_CNN, self).__init__()
        self.upsampling = tf.image.resize
        self.cnn1 = tf.keras.layers.Conv2D(3, 3, input_shape=[None, None, 3], padding='SAME')
        self.cnn2 = tf.keras.layers.Conv2D(3, 3, padding='SAME')
        self.cnn3 = tf.keras.layers.Conv2D(3, 3, padding='SAME')

    '''def compile(self, optimizer, loss, acc_metric):
        super(SR_CNN, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.acc_metric = acc_metric'''

    def call(self, x, training=False):

        x = self.upsampling(x, [x.shape[1] * 2, x.shape[2] * 2])
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        return x

    def fit(self, data, batch_size, epochs):
        data = data.batch(batch_size)
        for epoch in range(epochs):
            for d in data:
                loss = self.train_step(d)
                tf.print('Loss: ', loss['loss'])
        return loss['loss']
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data[0], data[1]

        with tf.GradientTape() as tape:
            y_pred = self.call(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}