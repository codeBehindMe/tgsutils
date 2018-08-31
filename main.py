from tensorflow.python.keras import Model

from engine.datafeed import TensorFeed
from engine.models import *


def run_model(model_fn, optimizer='adam', loss='binary_crossentropy',
              steps_per_epoch=1, epochs=1):
    inputs, outputs = model_fn(128, 128, 1)
    _model = Model(inputs=[inputs], outputs=[outputs])

    _model.compile(optimizer=optimizer, loss=loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Write the session graph to the logs.
        tf.summary.FileWriter('./logs/', sess.graph)

        _model.fit(TensorFeed().build_dataset().dataset,
                   steps_per_epoch=steps_per_epoch, epochs=epochs)

    return _model


if __name__ == '__main__':
    trained = run_model(kaggle_u_net_direct, steps_per_epoch=10, epochs=10)
