import flwr as fl
import tensorflow as tf
import sys

model = tf.keras.applications.MobileNetV2((32,32,3),classes=10,weights=None)
model.compile("adam","sparse_categorical_crossentropy",metrics=["accuracy"])

(x_train,y_train),(x_test,y_test)  = tf.keras.datasets.cifar10.load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(
            x_train,
            y_train,
            epochs=30,
            verbose=1)

        hist = r.history
        print("Fit history : ", hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(
            x_test, y_test, verbose=1, batch_size=32)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:"+str(sys.argv[1]),
    client=FlowerClient(),
    grpc_max_message_length=1024*1024*1024
)
