import tensorflow as tf
tf.debugging.set_log_device_placement(True)
# In tensorflow tf.Module is the base class of Model and Layer.
class OurModel:
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(num_inputs, )))
        self.model.add(tf.keras.layers.Linear(num_input_dims=num_inputs, units=5))
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(tf.keras.layers.Linear(num_input_dims=5, units=20))
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(tf.keras.layers.Linear(num_input_dims=20, units=num_classes))
        self.model.add(tf.keras.layers.Dropout(dropout_prob))
        self.model.add(tf.keras.layers.Softmax(axis=1))
    

if __name__ == "__main__":
    net = OurModel(num_inputs=2, num_classes=3)
    print(net)
    v = tf.constant([[2, 3]])
    out = net.model(v)
    print(out)
    print("Cuda's availability is %s" % len(tf.config.list_physical_devices('GPU')))
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/GPU:0'):
            v1 = tf.constant([[2, 3]])
        print("Data from cuda: %s" % v1)
    
