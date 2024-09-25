from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input


class ShallowNet:
    @staticmethod
    def build(width, height, depth)->Sequential:
        # initialize the model along with the input shape to be
        # channels last
        input_shape = (height, width, depth)


        # define the first and only CONV => RELU layer
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv2D(32, (3, 3),
                       padding="same",  activation='relu'),
                Flatten(),
                Dense(1, activation='sigmoid'),
            ]
        )
        return model