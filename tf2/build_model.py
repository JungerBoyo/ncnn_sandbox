from tensorflow import keras

### MODEL BUILDING
input = keras.Input(shape=(64, 64, 3))

x = keras.layers.Rescaling(1./255.)(input)

x = keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
for i in (64, 128):
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=i, kernel_size=3, activation="relu")(x)

x = keras.layers.Flatten()(x)

output = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=input, outputs=output)

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

path_to_models_dir = "../static_res/tf2Models/"

model.save(path_to_models_dir + "basic_img_classifier64x64.h5")
