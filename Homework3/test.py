from keras.engine.saving import load_model

import process_data

if __name__ == "__main__":
    x_test, y_test = process_data.load_full_numpy_arrays("test", "processed_data")
    model = load_model('model-emnist-nn.h5')

    y_test = process_data.transform_labels(y_test)
    print(model.evaluate(x_test, y_test))
