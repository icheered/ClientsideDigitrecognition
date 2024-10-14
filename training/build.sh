# bash
export TF_USE_LEGACY_KERAS=1
poetry run python main.py
poetry run tensorflowjs_converter --input_format keras mnist_model.h5 tfjs_model
cp -r tfjs_model/* ../tfjs_model