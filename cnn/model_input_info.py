from tensorflow.keras.models import load_model

model = load_model('model.h5')

print(f'Входная форма: {model.input_shape}')  # (None, 128, 128, 1)
print(f'Тип входных данных: {model.input.dtype}')  # float32