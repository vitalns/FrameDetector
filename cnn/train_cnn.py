from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout, GlobalAveragePooling2D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np


def train_model(data_dir, model_save_path='model.h5'):
    # Параметры
    img_size = (128, 128)
    classes = ['circle', 'square', 'triangle']
    batch_size = 32
    epochs = 50

    # Аугментация данных
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # Аугментация на поворот и сдвиг задана при генерации обучающего датасета
        # rotation_range=15,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # Добавили аугментацию масштаба, тк в предобработке ROI используется cv2.resize(interpolation=cv2.INTER_NEAREST)
        # (Метод INTER_NEAREST сохраняет бинарную структуру кадра [0/255])
        zoom_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    # Загрузка данных
    train_gen_flow = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen_flow = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Проверка валидационного набора
    x_tr, y_tr = next(train_gen_flow)
    print(f"Форма загруженных тренировочных данных: {x_tr.shape}")

    # Визуализируем первые 32 изображения
    plt.figure(figsize=(25, 15))
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.imshow(x_tr[i].squeeze(), cmap='gray')  # для серых изображений
        plt.title(f"Label: {np.argmax(y_tr[i])}")
    plt.tight_layout()
    plt.savefig('train_set.png', dpi=300)
    plt.show()

    # Проверка обучающего набора
    x_val, y_val = next(val_gen_flow)
    print(f"Форма загруженных валидационных данных: {x_val.shape}")

    # Визуализируем первые 5 изображений
    plt.figure(figsize=(25, 15))
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.imshow(x_val[i].squeeze(), cmap='gray')  # для серых изображений
        plt.title(f"Label: {np.argmax(y_val[i])}")
    plt.tight_layout()
    plt.savefig('validation_set.png', dpi=300)
    plt.show()

    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', # Автоматическая остановка
                                   patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,  # Уменьшение скорости обучения при застревании
                                  patience=3, min_lr=1e-4)
    model_checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy',  # Сохранение только лучшей версии модели
                                       save_best_only=True, mode='max', verbose=1)
    early_stop = [early_stopping, reduce_lr, model_checkpoint]
    # В качестве монитора может быть monitor='val_loss' и mode='min'

    # Модель CNN
    # Архитектура ниже достигает точности >98% на датасете геометрических фигур QuickDraw.
    #
    # Для нашем случае (простые фигуры, 3 класса)
    # При датасете < 10000:
    # - Dropout(0.5) в последнем блоке замедляет сходимость в 2 раза, но кривые train и val идут близко друг к другу,
    # при этом val_ac>train_ac.
    # - При использовании в блоках BatchNormalization модель не обучается (val_accuracy=0.33), поэтому в модели
    # отключены эти слои (слишком маленький обучающий датасет).
    # Если оставить BN только в одном блоке (2 или 3) модель учится сходимость гладкая, точность на том же уровне.
    # - GlobalAveragePooling2D улучшает точность, но кривые становятся менее гладкие.
    # (val_accuracy=0.33) при маленьком датасете (<10000), если оставить слой BN только в одном
    # поэтому многие слои отключены (во втором блоке Dropout и BatchNormalization активированы для примера).
    #
    # GlobalAveragePooling2D заменяет полносвязные слой (Flatten + Dense) на усреднение,
    # уменьшает колличество параметров и подавляет переобучение, очень полезен при малом датасете.
    # Перед использованием обязательно должно быть несколько сверточных слоев, а в последнем не менее 128.
    # GAP может ускорить сходимость val_loss, однако если данных много и GAP слишком упрощает модель,
    # то сходимость замедляется (недообучение).
    # При малых данных может улучшить точность, а на больших данных может незначительно снизить.

    # Dropout полезен при малом датасете, сильно снижает риск переобучения (коэффициент 0.5 - 0.7).
    # Для больших датасетов менее полезен (коэффициент 0.2 - 0.3).
    # (признаки переобучения - большая разница между train и val accuracy)
    # Ставится после активации и действует только на слой к которому подключён.
    # Dropout замедляет сходимость (нужно больше эпох), в начале обучения val_loss может сильнее колебаться,
    # часто приводит к лучшей точности и уменьшает разрыв между train и val accuracy.

    # BatchNormalization ускоряет обучение, позволяет использовать высокие learning rates (>0.1).
    # Максимально эффективен при большом датасете, стабильно работает с размером батча >32,
    # при размере батча менее 16 используется метод нормализации GroupNormalization.
    # BatchNormalization используется после свертки перед активацией.
    # BN ускоряет сходимость, снижает риск застревания в локальных минимумах.
    # На сложный датасетах может улучшить точность, на маленьких может привести к ухудшению (особенно при батчах < 16)

    # Dropout и BatchNormalization работают лучше вместе.
    # Dropout и BatchNormalization заметно замедляют сходимость.

    model = Sequential([
        # Блок 1 (базовые признаки)
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
        # BatchNormalization(),
        # Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Блок 2 (комплексные признаки)
        Conv2D(64, (3, 3), activation='relu'),
        # BatchNormalization(),
        # Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        # Блок 3 (глобальные признаки)
        Conv2D(128, (3, 3), activation='relu'),
        # BatchNormalization(),
        # Activation('relu'),
        # MaxPooling2D(2, 2),
        # Dropout(0.4),

        GlobalAveragePooling2D(),
        # BatchNormalization(),

        # Финальные слои и классификация
        # Flatten(),
        # Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(classes), activation='softmax')
    ])
    model.summary()

    # Простая модель для проверки
    # (Модель быстро переобучается: train_accuracy выходит на плато за 5 эпох,
    # val_accuracy останавливается на уровне 40-60%)
    # model = Sequential([
    #     Flatten(input_shape=(img_size[0], img_size[1], 1)),
    #     Dense(64, activation='relu'),
    #     Dense(len(classes), activation='softmax')
    # ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Можно без импорта указать optimizer='adam'
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    history = model.fit(train_gen_flow,
                        validation_data=val_gen_flow,
                        epochs=epochs,
                        callbacks=early_stop)
    print(f"Обучение остановленно на эпохе: {early_stopping.stopped_epoch}")
    print(f"Лучшая валидационная точность: {round(max(history.history['val_accuracy']), 2)}")
    print(f"История LR: {history.history['lr']}")

    model.save(model_save_path)
    print(f"Модель сохранена как {model_save_path}")


    # Создаем графики сходимости
    plt.figure(figsize=(14, 5))
    # Графики точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # Графики потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('training_convergence.png', dpi=300)
    plt.show()

train_model('cnn_data')



