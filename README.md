Лабораторная работа №4
====
# Цель лабораторной работы
Исследовать влияние различных техник аугментации данных на процесс обучения нейронной сети на примере решения задачи классификации Food-101 с использованием техники обучения Transfer Learning

# 1. С использованием, техники обучения Transfer Learning и оптимальной политики изменения темпа обучения, определенной в ходе выполнения лабораторной №3, обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Food-101 с использованием следующих техник аугментации данных:
 * Случайное горизонтальное и вертикальное отображение
 * Использование случайной части изображения
 * Поворот на случайный угол
 
## Случайное горизонтальное и вертикальное отображение
 
В нашем случае использовалась сторонняя функция
 ```
tf.keras.layers.experimental.preprocessing.RandomFlip(...)
```
[источник](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/RandomFlip)

### Графики обучения для сети EfficientNet-B0

**График метрики точности:** 
<img src="./random-flip/accuracy.png">
<img src="./random-flip/epoch_categorical_accuracy.svg">

**График функции потерь:**
<img src="./random-flip/loss.png">
<img src="./random-flip/epoch_loss.svg">

### Вывод:
//TODO


## Использование случайной части изображения
 
В нашем случае использовалась сторонняя функция
```
tf.keras.layers.experimental.preprocessing.RandomCrop(...)
```
[источник](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/RandomCrop)

### Графики обучения для сети EfficientNet-B0

**График метрики точности:** 
<img src="./random-crop/accuracy.png">
<img src="./random-crop/epoch_categorical_accuracy.svg">

**График функции потерь:**
<img src="./random-crop/loss.png">
<img src="./random-crop/epoch_loss.svg">

### Вывод:
//TODO


## Поворот на случайный угол
 
В нашем случае использовалась сторонняя функция
```
tf.keras.layers.experimental.preprocessing.RandomRotation(...)
```
[источник](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/RandomRotation)

### Графики обучения для сети EfficientNet-B0

**Графики метрики точности:** 
<img src="./random-rotation/0.1/accuracy.png">
<img src="./random-rotation/0.1/epoch_categorical_accuracy.svg">

<img src="./random-rotation/0.5/accuracy.png">
<img src="./random-rotation/0.5/epoch_categorical_accuracy.svg">

<img src="./random-rotation/1/accuracy.png">
<img src="./random-rotation/1/epoch_categorical_accuracy.svg">

**Графики функции потерь:**
<img src="./random-rotation/0.1/loss.png">
<img src="./random-rotation/0.1/epoch_loss.svg">

<img src="./random-rotation/0.5/loss.png">
<img src="./random-rotation/0.5/epoch_loss.svg">

<img src="./random-rotation/1/loss.png">
<img src="./random-rotation/1/epoch_loss.svg">

### Вывод:
//TODO


### Анализ результатов


