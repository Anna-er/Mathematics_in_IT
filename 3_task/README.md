# Сингулярное разложение и сжатие изображений в формате BMP

## Описаниие
Утилита способна:
- `compress` - формировать промежуточное представление на основе исходного изображения
- `decompress` -восстановливать изображение, используя промежуточное представление


Сингулярное разложение реализовано тремя способами: 
- `numpy` -`numpy.linalg.svd(...)`
- `simple` - использует [степенной метод](https://www.jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/)
- `advance` - использует [блочный степенной метод](https://www.degruyter.com/document/doi/10.1515/jisys-2018-0034/html)
## Результаты

Работа методов почти не отличается друг от друга. Например, вот разультаты сжатия в 2 раза:
| Numpy                            | Power simple                            | Block power                            |
|----------------------------------|-----------------------------------------|----------------------------------------|
| ![example](img/2/example_numpy.bmp) | ![example](img/2/example_simple.bmp) | ![example](img/2/example_advanced.bmp) |

А вот разультаты сжатия в 4 раза:
| Numpy                            | Power simple                            | Block power                            |
|----------------------------------|-----------------------------------------|----------------------------------------|
| ![Lena](img/4/Lena_numpy.bmp) | ![Lena](img/4/Lena_simple.bmp) | ![Lena](img/4/Lena_advanced.bmp) |


Теперь сожмем довольно большое(`1000 * 667`) изображение в 2, 4, 8, 16, 32 и 64 раз:
- на фото ниже результат работы `advance` метода. С другими результатами можно ознакомиться [здесь](img/deg)
- Время работы отличалось -- самым долгим оказался метод `advanced`, самым быстрым -- `numpy`


| <!-- -->      | <!-- -->        | <!-- -->      |
|:-------------:|:---------------:|:-------------:|
| ![mountains](img/deg/mountains_advanced_2.bmp) | ![mountains](img/deg/mountains_advanced_4.bmp) | ![mountains](img/deg/mountains_advanced_8.bmp) |
| ![mountains](img/deg/mountains_advanced_16.bmp) | ![mountains](img/deg/mountains_advanced_32.bmp) | ![mountains](img/deg/mountains_advanced_64.bmp) |

## Вывод

Реализованные методы достаточно корректно работают.

Методы в порядке увеличения скорости выполнения:
1. `numpy`
2. `advance`
3. `simple`
