# MACHINE LEARNING - STARS TYPE CLASSIFICATION PREDICTION

## Wstęp

Projekt ma na celu dokonanie analizy, wizualizacji i stworzenie modelu dla zbioru danych. Używany zbiór danych zawiera
specyfikację gwiazd w tym ich typy.

[Dataset](https://www.kaggle.com/brsdincer/star-type-classification)

### Podział zbioru danych

* Features:

    * Liczbowe:
        * Temperatura (K)
        * Jasność (L/Lo)
        * Promień (R/Ro)
        * Absolutna wielkość gwiazdowa (Mv)

    * Tekstowe:
        * Kolor
        * Klasa wydmowa

* Target:
    * Typ gwiazdy

### Jednostki

Lo = _3.828 x 10^26 W_ – jasność Słońca

Ro = _6.9551 x 10^8 m_ – promień Słońca

## Wizualizacja zbioru

### Intro

Rozpoczynamy poprzez pokazanie zależności pomiędzy parametrami. ScatterMatrix pokazujący jak zależą od siebie wartości.

![ScatterMatrix](https://github.com/ML-GroupB/ml-star-type-classification-data/img/scatter_matrix.png)

### Wartości liczbowe — features

####1. Temperatura

Rozkład temperatury (boxplot) pokazuje, że najwięcej gwiazd ma temperaturę z zakresu 4000K - 15000K. Mediana to ok
6000K. Niektóre gwiazdy mają jednak temperatury sięgające znacznie więcej.

![Temperatue](https://github.com/ML-GroupB/ml-star-type-classification-data/img/temp.png)


####2. Jasność

Światłość waha się bardziej niż temperatura, dlatego należy podawać ją logarytmicznie. Mediana jest mniejsza od
wielkości naszego Słońca.

![Luminosity](https://github.com/ML-GroupB/ml-star-type-classification-data/img/lum.png)


####3. Promień

Promienie tak samo, jak jasność należy pokazywać logarytmicznie z uwagi na duży rozrzut. Średni promień gwiazdy jest w
przybliżeniu równy Promieniowi naszego słońca. Większość gwiazd jest 100 razy większa lub mniejsza, lecz są też gwiazdy
mocno oddalone od średniej na wykresie.

![Radius](https://github.com/ML-GroupB/ml-star-type-classification-data/img/radius.png)


####4. Absolutna wielkość gwiazdowa

![Magnitude](https://github.com/ML-GroupB/ml-star-type-classification-data/img/magn.png)


### Wartości kategoryczne

####5. Kolor

W kolorze przeważa czerwony (połowa wszystkich gwiazd ze zbioru jest tego koloru). Znaczący udział w kolorach biorą
Czerwony, Niebieski oraz Biały (najczęściej gwiazda będzie miała jeden z tych lub ich mieszankę).

![Color](https://github.com/ML-GroupB/ml-star-type-classification-data/img/color.png)


####6. Klasa widmowa ([info](https://en.wikipedia.org/wiki/Asteroid_spectral_types))

![Spectral Class](https://github.com/ML-GroupB/ml-star-type-classification-data/img/spec.png)

## Zależności danych w zbiorze danych

### Korelacje

Wykresy korelacji pokazują, że najbardziej zależne są Promień-Jasność oraz Temperatura-Jasność, dlatego warto zobaczyć
wykresy Typu gwiazdy od tych 2 zestawów.

![Correlation](https://github.com/ML-GroupB/ml-star-type-classification-data/img/corel.png)

![Radius / Luminosity](https://github.com/ML-GroupB/ml-star-type-classification-data/img/radius-lum.png)

![Luminosity / Temperature](https://github.com/ML-GroupB/ml-star-type-classification-data/img/lum-temp.png)

Wykres Jasności od Temperatury jest właśnie dlatego dosyć popularny, aby pokazać jak dzielą się gwiazdy ze względu na te
dwa typy.

![Spectral Original](https://astropolis.pl/uploads/post-29939-0-90749600-1460753620.jpg)

### PCA

#### 2D

![PCA-2D](https://github.com/ML-GroupB/ml-star-type-classification-data/img/pca-2d.png)

#### 3D

![PCA-3D](https://github.com/ML-GroupB/ml-star-type-classification-data/img/pca-3d.png)