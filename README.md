# Praca_magisterska

# Analiza serii pików dyfrakcyjnych w kwazikryształach

## Spis treści
* [Wprowadzenie](#Wprowadzenie)
* [Technologie](#Technologie)
* [Uruchamianie](#Uruchamianie)
* [Wyniki](#Wyniki)
* [Status](#Status)
* [Żródła](#Żródła)

## Wprowadzenie
Przełom XX i XXI wieku przyniósł znaczny rozwój komputeryzacji. Dzisiaj jest ona wszechobecna i w znacznym stopniu ułatwia codzienne funkcjonowanie. Nie inaczej jest w świecie nauki. Znaczny stopień komplikacji modeli matematycznych spowodował, że praktycznie nie ma możliwości prowadzenia obliczeń bez wykorzystania komputera. Algorytmy uczenia maszynowego przenoszą rozważania na zupełnie inny poziom. Ich założeniem jest automatyczne podejmowanie optymalnych decyzji w drodze uczenia się na własnym doświadczeniu. Jednym z zakresów, w których mogą zostać wykorzystane jest dziedzina finansów i ekonomii. Na rynkach giełdowych szybkie i poprawne podejmowanie decyzji jest kluczowe. Notowania akcji charakteryzują się bardzo dużą zmiennością nawet rozpatrując dane dzień po dniu. Warto zaznaczyć, że zmienność ta jest w dodatku bardzo trudna do wyjaśnienia, a żeby podjąć właściwą decyzję należy wziąć pod uwagę nie tylko dane numeryczne, ale również wydarzenia ze świata, które mogą mieć na nie wpływ. Znaczna komplikacja problemu powoduje, że użyteczne mogą się stać algorytmy uczenia maszynowego, które na własnym doświadczeniu mogłyby podejmować odpowiednie decyzje dotyczące kupna, sprzedaży lub zachowania papierów wartościowych w portfelu. Odpowiednie oprogramowanie oczywiście nie byłoby w stanie całkowicie wykluczyć elementu ludzkiego jeśli chodzi o podejmowanie decyzji giełdowych, ale mogłoby w znaczącym stopniu ją ułatwić.

Celem pracy jest zbadanie, czy metody uczenia maszynowego mogą być użyteczne w zagadnieniach finansowych, a konkretnie do prognozowania notowań na giełdzie na podstawie ich danych historycznych oraz wskaźników analizy technicznej. Do badań zostały wykorzystane historyczne dane dotyczące indeksów giełdowych WIG20, S\&P 500, DAX, Nikkei 225, BSE SENSEX oraz FTSE 100 z lat 2010 - 2022, a prognozy wykonano na podstawie modeli drzew decyzyjnych, maszyny wektorów nośnych oraz regresji logistycznej. 

Struktura pracy prezentuje się następująco. W pierwszym rozdziale zaprezentowano opis literatury naukowej z zakresu wykorzystania uczenia maszynowego do prognozowania cen akcji. W drugim rozdziale opisana została metodyka badań, a także przedstawiono teoretyczne opisy  modeli wybranych do estymacji. W kolejnym rozdziale zaprezentowano sposób przygotowania danych, a także zawarto opis wskaźników analizy technicznej wykorzystanych w pracy. W czwartym rozdziale  zaprezentowane zostały wyniki każdego z modeli, a także wybrano najlepszy z nich na podstawie m.in. wskazania trafności prognoz. Na końcu przedstawiono również podsumowanie wyników pracy oraz dalsze plany rozwoju.

## Technologie 
* Python 3.9 (użyte biblioteki: Numpy, Matplotlib, scikit-learn, Seaborn)

## Uruchamianie

## Wyniki
W pracy testowano 6 modeli dla 6 zbiorów danych zawierających dane dzienne indeksów giełdowych. 4 modele pochodziły z grupy modeli drzew decyzyjnych - podstawowy model drzew decyzyjnych, bagging, lasy losowe oraz boosting. Dodatkowo przetestowano również model regresji logistycznej oraz maszyny wektorów nośnych. W poniższych tabelach  zaprezentowano końcowe wskaźniki trafności dla każdego z modeli z wyszczególnieniem testowanych indeksów. 

![Alt text](https://github.com/Lukkud/Praca_magisterska/blob/main/src/wyniki_accuracy_1horyzont.png)

Bazując na uzyskanych wynikach trafności można stwierdzić, że najlepszym modelem do predykcji zmian cen akcji na 1 sesję do przodu jest regresja logistyczna - średnia trafność na poziomie 0,709. Nieznacznie gorsze okazały się modele SVM oraz bagging - odpowiednio 0,708 i 0,706. Gorzej spisały się modele lasów losowych i boostingu, a zdecydowanie najsłabszym był podstawowy model drzew decyzyjnych. Można zauważyć również, że najlepsze prognozy uzyskano dla indeksu S\&P 500, co jest spodziewnym rezultatem biorąc pod uwagę, że to na danych tego indeksu strojono hiperparametry modeli. Najniższe wskaźniki trafności uzyskano dla DAX i WIG 20. Ciekawe wnioski daje również analiza pojedynczych wartości dla poszczególnych modeli. Model SVM uzyskał najwyższą trafność dla indeksu BSE Sensex 30, ale jednocześnie jedną z najniższych wartości dla indeksów DAX oraz UK 100. Można więc stwierdzić, że SVM cechował się wysoką czułością (wariancją). Zupełnie odwrotnie jest dla modeli regresji logistycznej oraz baggingu, które były bardzo stabilne. 

Wszystkie modele sprawdzono również pod kątem prognozowania zmian cen akcji na dwie sesje do przodu. W poniższej tabeli zaprezentowano końcowe wskaźniki trafności dla każdego z modeli z wyszczególnieniem testowanych indeksów. 

![Alt text](https://github.com/Lukkud/Praca_magisterska/blob/main/src/wyniki_accuracy_2horyzont.png)

Również przy dwusesyjnym horyzoncie predykcyjnym najlepszym modelem okazała sie regresja logistyczna ze średnią trafnością na poziomie 0,656. Wysokie wyniki odnotowano również dla modelu lasów losowych i baggingu - odpowiednio 0,645 i 0,651. Najniższe trafności uzyskano dla drzew decyzyjnych. Jeśli chodzi o wskazania dla poszczególnych indeksów to najlepiej w tej kwestii modele prognozowały dla indeksu BSE Sensex 30. Najgorzej w zestawieniu wypada DAX. Zarówno regresja logistyczna jak i bagging są bardzo stabilnymi modelami. 

Dla modeli przewidujących na 1 sesję do przodu otrzymano średnią trafność na poziomie odpowiednio 0,699. Dla horyzontu dwusesyjnego było to już 0,637. Trafność jest więc niższa o 6 punktów procentowych. Trzeba jednak przyznać, że daleko tym wartościom do wyników całkowicie losowych i badane modele mogą być pomocne w kontekście decyzji o kupnie lub sprzedaży. Ciekawa mogłaby być tutaj również analiza dla dłuższego okna predykcji, jednak to wykracza poza zakres tej pracy.

Jeśli chodzi o wpływ poszczególnych zmiennych na zdolności prognostyczne to wykorzystano do tego metodę permutacyjnej oceny istotności. Dla wszystkich modeli z wyjątkiem SVM najistotniejsze okazały się wskazania oscylatora stochastycznego. Usunięcie tej zmiennej powoduje spadek trafności o 0,12 - 0,28. Jednocześnie dla tej zmiennej odnotowano też największy rozrzut danych, o czym świadczą długości pudełek i wąsów na wykresach. Mniej istotnymi zmiennymi okazały się CCI, MACD oraz RSI. Ostatnia z wymienionych co prawda była najistotniejsza dla SVM, ale dla pozostałych modeli permutacyjna ocena istotności wskazuje na spadek trafności o maksymalnie 0,15 w jej przypadku. Jeśli chodzi o CCI oraz MACD to okazały się one jednymi z najistotniejszych zmiennych w każdym z modeli notując spadek trafności nawet o 0,15, po wykonaniu permutacji na kolumnach z wartościami dla tych zmiennych. Warto odnotować, że binarne zmienne wskazujące na przekroczenie granic wstęgi Bollingera w niskim stopniu wpływały na zdolności predykcyjne badanych modeli. Wyjątkiem jest regresja logistyczna, dla której permutacje wykonane na tej zmiennej powodowały spadek trafności nawet o 0,05. Podobne wnioski nasuwają się po analizie spadków wartości dla zmiennych EMA oraz ADX. Ciekawy jest jednak przypadek modelu bagging. Po wykonaniu permutacji dla tych dwóch zmiennych wskaźnik trafności średnio ulegał wręcz nieznacznej poprawie. Poniżej zaprezentowano przykładowy wykres permutacyjnej istotności zmiennych dla modelu regresji logistycznej. Analizę przeprowadzono na zbiorze testowym dla danych indeksu S&P 500.

![Alt text](https://github.com/Lukkud/Praca_magisterska/blob/main/src/Logistic_regression_permutation_importance_spx.png)

Całą analizę i wyniki przedstawiono w pracy magisterskiej dostępnej pod poniższym adresem

[Praca magisterska](https://github.com/Lukkud/Praca_magisterska/blob/main/src/Praca_Łukasz_Chuchra.pdf)

## Status
Projekt zakończony

## Żródła
Główne źródła:
* Hastie, Trevor and Tibshirani, Robert and Friedman, Jerome;  <i>The Elements of Statistical Learning</i>; Springer New York Inc. (2009) [https://doi.org/10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
* James, Gareth and Witten, Daniela and Hastie, Trevor and Tibshirani, Robert; <i>An Introduction to Statistical Learning: With Applications in R</i>; Springer Publishing Company, Incorporated (2014)
* Geron, Aurelien;   <i>Uczenie maszynowe z uzyciem Scikit-Learn i TensorFlow wyd. II</i>; Helion (2019)

Pełna lista źródeł zawarta w bibliografii pracy magisterskiej
