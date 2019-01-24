1. Opis
Pliki Ÿród³owe clustering_experiments.R i classification_experiments.R zawieraj¹ kod eksperymentów przeprowadzonych 
z u¿yciem danych wyeksportowanych w reprezentacjach wektorowych.

2. Struktura podkatalogów
Zbiory danych znajduj¹ siê w folderze datasets. Szczegó³owy opis zbiorów danych, zamieszczony zosta³ w tekœcie pracy in¿ynierskiej
w podrozdziale 4.2.
Folder charts zawiera hierarchiê folderów, w których zapisywane bêd¹ wykresy z procesu wyznaczania optymalnej liczby podgrup.

3. Uruchomienie
Zbiory danych zosta³y spakowane w archiwum .zip ze wzglêdu na ograniczon¹ liczbê pamiêci na p³ycie CD.
W celu uruchomienia eksperymentów, nale¿y uprzednio wypakowaæ zbiory danych w lokalizacji datasets oraz zainstalowaæ wymagane paczki.

Eksperymenty mo¿na uruchomiæ w œrodowisku R poleceniami:
source("clustering_experiments.R")
source("classification_experiments.R")