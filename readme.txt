1. Opis
Pliki �r�d�owe clustering_experiments.R i classification_experiments.R zawieraj� kod eksperyment�w przeprowadzonych 
z u�yciem danych wyeksportowanych w reprezentacjach wektorowych.

2. Struktura podkatalog�w
Zbiory danych znajduj� si� w folderze datasets. Szczeg�owy opis zbior�w danych, zamieszczony zosta� w tek�cie pracy in�ynierskiej
w podrozdziale 4.2.
Folder charts zawiera hierarchi� folder�w, w kt�rych zapisywane b�d� wykresy z procesu wyznaczania optymalnej liczby podgrup.

3. Uruchomienie
Zbiory danych zosta�y spakowane w archiwum .zip ze wzgl�du na ograniczon� liczb� pami�ci na p�ycie CD.
W celu uruchomienia eksperyment�w, nale�y uprzednio wypakowa� zbiory danych w lokalizacji datasets oraz zainstalowa� wymagane paczki.

Eksperymenty mo�na uruchomi� w �rodowisku R poleceniami:
source("clustering_experiments.R")
source("classification_experiments.R")