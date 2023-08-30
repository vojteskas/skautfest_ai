# SkautFest AI

Klasifikátor rozpoznávající pohlaví řečníka na nahrávce, nezávisle na jazyku.

## Přednáška o umělé inteligenci na SkautFestu 2023

Obsahuje kód použitý při přednáše o umělé inteligenci pro zpracování zvukových 
nahrávek, extrakci příznaků, trénování klasifikátorů a samotnou klasifikaci 
nahrávek.

Obsah repozitáře:
- `src` obsahuje zdrojové kódy v Pythonu
  - `skautfestlib.py` je knihovna funkcí pro načtení a zpracování nahrávek, 
  extrakci MFCC příznaků z nich a nakonec samotné trénování dvou klasifikátorů - *Gausovského/Bayesovského* (Naive Bayes Classifer) a *Support Vector Machines*
  - `train.py` obsahuje funkci pro natrénování zmíněných dvou klasifikátorů.
  Aktuálně obsahuje pevné cesty k trénovacímu datasetu. Spuštěním tohoto souboru
  vzniknou tři nové soubory:
    - `bayes_classifier.pk` - natrénováný gausovský klasifikátor
    - `svm_classifier.pk` - natrénovaný SVM klasifikátor
    - `lda.pk` - objekt LinearDiscriminantAnalysis použitý pro redukci dimenzí
    dat a transformaci při trénování. Pomocí něj je třeba data transformovat před
    klasifikací (viz `classifier.py` řádek `22`).
  - `classifier.py` slouží ke spouštění klasifikace na natrénovaných 
  klasifikátorech.

## Použití

1. Nejprve je potřeba klasifikátory natrénovat na datasetu

```python train.py```

2. Pak je možné klasifikovat nahrávky pomocí

```python classifier.py /path/to/your/wav/file.wav```

Je možné klasifikovat více souborů najednou:

```python classifier.py /path/file1.wav /path/file2.wav /path/file3.wav```

nebo přímo celé složky:

```python classifier.py /path/dir/with/wav/files/```

případně oba přístupy kombinovat:

```python classifier.py /dir/wav/files/1/ /path/file1.wav /dir/wav/files/2/```