Na podstawie porównania **Neural SA** (Correia et al., AISTATS 2023) i **RL-Based SA** (Qiu & Liang, AAMAS 2025), RL-Based SA wprowadza następujące **konkretne rozszerzenia/ulepszenia**:

---

### 1) Bogatszy stan: dodanie zmiany energii ΔE

W Neural SA stan ma postać *(x, ψ, T)*.
RL-Based SA rozszerza go do *(x, ψ, ΔE, T)*, gdzie **ΔE = E(x′) − E(x)** z poprzedniego kroku.

**Efekt:** agent dostaje jawnie informację o tym, czy ostatni ruch był dobry czy zły (lokalna „pochodna” krajobrazu energii), co ułatwia uczenie polityki i poprawia równowagę eksploracja/eksploatacja. 

---

### 2) Zastąpienie MLP przez LSTM (model sekwencyjny)

Neural SA:

* polityka oparta o **MLP**,
* decyzja zależy tylko od bieżącego stanu.

RL-Based SA:

* używa **LSTM** w aktorze i krytyku (PPO),
* przetwarza **całą sekwencję stanów z rollout’u SA** (szereg czasowy).

**Efekt:** model uczy się zależności długoterminowych (jak wcześniejsze decyzje i temperatura wpływają na przyszłe zyski), co poprawia jakość propozycji sąsiedztwa, zwłaszcza w trudnych krajobrazach energii. 

---

### 3) Lepsza generalizacja do problemów ciągłych

Neural SA:

* bardzo dobre wyniki w problemach dyskretnych (TSP, knapsack, bin packing),
* **słabsza ogólność dla problemów ciągłych**.

RL-Based SA:

* trenowany i testowany również na funkcjach ciągłych (Rosenbrock, Ackley, Eggholder),
* pokazuje **transfer między różnymi funkcjami ciągłymi** przy tej samej architekturze.



---

### 4) Wyższa jakość rozwiązań (empirycznie)

Autorzy RL-Based SA raportują:

* lepsze wyniki niż:

  * vanilla SA,
  * Adaptive SA,
  * oryginalne Neural SA,
* oraz **wyniki zbliżone do solverów komercyjnych** pod względem jakości i czasu.



---

### 5) Co pozostaje takie samo jak w Neural SA

Obie metody:

* uczą **rozkład propozycji sąsiedztwa** jako politykę RL,
* zachowują krok Metropolis–Hastings → **teoretyczna zbieżność SA pozostaje gwarantowana**,
* używają PPO / ES jako metod uczenia.

(Neural SA – opis bazowy) 

---

## Krótkie podsumowanie w 1 zdaniu

**RL-Based SA = Neural SA + (ΔE w stanie) + LSTM zamiast MLP + sekwencyjne uczenie z całych rolloutów → lepsza skuteczność i znacznie lepsza ogólność, zwłaszcza dla problemów ciągłych.**

Jeśli chcesz, mogę też zrobić tabelę porównawczą Neural SA vs RL-Based SA albo schemat architektur.
