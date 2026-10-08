# Stable row softmax

26 / 28 Ekim · PMPP Ch10 / Chris Ch9

## Kendi çözümün

CPU: max → exp(x−max) → sum → normalize. Ayrı GPU kernel baseline → block-oriented versiyon → ölçüme dayalı fusion. Read/write/launch/barrier sayısını çıkar.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../../README.md).

## Tamamlanma ölçütü

D=127/128/129/511/512/513, batch=1/4/8 PASS. Büyük pozitif/negatif sonlu girdilerle stabilite; satır toplamı ≈1; atol/rtol gerekçeli.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
