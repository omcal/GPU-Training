# 1D convolution

15 Ekim · PMPP Ch7 / Chris Ch7

## Kendi çözümün

Küçük 1D convolution: naive → tiled. Filtreyi ters çevirip çevirmediğini tanımla; CPU/GPU aynı semantiği kullansın. Zero padding ve shared-memory halo tasarla.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../README.md).

## Tamamlanma ölçütü

Küçük ve nonmultiple N; farklı tek sayılı filtre uzunlukları PASS. Edge handling ve veri tekrar kullanımını anlat.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
