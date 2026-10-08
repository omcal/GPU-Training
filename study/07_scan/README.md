# Inclusive scan

21 Ekim · PMPP Ch11 / Chris Ch12

## Kendi çözümün

Önce küçük inclusive scan yaz; sonra block sums/offset zincirini tasarla. Mevcut lab exclusive scan: inclusive/exclusive dönüşümünü kendin açıkla.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../README.md).

## Tamamlanma ölçütü

N=1/31/32/33/1003 ve birden fazla blok PASS. Dependency ve blocklar arası koordinasyonu anlat.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
