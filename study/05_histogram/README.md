# Histogram ve atomics

19 Ekim · PMPP Ch9 / Chris Ch8

## Kendi çözümün

Global atomic baseline yaz. Sonra block-private histogram ve merge fikrini tasarla. Uniform ve tek bin üzerinde yoğunlaşan inputları karşılaştır.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../README.md).

## Tamamlanma ölçütü

Bin sınırları doğru; toplam histogram sayısı N; CPU binleriyle tam eşleşme. Contention/privatization mekanizmasını açıkla.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
