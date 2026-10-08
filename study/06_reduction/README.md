# Sum reduction

20 / 24 / 26 Ekim · PMPP Ch10 / Chris Ch8–9–11

## Kendi çözümün

Naive baseline → shared/block reduction → partial sums. Her reduction seviyesinin girdisini ve senkronizasyon kapsamını tasarla.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../README.md).

## Tamamlanma ölçütü

N=31/32/33/1000/1003 ve büyük, çok bloklu girdi PASS. Operasyona uygun atol/rtol; kitapsız boundary-safe yeniden yazım.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
