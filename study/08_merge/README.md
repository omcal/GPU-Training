# Parallel merge

22 Ekim · PMPP Ch12 / Chris Ch13 seçili

## Kendi çözümün

İki sıralı diziyi birleştir; co-rank/partitioning fikrini çiz. Eşit elemanlar için tie kuralını ve iş paylaşımını tanımla.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../README.md).

## Tamamlanma ölçütü

Boş/küçük/eşit değerli/nonmultiple diziler CPU merge ile tam eşleşir. Chris Ch13 PyTorch gate opsiyoneldir.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
