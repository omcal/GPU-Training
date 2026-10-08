# Stencil

16 Ekim · PMPP Ch8 / Chris Ch9

## Kendi çözümün

1D ya da küçük 2D stencil seç. Sınır semantiğini belirle; naive ve shared-memory halo tasarımını karşılaştır. Trafik ve arithmetic intensity tahmini yap.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../README.md).

## Tamamlanma ölçütü

CPU referansı PASS. Tiling faydasını ölçümle değerlendir; Tensor Core gibi modern mimari özelliklerini şart koşma.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
