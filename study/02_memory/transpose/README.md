# Transpose ve memory access

13–14 Ekim · PMPP Ch5–6 / Chris Ch7–8

## Kendi çözümün

Önce naive transpose, sonra shared-memory tiled transpose. Coalesced/strided erişimleri çiz; tile padding ve 64/128/256/512 thread seçenekleri için hipotez yaz.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../../README.md).

## Tamamlanma ölçütü

Kare ve dikdörtgen, nonmultiple boyutlar CPU referansı PASS. Memory-access farkını ve ölçümdeki en az iki olası mekanizmayı açıkla.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
