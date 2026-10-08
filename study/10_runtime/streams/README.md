# Streams ve overlap

29 Ekim · PMPP Ch7 tekrar / Chris Ch11

## Kendi çözümün

İki bağımsız buffer/pipeline; stream/event bağımlılıklarını çiz. Transfer overlap deniyorsan pinned host memory ve donanım yeteneğini kontrol et.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../../README.md).

## Tamamlanma ölçütü

Çıktılar CPU referansı PASS. Synchronization placement açıklanmış. Overlap yoksa sonucu yok olarak yaz; her kernel sonunda device sync ekleme.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
