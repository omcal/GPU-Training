# CUDA Graph replay

30 Ekim · PMPP Ch18 / Chris Ch12

## Kendi çözümün

Sabit A→B→C kernel zinciri: normal launches vs capture/instantiate/replay. Sabit buffer lifetime ve capture şartlarını kontrol et.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../../README.md).

## Tamamlanma ölçütü

Eager/graph CPU referansı PASS. Capture, instantiate, replay ayrı süreler; aynı workload/metrik. Ortam gate geçmezse nedenini kaydet; break-even yalnız eager>replay ise hesaplanır.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
