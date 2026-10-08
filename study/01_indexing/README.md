# 2D indexing

8–11 Ekim · PMPP Ch3–4 / Chris Ch6

## Kendi çözümün

1001×777 float matrix üzerinde seçtiğin basit elementwise dönüşümü yaz. Block 16×16; width=1001, height=777. Row-major index, ceil division ve bounds kontrolünü kağıtta çıkar.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../README.md).

## Tamamlanma ölçütü

1×1, 31×33, 1001×777 için CPU referansı PASS. x/y ve row/col ilişkisini kitapsız açıkla; kodu ≤20 dakikada yeniden kur.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
