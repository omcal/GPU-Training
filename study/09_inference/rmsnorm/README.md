# RMSNorm

27 / 28 Ekim · PMPP Ch5–10 / Chris Ch9

## Kendi çözümün

y_i = x_i / sqrt(mean(x²)+epsilon) * gamma_i. Önce sum-of-squares + normalization ayrı kernel, sonra fusion adayı. Epsilon ve gamma semantiğini belirle.

Önce input/output, grid/block, intermediate state, synchronization ve memory access
pseudocode'unu yaz. İlk uygulaman `main.cu`; hazır çözüm bu dizine eklenmemiştir.
Derleme ve ortak yardımcılar: [çalışma rehberi](../../README.md).

## Tamamlanma ölçütü

D=255/256/257/511/512/513, batch=1/4/8 PASS. Sıfır ve karışık işaretli girdi, epsilon>0, farklı gamma; memory traffic ve tolerans açıklanmış.

Doğruluk → aynı protokolle timing → Hypothesis / Experiment / Result / Why / Next.
Ölçüm olmadan performans mekanizmasını kesinleşmiş sayma.
