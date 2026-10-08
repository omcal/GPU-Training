# CUDA bağımsız çalışma alanı

[Proje özeti](../README.md) · [Doküman dizini](../docs/README.md)

8 Ekim–1 Kasım 2026 programı: [tam plan](../docs/08-cuda-inference-study-plan.md).
[İlerleme çizelgesi](progress.md) öğrenme durumunu izler; derleme/test başarısı bir
chapter'ın öğrenildiği anlamına gelmez. Tüm oturumlar başlangıçta bekliyor.

## İlk oturum — 8 Ekim

1. İlk 15 dakika kitap kapalı: 2D indexing, ceil division, warp/block/SM ilişkisini anlat.
2. Gate geçmezse 45–60 dakika PMPP Ch3/Ch4 eksiklerini kapat. Geçerse PMPP Ch3 doğrulama + Chris Ch6.
3. [Indexing egzersizini](01_indexing/README.md) boş dosyadan yaz: `1001×777`, `16×16` block, CPU referansı.
4. [Bugünün notunu](notes/daily/2026-10-08.md) beş satırla doldur.
5. Çıktının yolu ve doğrulama sonucu ile ilerleme çizelgesini güncelle.

## Uygulama ve referans eşleştirmesi

| Kendi egzersizin | Mevcut referans (ilk denemeden sonra aç) | Fark / hedef |
|---|---|---|
| [Indexing](01_indexing/README.md) | [lab01_vector_add.cu](../cuda/lab01_vector_add.cu) | Bağımsız 2D matrix kernel |
| [Transpose](02_memory/transpose/README.md) | [lab02_coalescing.cu](../cuda/lab02_coalescing.cu) | Naive → tiled; erişimleri çiz |
| [Convolution](03_convolution/README.md) | [lab05_convolution.cu](../cuda/lab05_convolution.cu) | Bu ödev 1D convolution; referans 2D cross-correlation |
| [Stencil](04_stencil/README.md) | [lab05_convolution.cu](../cuda/lab05_convolution.cu) | Bağımsız halo/sınır tasarımı |
| [Histogram](05_histogram/README.md) | Henüz yok | Atomics → privatization |
| [Reduction](06_reduction/README.md) | [lab03_reduction.cu](../cuda/lab03_reduction.cu) | Kitapsız yeniden yaz |
| [Scan](07_scan/README.md) | [lab06_scan.cu](../cuda/lab06_scan.cu) | Ödev inclusive; mevcut lab exclusive |
| [Merge](08_merge/README.md) | Henüz yok | Paralel partitioning |
| [Softmax](09_inference/softmax/README.md) | Henüz yok | CPU → ayrı kernel → block reduction → fusion |
| [RMSNorm](09_inference/rmsnorm/README.md) | Henüz yok | Sum-of-squares → normalize → fusion |
| [Streams](10_runtime/streams/README.md) | Henüz yok | İki bağımsız pipeline |
| [Graphs](10_runtime/graphs/README.md) | Henüz yok | Eager vs replay; setup ayrı |

## Derleme ve çalıştırma

Mac üzerinde CUDA çalışmaz. Mevcut lab'ların doğruluk kontrolü:

```bash
ssh cuda 'cd ~/GPU-Training && make -C cuda test'
```

Kendi egzersizini CUDA makinesinde, proje kökünden derle.
Aşağıdaki komut `study/01_indexing/main.cu` dosyasını **sen yazdıktan sonra** çalışır:

```bash
mkdir -p build/study
/usr/local/cuda-12.9/bin/nvcc -O2 -lineinfo -std=c++17 -arch=sm_61 \
  -I cuda/include study/01_indexing/main.cu -o build/study/01_indexing
./build/study/01_indexing
```

`#include "common.cuh"` ile mevcut `CUDA_CHECK`, `check`, `input` ve `benchmark`
yardımcılarını kullanabilirsin. Kernel launch hatalarını ve asynchronous hataları
kontrol et. CPU referansı PASS vermeden performans sonucu kaydetme.
Hazır referans kodları, ilk bağımsız denemeden sonra karşılaştırma içindir.

## Günlük yöntem ve kanıt

Kapalı kitap hatırlama → PMPP decomposition → Chris bottleneck hipotezi → kendi
pseudocode/launch/kodun → CPU referansı → ölçüm → beş satırlık günlük not.
İlk compiler/runtime hatasını 10–15 dakika kendin incele; AI'den çözüm yerine ipucu iste.
Hafta sonu yeni chapter yüklemek yerine kernel'i yeniden yaz ve haftalık notu doldur.

- [Günlük şablon](notes/daily/TEMPLATE.md): yalnız Hypothesis / Experiment / Result / Why / Next.
- [Haftalık şablon](notes/weekly/TEMPLATE.md): anladıkların, yazabildiklerin, eksikler ve ölçüm yorumu.
- [Benchmark CSV](benchmarks/results/results.csv): başlangıçta yalnız başlık; ölçülmemiş sayı yok.
- [Final değerlendirmesi](notes/final-competency.md): 1 Kasım'da bağımsız yeterlilik kontrolü.

CSV süreleri `us`; bilinmeyen alanı boş bırak. `estimated_bytes` tahmini trafiği,
`effective_gbps` yararlı veri üzerinden hesaplanan hızı belirtir. Event ölçümünün
host launch boşluklarından etkilenebileceğini not et. Graph setup/capture/instantiate
host sürelerini CPU monotonic clock ile, replay GPU süresini events ile ölç;
aynı metriği karşılaştır. Eager > replay ise break-even = setup / (eager − replay).

## Mac ↔ CUDA çalışma düzeni

```bash
bash tools/sync_cuda.sh --dry-run
bash tools/sync_cuda.sh
```

Script Mac → CUDA yönünde aynı adlı dosyaları günceller. Ubuntu'da yazdığın kernel,
not ve sonuçları sonraki aktarım öncesinde Mac'e al:

```bash
rsync -azv --update cuda:GPU-Training/study/ study/
```

İki tarafta aynı dosyayı düzenlediysen önce karşılaştırıp birleştir;
`--update` yalnız daha yeni yerel dosyanın üzerine yazmayı önler, merge yapmaz.
İlk kuruluma ait seçili dosya aktarımı mevcut uzak dosyaları yedekler.
CUDA 12.9 / sm_61 ortamını koru; ileri mimari özellikleri bu ödevlerin şartı yapma.
