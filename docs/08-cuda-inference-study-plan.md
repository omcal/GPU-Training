# CUDA inference çalışma programı — 8 Ekim–1 Kasım 2026

Bu program PMPP 5. baskı ile Chris Fregly'nin *AI Systems Performance Engineering*
kitabını tek bir uygulama zincirine bağlar:
**CUDA execution model → memory → optimization → parallel patterns → inference kernels → profiling → launch/runtime**.

Başlangıç: [çalışma alanı](../study/README.md) · [ilerleme çizelgesi](../study/progress.md)
· [CUDA çalıştırma rehberi](../cuda/README.md).

Kaynak: 8 Ekim 2026'da kullanıcı tarafından sağlanan plan. Aşağıdaki kaynak
bağlantıları bu plandan korunmuştur; bu ekleme sırasında yayıncıların içerikleri
ve güncel araç destek matrisleri yeniden doğrulanmamıştır. Kurulu ortamın ölçülen
kaydı [CUDA doğrulama dokümanında](07-cuda-validation.md) bulunur.
7 Ekim oturumu ve hiçbir öğrenme hedefi tamamlanmış kabul edilmez.

Önce iki önemli doğrulama:

- PMPP 5. baskının resmi içindekilerinde Part I gerçekten **Ch2 Heterogeneous Data Parallel Computing, Ch3 Multidimensional Grids and Data, Ch4 Compute Architecture and Scheduling, Ch5 Memory Architecture and Data Locality, Ch6 Performance Considerations**; Part II ise Ch7 Convolution, Ch8 Stencil, Ch9 Parallel Histogram, Ch10 Reduction, Ch11 Prefix Sum (Scan), Ch12 Merge. Deep Learning Ch18. [Elsevier Shop](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1)
- Chris Fregly kitabında hedeflediğimiz sıra **Ch6 GPU Architecture/CUDA/Occupancy, Ch7 Memory Access, Ch8 Occupancy/Warp Efficiency/ILP, Ch9 Kernel Efficiency/Arithmetic Intensity, Ch11 Inter-Kernel Pipelining/Streams, Ch12 CUDA Graphs, Ch13 PyTorch Profiling, Ch16 Inference Profiling**. Yani önceki kaba eşleştirmelerdeki gibi “Ch11 = CUDA Graphs” değil; Graphs esas olarak Ch12'de. [O'Reilly Media](https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/)

Sağlanan planın chapter başlıkları ve kapsam eşleştirmesi resmi Elsevier/O'Reilly TOC'larına ve Fregly'nin yazar reposundaki chapter rehberine dayanır; tam kitapların tüm sayfaları incelenmiş kabul edilmez. Dolayısıyla aşağıda uydurma sayfa veya exercise numarası yok. [Elsevier Shop](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1)

---

# 1. İki kitap birbirini nasıl tamamlıyor?

**PMPP senin “GPU programcısı” kitabın.** Bir problemi thread/block/grid'e nasıl böldüğünü, locality'yi, shared memory'yi, reduction/scan/convolution gibi paralel yapı taşlarını nasıl düşündüğünü öğretecek.

**AI Systems Performance Engineering ise “performance engineer” kitabın.** Yazdığın kernel neden yavaş, hangi metriği ölçmelisin, occupancy neden tek başına hedef değil, memory traffic/kernel launch/runtime davranışı inference sistemini nasıl etkiliyor gibi ikinci katmanı öğretecek.

Bunun ana zinciri şu:

> **PMPP:** algoritmayı GPU'ya doğru map et
> → **Chris:** neden hızlı/yavaş olduğunu teşhis et
> → **repo:** hipotezi kodla
> → **ölç:** sayı üret
> → **yorumla:** sonucu mimariyle açıkla.

Özellikle şu eşleştirmeler kritik:

| PMPP | Chris | Ortak fikir |
|---|---|---|
| Ch3–4 | Ch6 | grid/block/warp, scheduling, occupancy temeli |
| Ch5 | Ch7 | coalescing, locality, shared memory |
| Ch6 | Ch8 | divergence, occupancy, instruction/memory bottleneck |
| Ch7–8 | Ch7/9 | tiling, locality, arithmetic intensity |
| Ch9 | Ch8 | contention, atomics |
| Ch10 | Ch8/9 | reduction, warp/block cooperation |
| Ch11 | Ch8/9 | scan ve parallel dependency yapıları |
| Ch18 | Ch9/13/16 | inference kernel'lerine geçiş |
| tekrar | Ch11 | streams, overlap |
| tekrar | Ch12 | launch overhead, CUDA Graphs |
| tekrar | Ch16 | kernel → inference-system etkisi |

PMPP Ch12 Merge'i de çalışacağız; çünkü paralel work partitioning açısından faydalı. Ancak softmax/RMSNorm hedefin için Ch10/11 kadar ağırlık vermeyeceğiz.

---

# 2. Donanım gerçeği: GTX 1050 Ti planı nasıl değiştiriyor?

1050 Ti Max-Q'n **Pascal, compute capability 6.1**. CUDA 12.9 hâlâ `compute_61` hedefini biliyor; NVIDIA'nın güncel destek matrisi Pascal için **son toolkit ailesini CUDA 12.x, son driver branch'ini R580** olarak gösteriyor. Dolayısıyla çalışan 12.9/R580 sistemini öğrenme sürecinin ortasında CUDA 13'e yükseltmeye çalışma. [NVIDIA Docs](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html?utm_source=chatgpt.com)

Profiler tarafında daha önemli bir sınırlama var:

- güncel **Nsight Compute Pascal desteklemiyor**;
- Nsight Systems **2025.4 itibarıyla Pascal desteğini kaldırdı**;
- güncel Compute Sanitizer da Pascal'ı artık desteklemiyor; desteğin kaldırılması 2025.3 sürümünde duyuruldu. [NVIDIA Docs](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/gpu-support.html?utm_source=chatgpt.com)

Bu nedenle bu programın zorunlu laboratuvarları güncel profiler veya sanitizer
kurulumuna bağlı değildir. Mevcut makinenin CUDA 12.9 ile gelen Compute Sanitizer'ı
27 Eylül'de çalıştırılmıştır; bu tarihli sonuç, güncel sürümlerin Pascal desteğiyle
karıştırılmamalıdır. Çalışan araçları yükseltmeden kullan; yeni bir araç için önce
sürüm/GPU destek kontrolü yap.

Senin ölçüm stack'in ilk etapta:

**CUDA events → CPU reference correctness → explicit CUDA error checking → architecture/resource reasoning → gerekiyorsa Pascal destekli eski Nsight Systems sürümü.**

`cudaEventElapsedTime()` CUDA 12.9 Runtime API'de doğrudan destekleniyor ve yaklaşık 0.5 µs resolution belirtiliyor. [NVIDIA Docs](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__EVENT.html?utm_source=chatgpt.com)

Chris'teki TMA, Thread Block Cluster, FP8/FP4, modern Tensor Core, Blackwell/Hopper vb. kısımlar **okuma/gelecek-donanım notu** olacak. 1050 Ti ödevi değil.

Bu kötü değil. Tam tersine başlangıç için çok temiz bir durum:

> Tensor Core büyüsüne kaçmadan önce grid, memory, reduction, synchronization ve launch cost öğrenmiş olacaksın.

---

# 3. Zaman varsayımı

Sen süre vermediğin için planı şu varsayımla kuruyorum:

**Hafta içi:** yaklaşık **2–3 saat/gün**, hedef ≈ 2 saat 30 dakika.

**Hafta sonu:** Cumartesi ≈ 2 saat + Pazar ≈ 1.5–2 saat = **toplam 3.5–4 saat**.

7 Ekim başlangıç oturumu hariç 8 Ekim–1 Kasım arasında yaklaşık **58–60 saatlik** ciddi çalışma çıkar.

Bir chapter'ın “çalışılmış” sayılması, tüm sayfalarının altını çizmek anlamına gelmeyecek. Minimum:

> kavramı kitapsız anlat → ilgili çekirdek bölümü oku → bir problemi kendin tasarla → kodla/doğrula veya performans hipotezi oluştur → kısa sonuç notu yaz.

### Gerçekçi chapter süre tahmini

Bunlar yayıncının verdiği süre değil, **aktif çalışma + kod + geri çağırma için benim planlama tahminim**:

| Chapter grubu | Derin aktif çalışma |
|---|---:|
| PMPP 2–4 | 1–1.5 saat/chapter |
| PMPP 5–6 | 1.5–2.5 saat/chapter |
| PMPP 7–9 | 1.5–2 saat/chapter |
| PMPP 10 Reduction | 2–3 saat |
| PMPP 11 Scan | 2.5–3.5 saat |
| PMPP 12 Merge | 1.5–2 saat |
| PMPP 18 Deep Learning | 2–3+ saat seçili bölümler |
| Chris 6–7 | 2–3 saat/chapter tam aktif çalışma |
| **Chris 8** | **4–6+ saat — ağır** |
| **Chris 9** | **4–6+ saat — ağır** |
| Chris 11–12 | 2–4 saat/chapter |
| Chris 13 | 4+ saat; yalnız ilgili parçalar |
| Chris 16 | 4+ saat; yalnız ilgili parçalar |

Dolayısıyla özellikle **Chris Ch8 ve Ch9'u bir akşamda “bitirdim” demeyeceğiz.** Takvimde tekrar görmelerinin nedeni bu.

---

# 4. 7 Ekim 2026 — başlangıç oturumu

Bu oturumun yapılmış olduğunu **varsaymıyorum**.

### Hedef

**PMPP Ch3 – Multidimensional Grids and Data**

ardından

**PMPP Ch4 – Compute Architecture and Scheduling**

ve zaman kalırsa:

**PMPP Ch5 – Memory Architecture and Data Locality'e giriş.**

Yaklaşık 3 saatlik agresif oturum:

**0:00–0:15 — kitapsız**

Kağıda:

```text
threadIdx
blockIdx
blockDim
gridDim
```

yaz.

1D index formülünü ve 2D row/column → linear memory index dönüşümünü çıkarmaya çalış.

**0:15–1:10 — PMPP Ch3**

Özellikle:

- 1D/2D/3D grids
- row/column mapping
- ceil division
- boundary conditions
- input dimension ≠ block dimension multiple

### Kendi egzersizim

Kitaptan bağımsız:

`1001 × 777` float matrix için kernel yaz.

Block:

```text
16 × 16
```

olsun.

CPU'da aynı sonucu hesapla.

Şunları açıklayabil:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

ve neden:

```cpp
idx = row * width + col;
```

olduğunu.

**1:10–1:20 mola**

**1:20–2:10 — PMPP Ch4**

Çekirdek:

- warp
- block
- SM
- block scheduling
- latency hiding
- occupancy kavramı

Henüz occupancy'yi “maksimum = en hızlı” diye öğrenme.

**2:10–2:40 — book exercises**

Kitaptaki mevcut Ch3/Ch4 exercise'lardan indexing, grid sizing ve execution hierarchy'yi test eden birkaçını çöz.

Exercise numarası vermiyorum çünkü tam edition exercise metnini doğrulamış değilim.

**2:40–3:00 — blank-page recall**

Kitabı kapat:

> 1000×1000 matrix nasıl launch edilir?

> warp ile block farkı ne?

> block neden bir SM'den diğerine bölünmez?

> 257 eleman, 256-thread block ise kaç block gerekir?

cevapla.

Ch5 yetişirse sadece ilk kavramsal giriş. **“Ch5 tamamlandı” işaretleme.**

---

# 5. 8 Ekim–1 Kasım: 25 günlük ana takvim

## Hafta 0 — Temeli sabitle
### 8–11 Ekim

| Tarih/gün | PMPP chapter ve odak | Chris chapter ve odak | Ortak egzersiz / çıktı | Süre | Tamamlanma ölçütü |
|---|---|---|---|---|---|
| **8 Eki Per** | **Ch3 – Multidimensional Grids and Data**. 7 Ekim'in gerçek durumunu doğrula; 2D mapping, bounds | **Ch6 – GPU Architecture, CUDA Programming, and Maximizing Occupancy**: threads/warps/blocks/grids | 2D matrix kernel; `1001×777`; CPU reference; bounds | 2h30: P55/C55/lab40 | Kod sıfırdan yazılıyor; x/y/row/col açıklanıyor; PASS |
| **9 Eki Cum** | **Ch4 – Compute Architecture and Scheduling** | **Ch6 tekrar/derinleşme**: SM, scheduling, launch config, occupancy | block size 64/128/256/512; sadece ilk gözlem, benchmark yarışına girme | 2h30 | Warp/block/SM ilişkisini kitapsız anlat; sonuç tablosu çıkar |
| **10 Eki Cmt — TEKRAR** | **Ch3–4** | **Ch6** | Kitapsız grid çiz; 2D kernel'i kaynaksız yeniden yaz | ≈2h | ≤20 dk'da doğru indexing + bounds |
| **11 Eki Paz — TEKRAR** | **Ch3–4 + Ch5 ön bakış** | **Ch6** | Eksikleri kapat; 1 sayfalık “GPU execution model” notu | 1.5–2h | Kitapsız 5 dk anlatım + kod PASS |

Bu ilk hafta performance sayılarından çok **doğruluk + zihinsel model** haftası.

---

# Hafta 1 — Memory ve temel paralel pattern
## 12–18 Ekim

| Tarih/gün | PMPP chapter ve odak | Chris chapter ve odak | Ortak egzersiz / çıktı | Süre | Tamamlanma ölçütü |
|---|---|---|---|---|---|
| **12 Eki Pzt** | **Ch2 – Heterogeneous Data Parallel Computing**: host/device modeli, transfer, kernel lifecycle | **Ch6 tekrar**: CUDA programming + roofline kavramına giriş | vector add'ı tamamen sıfırdan yaz; allocation→copy→launch→copy | 2–2.5h | Her CUDA çağrısının nedenini açıklayabiliyorsun |
| **13 Eki Sal** | **Ch5 – Memory Architecture and Data Locality** | **Ch7 – Profiling and Tuning GPU Memory Access Patterns**: coalescing, shared memory, tiling | naive matrix transpose → tiled transpose. Eski kodu bakmadan yeniden yaz | 2.5–3h | CPU reference PASS; iki versiyon arasındaki memory-access farkını çizerek anlat |
| **14 Eki Çar** | **Ch6 – Performance Considerations** | **⚠ Ch8 – Occupancy Tuning, Warp Efficiency, and ILP. ÇEKİRDEK:** divergence, occupancy, stall mantığı. **Genişletme:** profiler counter'ları | coalesced vs deliberately-strided kernel; block-size sweep | 2.5–3h | “neden yavaşladı?” için en az iki mekanizma hipotezi |
| **15 Eki Per** | **Ch7 – Convolution** | **Ch7 tekrar**: tiling/data reuse/shared memory | **Kendi exercise:** küçük 1D convolution; naive → tiled | 2.5–3h | edge handling + CPU reference + nonmultiple N |
| **16 Eki Cum** | **Ch8 – Stencil** | **⚠ Ch9 – Increasing CUDA Kernel Efficiency and Arithmetic Intensity. ÇEKİRDEK:** reuse, fusion, arithmetic intensity. **Genişletme:** Tensor Core/CUTLASS sadece teori | 1D stencil veya küçük 2D stencil; shared-memory halo düşüncesi | 2.5–3h | naive/tiled tasarımı açıklayabiliyor; modern-GPU-only özellikleri ayırabiliyorsun |
| **17 Eki Cmt — TEKRAR** | **Ch2,5,6,7,8** | **Ch6,7,8,9** | Kitapsız: coalescing/shared memory/divergence/occupancy anlat. Transpose veya convolution kernel'ini sıfırdan yaz | ≈2h | Notlara bakmadan derlenebilir temel kernel |
| **18 Eki Paz — TEKRAR** | **Ch5–8 eksik kapatma** | **Ch7–9 çekirdek tekrar** | Haftanın timing sonuçlarını yorumla; yanlış hipotezleri düzelt | 1.5–2h | `week1.md`: “ölçüm → sebep → sonraki deney” tablosu |

Burada Ch8 ve Ch9 **derinlemesine bitmiş sayılmıyor**. İleride tekrar ediyoruz.

---

# Hafta 2 — Atomics → reduction → scan → inference bağlantısı
## 19–25 Ekim

| Tarih/gün | PMPP chapter ve odak | Chris chapter ve odak | Ortak egzersiz / çıktı | Süre | Tamamlanma ölçütü |
|---|---|---|---|---|---|
| **19 Eki Pzt** | **Ch9 – Parallel Histogram** | **Ch8 tekrar:** contention, warp behavior, occupancy | histogram: global atomic baseline; privatization mantığını incele | 2.5h | Atomics'in niçin bottleneck olabileceğini açıklayabil |
| **20 Eki Sal** | **Ch10 – Reduction** | **Ch11 – Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocations** | sum reduction: naive → shared/block reduction. Stream/Event kavramını ekle | 2.5–3h | N=31/32/33/1000/1003 PASS |
| **21 Eki Çar** | **Ch11 – Prefix Sum (Scan)** | **Ch12 – Dynamic Scheduling, CUDA Graphs, and Device-Initiated Kernel Orchestration**: **yalnız CUDA Graph temel modeli** | küçük inclusive scan; ardından 2–3 basit kernel zincirinin graph fikrini çiz | 2.5–3h | Scan dependency modelini ve graph capture/replay farkını anlat |
| **22 Eki Per** | **Ch12 – Merge** | **Ch13 – Profiling, Tuning, and Scaling PyTorch**: yalnız profiler/NVTX/kernel launch tarafı | merge partitioning; PyTorch CUDA çalışıyorsa basit op profile gate; çalışmıyorsa CUDA C++ ile devam | 2–2.5h | PyTorch'u prerequisite yapmadan concept aktarımı yapılmış |
| **23 Eki Cum** | **Ch18 – Deep Learning**: inference'la ilgili GPU hesap yapıları | **⚠ Ch16 – Profiling, Debugging, and Tuning Inference at Scale. ÇEKİRDEK:** latency/throughput, batching, bottleneck workflow. **Genişletme:** cluster/distributed | basit `Linear → activation` pipeline'ını operasyonlar olarak parçala; kernel sınırlarını çiz | 2.5–3h | kernel latency ≠ request latency ayrımını açıkça anlat |
| **24 Eki Cmt — TEKRAR** | **Ch9–12 + Ch18** | **Ch8,11,12,13,16** | Reduction'ı kitapsız yeniden yaz. Scan algoritmasını tahtada anlat | ≈2h | Reduction doğru + boundary-safe |
| **25 Eki Paz — TEKRAR** | **özellikle Ch10–11** | **Ch11–12–16** | Haftalık sonuç yorumu; softmax/RMSNorm için tasarım notu | 1.5–2h | Softmax'ın hangi reduction'lara ihtiyaç duyduğunu çıkar |

Bu hafta sonunda önemli bir zihinsel geçiş olacak:

```text
reduction
    ↓
max(x)
    ↓
exp(x - max)
    ↓
sum(exp)
    ↓
normalize
```

Yani softmax artık “ML sihri” değil, paralel primitives bileşimi olmaya başlayacak.

---

# Hafta 3 — inference kernel haftası
## 26 Ekim–1 Kasım

| Tarih/gün | PMPP chapter ve odak | Chris chapter ve odak | Ortak egzersiz / çıktı | Süre | Tamamlanma ölçütü |
|---|---|---|---|---|---|
| **26 Eki Pzt** | **Ch10 Reduction tekrar** | **Ch9 tekrar – Kernel Efficiency/Fusion** | **Kendi exercise: Softmax v0/v1.** CPU reference → multi-kernel GPU baseline → block reduction versiyonu | 2.5–3h | D=127/128/129/511/512/513; numerical tolerance PASS |
| **27 Eki Sal** | **Ch5 Memory Architecture tekrar** | **Ch9 tekrar – arithmetic intensity + fusion** | **Kendi exercise: RMSNorm.** sum-of-squares reduction + normalize; global traffic'i hesapla | 2.5–3h | CPU reference; epsilon/tolerance; memory-read/write reasoning |
| **28 Eki Çar** | **Ch6 Performance tekrar** | **⚠ Ch8 tekrar – occupancy/divergence/ILP** | Softmax/RMSNorm block-size sweep; 64/128/256/512; event timing | 2.5–3h | En hızlı config'i seçmekten öte nedenini açıklamaya çalış |
| **29 Eki Per** | **Ch7 Convolution tekrar — reuse/fusion transferi** | **Ch11 tekrar – streams/events** | iki bağımsız buffer/pipeline üzerinde streams deneyi; overlap varsa ölç, yoksa “yok” de | 2–2.5h | synchronization placement'ini açıklayabiliyorsun |
| **30 Eki Cum** | **Ch18 Deep Learning tekrar** | **Ch12 tekrar – CUDA Graphs** | küçük sabit kernel chain: normal launches vs graph replay; **Graph environment'ta çalışıyorsa** capture/setup + replay ayrı ölç | 2.5–3h | capture/instantiate cost ≠ replay cost ayrılmış |
| **31 Eki Cmt — FINAL REWRITE** | **Ch3,5,6,10,11,18** | **Ch6–9,11–12,16** | Not kapalı: indexing kernel + reduction + RMSNorm veya softmax'ın çekirdeğini yeniden yaz | ≈2h | Kodun her satırını açıklayabil; CPU reference PASS |
| **1 Kas Paz — FINAL YETERLİLİK** | **PMPP 2–12 + 18 seçili tekrar** | **Chris 6–9,11–13,16 seçili tekrar** | aşağıdaki final değerlendirmesi | ≈2h | “orta seviyeye geçiş” checklist'i; eksiklerin açık listesi |

---

# 6. Günlük çalışma protokolü

Her hafta içi gününde mümkün olduğunca aynı formatı kullan.

### A. 10–15 dk — kapalı kitap retrieval

Dünkü konudan üç soru:

```text
Bu mekanizma ne yapıyor?
Neden gerekiyor?
Yanlış kullanırsam ne olur?
```

Kod varsa 5–10 satırlık kritik kısmını hafızadan yaz.

### B. 45–60 dk — PMPP

Amaç chapter özeti çıkarmak değil.

Üç şeyi çıkar:

```text
problem
→ parallel decomposition
→ GPU constraint
```

Örneğin reduction için:

```text
N eleman
→ N işi paralel başlat
→ giderek azalt
→ synchronization + divergence + memory traffic problemi
```

### C. 45–60 dk — Chris

Sorun şu forma dönüşmeli:

```text
PMPP bana algoritmayı verdi.

Chris:
"Bu algoritmanın bottleneck'i nerede olabilir?"
```

### D. 35–60 dk — kendi çözümün

AI'ye:

> “RMSNorm kernel yaz.”

deme.

Önce sen:

```text
input
output
grid
block
intermediate state
synchronization
memory access
reference implementation
```

tasarla.

Sonra kodla.

### E. 10–15 dk — sonuç notu

Her gün yalnızca:

```text
Hypothesis:
Experiment:
Result:
Why:
Next:
```

Beş satır.

Bu gelecekte makale yazmayı inanılmaz kolaylaştırır.

---

# 7. AI kullanım kuralı

Senin “AI'nin yazdığı kodlara yaslanıyorum” endişen burada özellikle çözülmeli.

Bir CUDA lab'ında doğru sıra:

```text
1. Problemi ben açıklarım.
2. Pseudocode'u ben çıkarırım.
3. Kernel launch configuration'ı ben seçerim.
4. İlk implementasyonu ben yazarım.
5. Compiler/runtime hatasını kendim 10-15 dk incelerim.
6. Sonra AI kullanabilirim.
```

AI şu görevlerde gayet iyi:

```text
"Bu race'in kaynağını bulmam için ipucu ver."
"Bu memory-access pattern'i değerlendir."
"Şu ölçüm sonucuyla hipotezim çelişiyor mu?"
"Kernel'i yazma; bana üç soru sorarak bug'ı buldur."
```

AI'nin verdiği düzeltmeyi koduna almadan önce:

> “Bu satırın correctness veya performance açısından tam olarak ne değiştirdiğini açıklayabiliyor muyum?”

sorusunun cevabı **evet** olmalı.

---

# 8. Doğruluk protokolü

Performance engineering'de hızlı ama yanlış kernel = **başarısız kernel**.

Her kernelin yanında CPU reference bulunsun.

Örneğin:

```text
vector add:
N = 1, 31, 32, 33, 255, 256, 257, 1003

reduction:
N = 31, 32, 33, 1000, 1003

softmax:
D = 127, 128, 129, 511, 512, 513
batch = 1, 4, 8

RMSNorm:
D = 255, 256, 257, 511, 512, 513
```

Bunlar benim önerdiğim testler; kitabın exercise'ları değil.

Float sonuçlarında:

```text
gpu == cpu
```

bekleme.

Kullan:

```text
abs_error <= atol + rtol * abs(reference)
```

Örneğin başlangıçta `1e-5` / `1e-4` gibi toleransları deneyebilirsin ama **operasyona göre doğrula**; bunu evrensel doğruluk sınırı olarak ezberleme.

Ayrıca şu üç sınıf mutlaka test edilmeli:

```text
small
awkward/nonmultiple
large
```

---

# 9. Benchmark protokolü

İlk hafta bunu abartma.

Ama Ch6 sonrasında standartlaştır.

Ölçümden önce:

```text
Ollama/local LLM stop
GPU idle
gereksiz process yok
```

Laptop olduğu için termal durum da önemli.

Mümkünse sonuçla beraber:

```text
GPU temperature
SM clock
memory clock
```

kaydet.

### Kernel microbenchmark

Örneğin:

```text
warm-up: 10–20
measurement: 100+
```

Başlangıç için yeterli.

Çok kısa kernel'lerde tek launch zamanlamak yerine:

```text
start event

for 1000 launches:
    kernel()

stop event
```

yapıp toplam süreyi launch sayısına bölmek daha anlamlı olabilir.

Event'ler:

```cpp
cudaEventRecord(start);

kernel<<<...>>>();

cudaEventRecord(stop);
cudaEventSynchronize(stop);

cudaEventElapsedTime(...);
```

CUDA event API'si tam bu amaç için mevcut. [NVIDIA Docs](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__EVENT.html?utm_source=chatgpt.com)

Kernel zinciri ölçerken her kernel sonrasında:

```cpp
cudaDeviceSynchronize();
```

koyma.

Bu pipeline davranışını bozabilir.

Tek kernel latency ölçüyorsan synchronization bilinçli kullanılabilir.

---

# 10. Softmax progression

26 Ekim'de doğrudan “ultra fused SOTA softmax” yazmaya çalışma.

### v0 — CPU

```text
max
exp
sum
normalize
```

### v1 — GPU naive

Her aşama ayrı kernel olabilir.

Örneğin:

```text
max reduction
exp
sum reduction
normalize
```

Çirkin olması sorun değil.

Çünkü baseline gerekiyor.

### v2 — block-oriented

Bir row/block veya uygun mapping.

Shared memory reduction kullan.

### v3 — optimize

Şunları sor:

```text
kaç global read?
kaç global write?
kaç kernel launch?
kaç synchronization?
kaç register?
```

### v4 — fusion deneyi

Uygun safhaları birleştir.

Sonra:

```text
v1 vs v2 vs v3/v4
```

karşılaştır.

Bu tam olarak PMPP + Chris birleşimi.

---

# 11. RMSNorm progression

RMSNorm:

\[
y_i = \frac{x_i}{\sqrt{\frac{1}{N}\sum_j x_j^2 + \epsilon}}\gamma_i
\]

Aslında öğrendiğin primitive'ler:

```text
map: x²
reduce: sum
scalar operation: rsqrt
map: x * scale * gamma
```

Dolayısıyla 27 Ekim'de sıfırdan yaklaşabilirsin.

İlk versiyon:

```text
kernel A → sum(x²)
kernel B → normalization
```

Sonra sor:

> `x` global memory'den kaç defa okunuyor?

> intermediate result yazılıyor mu?

> launch sayısı nedir?

> fusion ne kazandırabilir?

İşte burada “arithmetic intensity” lafı soyut olmaktan çıkar.

---

# 12. Streams ve CUDA Graphs neden daha sonra?

CUDA Graphs'ın amacı, tekrarlanan GPU work graph'ını önceden tanımlayıp tekrar launch ederek CPU-side launch/setup overhead'ini azaltabilmek. NVIDIA da bunu CUDA Graph modelinin temel motivasyonlarından biri olarak açıklıyor. [NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html?utm_source=chatgpt.com)

Ama:

> yanlış/yavaş kernel zincirini graph'a koymak onu iyi kernel yapmaz.

O yüzden sıra:

```text
correctness
↓
memory access
↓
kernel optimization
↓
launch overhead
↓
streams/graphs
```

olacak.

30 Ekim experiment'i örneğin:

```text
A → B → C
A → B → C
A → B → C
...
```

için:

**Normal:**

```text
1000 × [launch A+B+C]
```

**Graph:**

```text
capture
instantiate

1000 × graph replay
```

ölç.

Ve ayrı ayrı kaydet:

```text
capture/setup
instantiate
replay
```

Böylece gelecekteki araştırma soruna doğrudan veri biriktirmeye başlarsın.

---

# 13. Cumartesi protokolü

Cumartesi yeni chapter sıkıştırmayacaksın.

İki görev:

### 1. Kitapsız anlatım

Yaklaşık 30–40 dakika.

Örneğin:

> coalescing nedir?

> shared memory niçin hızlı olabilir?

> bank conflict nedir?

> occupancy nedir?

> divergence nedir?

> reduction neden naive haliyle kötü?

> launch overhead nedir?

### 2. Kernel reconstruction

45–60 dakika.

Eski dosyaya bakmadan:

```text
transpose
convolution
reduction
softmax
RMSNorm
```

o haftanın bir temel kernelini yeniden yaz.

Eski implementasyona ancak önce kendi versiyonunu bitirdikten sonra bak.

---

# 14. Pazar protokolü

Pazar gününün amacı yeni şey yetiştirmek değil.

Şunu yap:

```text
20 dk – retrieval
40 dk – kaçırılan core konu
30 dk – benchmark sonuçlarını yorumla
20 dk – hatalı kernel/test düzelt
20 dk – gelecek hafta hazırlığı
```

Pazar akşamında README veya weekly note'a:

```text
What I understand
What I can implement
What I cannot implement yet
What measurement surprised me
Next week's question
```

yaz.

---

# 15. Gün kaçırırsan ne olacak?

En kötü hata:

> “Dün 2 chapter kaçırdım → bugün 4 chapter.”

yapmak.

Yapma.

Kaçırılmış chapter'ı üç seviyeye ayır:

```text
CORE
EXTENSION
LAB
```

Sonraki gün sadece **CORE** kısmını 30–40 dakikada tamamla.

Lab pazar gününe geçebilir.

Extension ise gerekirse 1 Kasım sonrasına kalabilir.

Öncelik sırası:

```text
PMPP 3–6
Chris 6–9
PMPP 10
PMPP 11
Softmax/RMSNorm
Chris 11/12
Chris 16
───────────────
PMPP 9/12'in derinleşmesi
Chris 13 extension
modern architecture sections
```

Yani fundamental konu kaybedip chapter count'u korumayacağız.

---

# 16. Tek çalışma reposu

Mevcut `GPU-Training` dizini kullanılacak; ikinci bir repo kurulmayacak.
Çalışan CUDA C++ örnekleri `cuda/`, bağımsız yazacağın egzersizler `study/` altında.

```text
GPU-Training/
├── cuda/
│   ├── Makefile
│   ├── include/common.cuh    # CUDA_CHECK, CPU karşılaştırması, input, event timing
│   └── lab01...lab06.cu      # mevcut referans uygulamaları
├── study/
│   ├── README.md
│   ├── progress.md
│   ├── 01_indexing/
│   ├── 02_memory/transpose/
│   ├── 03_convolution/
│   ├── 04_stencil/
│   ├── 05_histogram/
│   ├── 06_reduction/
│   ├── 07_scan/
│   ├── 08_merge/
│   ├── 09_inference/softmax/
│   ├── 09_inference/rmsnorm/
│   ├── 10_runtime/streams/
│   ├── 10_runtime/graphs/
│   ├── benchmarks/results/results.csv
│   ├── notes/daily/
│   ├── notes/weekly/
│   └── notes/final-competency.md
└── docs/08-cuda-inference-study-plan.md
```

Her egzersiz dizinindeki README problemi ve kabul ölçütlerini verir.
Kernel çözümü hazır değildir: önce tasarımı ve ilk kodu sen yaz.
Ortak yardımcıları `cuda/include/common.cuh` içinden kullanabilirsin;
aynı timing/error-checking altyapısını yeniden oluşturmaya gerek yok.
Mevcut event harness 5 warm-up ve 5×20 launch kullanır; aşağıdaki 10–20 warm-up /
100+ ölçüm önerisiyle aynı protokol değildir. Karşılaştırdığın varyantlarda aynı
protokolü kullanıp CSV'ye kaydet.

Çalıştırma, mevcut lab eşleştirmeleri ve SSH aktarımı için
[study/README.md](../study/README.md) dosyasını izle.

---

# 17. Haftalık somut çıktılar

### 11 Ekim

```text
2D indexing kernel
execution model note
```

### 18 Ekim

```text
transpose naive/tiled
convolution/stencil
memory/coalescing measurements
```

### 25 Ekim

```text
histogram
reduction
scan
CUDA event benchmark harness
```

### 30 Ekim

```text
softmax baseline
softmax optimized candidate
RMSNorm baseline
RMSNorm optimized candidate
streams experiment
CUDA Graph experiment
```

### 1 Kasım

```text
README
benchmark CSV/table
correctness suite
final competency report
next-stage experiment plan
```

Bu point'te repo “tutorial çözdüm” reposu olmaktan çıkıp küçük bir **performance engineering notebook in code** haline gelmiş olacak.

---

# 18. 1 Kasım final yeterlilik testi

1 Kasım'da hedef **CUDA expert/senior olmak değil.**

Hedef şu soruların çoğuna bağımsız cevap verebilmek.

### A. Execution model

Kağıda çiz:

```text
grid
block
warp
thread
SM
```

ve scheduler ilişkisini açıkla.

### B. Memory

Verilen iki kernel'e bakıp hangisinin global memory access'inin daha iyi olduğunu tahmin et.

Shared memory'nin neden işe yarayabileceğini ve neden bazen işe yaramayacağını söyle.

### C. Correctness

Şu input geldiğinde:

```text
N = 1003
blockDim = 256
```

hangi boundary handling gerektiğini yaz.

### D. Reduction

Book/AI olmadan sum reduction kernel'i yaz.

### E. Inference

Softmax'ın paralel decomposition'ını anlat.

RMSNorm'un reduction kısmını anlat.

### F. Measurement

Şunları ayır:

```text
kernel execution time
kernel launch overhead
H2D/D2H
CPU overhead
end-to-end request latency
throughput
```

### G. Performance diagnosis

Bir kernel:

```text
128 threads: 40 us
256 threads: 31 us
512 threads: 36 us
```

çıkarsa:

> “256 en hızlıdır.”

deyip bırakma.

En azından:

> occupancy/resource usage/memory behavior/instruction scheduling ölçmeden kesin mekanizma söyleyemem.

diyebil.

Bu performance engineer refleksi.

---

# 19. Runtime'a ne zaman geçilir?

Burada önemli bir sınır var.

Custom kernel şu seviyeye gelmeden vLLM benzeri runtime'a entegre etmeye çalışma:

```text
correct
+
reproducibly benchmarked
+
applicable shape range known
+
baseline comparison available
+
bottleneck hypothesis exists
```

Örneğin:

```text
RMSNorm
batch ∈ {1,4,8}
hidden ∈ {512,1024,2048}
```

için kernel yaz.

Önce kendi baseline'ınla karşılaştır.

Daha sonra uygun modern GPU/framework ortamında mevcut optimize runtime/library implementation'ıyla karşılaştır.

Çünkü:

> custom kernel'in naive CUDA kodundan %40 hızlı olması

ile

> vLLM/PyTorch/CUTLASS/cuDNN optimized path'ten %40 hızlı olması

aynı şey değil.

Hatta ikinci karşılaştırmada kaybetmen gayet normal.

---

# 20. Kernel speedup ile inference speedup'ı karıştırma

Örneğin:

```text
RMSNorm:
100 us → 50 us
```

Kernel gecikmesinde %50 azalma; hızlanma oranı 2×.

Ama model request'i:

```text
20 ms → 19.95 ms
```

oluyorsa sistem açısından neredeyse önemsiz.

Amdahl mantığıyla:

\[
T_{total}
=
T_{kernel}
+
T_{other\ kernels}
+
T_{runtime}
+
T_{memory}
+
T_{scheduler}
+\dots
\]

O yüzden gelecekte benchmark tablon iki katmanlı olacak:

| Seviye | Metric |
|---|---|
| Kernel | µs/op |
| Kernel | effective bandwidth |
| Kernel | launch count |
| Kernel | bytes read/write |
| Pipeline | µs/sequence |
| Runtime | latency |
| Runtime | throughput |
| Runtime | TTFT/TPOT uygun olduğunda |

Chris Ch16 tam olarak bu zihinsel geçiş için önemli; chapter'ın resmi kapsamı inference profiling, batching/scheduling, system optimization ve latency/throughput trade-off'larını içeriyor. [O'Reilly Media](https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/)

---

# 21. Bundan çıkabilecek mini proje / makale yönü

Senin şu anki hedeflerinle en doğal dar soru bence:

> **“Small-batch inference normalization kernels'ında memory traffic azaltımı ve kernel fusion'ın kernel latency ve launch overhead üzerindeki etkisi nedir; CUDA Graph replay kullanıldığında hangi tekrar sayısında capture/instantiation maliyeti amortize edilir?”**

İlk scope:

```text
batch = 1, 4, 8

hidden size =
512
1024
2048
4096 (VRAM/implementation uygunsa)

operation =
RMSNorm
veya küçük Softmax
```

Implementations:

```text
A: separate naive kernels
B: optimized separate kernels
C: fused kernel
D: fused + CUDA Graph
```

Ölç:

```text
correctness
kernel time
kernel count
capture time
instantiate time
graph replay time
estimated global memory traffic
break-even iteration count
```

Break-even yaklaşık:

\[
N_{break-even}
=
\frac{T_{setup}}
{T_{eager}-T_{replay}}
\]

tabii ancak:

\[
T_{eager} > T_{replay}
\]

ise anlamlı.

Bu tek başına yayın yeniliği demek değil. Ayrıca Pascal sonucu modern inference GPU'larına genellenemez.

1050 Ti'ı:

> **algorithm + methodology development platform**

olarak kullan.

İlginç sonuç ortaya çıkarsa daha sonra:

```text
Ampere
Ada
Hopper/Blackwell
```

gibi modern bir GPU'yu kiralayıp tekrarlarsın.

GPU satın almak başlangıç şartı değil.

---

# 22. 1 Kasım'da beklediğim gerçek seviye

Takvimi ciddi uygularsan hedef şu:

**Bağımsız yapabilir:**

- 1D/2D CUDA indexing
- doğru grid/block hesabı
- boundary-safe kernel
- CPU reference test
- basic coalescing analizi
- shared-memory tiling
- basit convolution/stencil
- basic atomics/histogram
- reduction
- scan'in temel algoritmik yapısı
- CUDA-event timing
- block-size experiment
- küçük softmax decomposition
- küçük RMSNorm kernel
- basit fusion experiment
- streams temel kullanımı
- CUDA Graph temel experiment'i
- ölçüm sonucundan makul performance hipotezi üretmek

**Henüz beklemiyorum:**

```text
SOTA FlashAttention kernel
CUTLASS uzmanlığı
Tensor Core microkernels
Triton uzmanlığı
PTX/SASS micro-optimization
Hopper warp specialization
TMA
thread-block clusters
vLLM custom-op production integration
multi-GPU serving
senior GPU performance engineering
```

Bunlar **Phase 2**.

Ve bence bu ayrım önemli: 1 Kasım'a kadar amacımız “CUDA'nın her şeyini bitirmek” değil;

> **bir inference kernelini kendin tasarlayıp yazabilecek, correctness'ini kanıtlayabilecek, benchmark edebilecek ve çıkan sayıyı teknik olarak tartışabilecek hale gelmek.**

Bu altyapı oturduğunda Chris'in ileri bölümleri, vLLM, Triton, CUTLASS ve modern GPU profiling çok daha anlamlı gelecek.

## Bugün/ilk oturumun uygulaması

7 Ekim oturumunu tamamladığını varsayma. **Bugünkü ilk 15 dakikada Ch3/Ch4 gate yap:** boş kağıtta 2D indexing, grid sizing, warp/block/SM'yi anlat. Gate geçerse 8 Ekim tablosundaki **PMPP Ch3 doğrulama + Chris Ch6** oturumuna başla; geçmezse ilk 45–60 dakikayı 7 Ekim Ch3/Ch4 eksiklerine ver. Ardından `1001×777` matrix kernel'ini **AI kodu olmadan** yaz, CPU reference ile doğrula ve `study/notes/daily/2026-10-08.md` içine yalnızca `Hypothesis / Experiment / Result / Why / Next` beşlisini koy. Bu planın ilk gerçek çıktısı o kernel ve o beş satır olsun.
