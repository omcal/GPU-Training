# Ubuntu CUDA geçişi — envanter ve durum

## Güncel durum: geçiş tamamlandı

27 Eylül 2026: kaynaklar `~/GPU-Training` dizinine aktarıldı.
İki PMPP örneği ve altı yeni CUDA laboratuvarı CUDA 12.9 / `sm_61` ile
derlenip gerçek GPU üzerinde çalıştırıldı. 78 CPU karşılaştırması geçti;
sekiz programın memcheck, racecheck ve synccheck kontrolleri temiz.

Kullanım: [`../cuda/README.md`](../cuda/README.md).
Ölçümler ve doğrulama: [`07-cuda-validation.md`](07-cuda-validation.md).

## İlk inceleme kaydı (uygulama öncesi)

Aşağıdaki bölümler aktarım öncesindeki durumu ve o sıradaki planı kaydeder.

27 Eylül 2026 tarihinde yerel kaynaklar ve `ssh cuda` üzerinden sistem incelendi.
Bu aşamada proje aktarılmadı, uzak makineye paket kurulmadı ve proje kernel'leri
derlenip çalıştırılmadı.

## Yerel proje

- Altı laboratuvar: vector add, coalescing/transpose, reduction, matmul,
  convolution ve scan. Hepsi Python içinden MLX Metal kernel'leri çalıştırıyor.
- `gpuk/` ölçüm ve donanım yardımcıları da Metal/MLX'e bağlı.
  `gpuk.launch.cuda()` bir CUDA backend'i değil; CUDA launch geometrisini
  Metal launch geometrisine çeviren yardımcı.
- `pmpp/01_vector_add/kernel.cu` ve `pmpp/02_tiled_matmul/kernel.cu`
  gerçek CUDA C++ örnekleri ve host kodu içeriyor.
- `pmpp/run_all.py` ve mevcut pytest testleri Metal sürümlerini çalıştırıyor.
- `pyproject.toml` mevcut Mac müfredatının bağımlılıklarını tanımlıyor;
  Python paketini kurmak Metal kernel'lerini CUDA'ya dönüştürmez.
- Klasör bir Git deposu değil. Kaynak/dokümanlar yaklaşık 233 KB;
  toplam 693 MB'ın büyük kısmı `.venv`, `.uv-cache`, `.swiftcache`.

## Uzak makine: doğrulanan durum

| Bileşen | Durum |
|---|---|
| SSH | Yerel `cuda` alias'ı; bağlantı bilgileri repoda tutulmaz |
| İşletim sistemi | Ubuntu 24.04.5 LTS, x86_64 |
| CPU | Intel Core i5-8300H, 4 çekirdek / 8 thread |
| RAM | Yaklaşık 16 GB fiziksel RAM; `free -h`: 15 GiB |
| GPU | NVIDIA GeForce GTX 1050 Ti with Max-Q Design |
| GPU belleği | 4096 MiB |
| Compute capability | 6.1; derleme hedefi `sm_61` |
| Sürücü | 580.178.04 |
| CUDA Toolkit | 12.9; `nvcc` V12.9.86 |
| Derleyici yolu | `/usr/local/cuda-12.9/bin/nvcc` |
| Host derleyici | GCC 13.3.0 |
| Python | 3.12.3 |
| Araçlar | Git, make, rsync mevcut; `uv` SSH PATH'inde bulunmadı |
| Disk | Ev dizininin bulunduğu dosya sisteminde yaklaşık 196 GB boş |

`nvcc --list-gpu-code` çıktısında `sm_61` doğrulandı. Etkileşimsiz SSH
oturumunun PATH'inde CUDA bin dizini yok; otomasyonlarda tam yol kullanmak
veya PATH'i komut içinde açıkça ayarlamak gerekiyor.

`nvidia-smi` başlığındaki CUDA 13.0, kurulu Toolkit sürümü değildir.
Bu Pascal GPU için mevcut CUDA 12.9 doğru seçimdir: CUDA 13.0, Pascal için
offline derleme ve kütüphane desteğini kaldırdı. NVIDIA, bu mimariler için
CUDA 12.9 veya öncesini ve R580 sürücü dalını belirtir.

Kaynak: [NVIDIA CUDA 13.0 açıklaması](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/).

## Önerilen geçiş sırası

1. Mac sürümünü koruyup Ubuntu'da ayrı bir `GPU-Training` çalışma dizini oluştur.
   Kaynakları ve dokümanları aktar; sanal ortam, önbellekler, `__pycache__`,
   derleme çıktıları ve test önbelleğini dışarıda bırak.
2. İlk doğrulama olarak mevcut iki `.cu` örneğini CUDA 12.9 ile `-arch=sm_61`
   kullanarak derle ve çalıştır. Dosyalardaki mevcut örnek komutlar `sm_75`
   ve farklı dosya adları kullanıyor; bu makine için düzeltilmeli.
3. CUDA API/launch hatalarını kontrol eden ortak bir yardımcı ve tekrarlanabilir
   derleme/çalıştırma komutu ekle. Mevcut `.cu` kodları CUDA dönüş değerlerini
   kontrol etmiyor; matmul örneği yalnızca 256x256, tüm elemanları 1 olan girdiyi
   doğruluyor. Bunlar başlangıç örnekleri, kapsamlı doğruluk testleri değil.
4. CUDA laboratuvarlarını ayrı bir dizinde sırayla geliştir:
   vector add → transpose/coalescing → reduction → tiled matmul → convolution → scan.
   Mevcut Metal kodu ve dokümanlar karşılaştırma kaynağı olarak kalsın.
5. Her CUDA laboratuvarına CPU referansı ve sınır boyutlarıyla doğruluk kontrolü ekle.
   Benchmark'larda CUDA events, ısınma ve kernel/aktarım sürelerinin ayrı ölçümü kullan.
   M4 ölçümlerini bu karta taşımak yerine yeniden ölç.

Başlangıç için Python/MLX kurulumu gerekmeyen CUDA C++ yolu yeterli.
Bu donanım temel kernel, bellek erişimi, shared memory, senkronizasyon ve tiling
çalışmaları için uygun; 4 GB VRAM ve eski mimarisi nedeniyle büyük model eğitimi
ya da modern GPU'lara özgü optimizasyonları ana hedef yapmak uygun değil.

Aktarım sonrasında, proje kökünde kullanılabilecek ilk komutlar
(bu incelemede çalıştırılmadı):

```bash
mkdir -p build/cuda
/usr/local/cuda-12.9/bin/nvcc -O2 -arch=sm_61 \
  pmpp/01_vector_add/kernel.cu -o build/cuda/vector_add
/usr/local/cuda-12.9/bin/nvcc -O2 -arch=sm_61 \
  pmpp/02_tiled_matmul/kernel.cu -o build/cuda/tiled_matmul
./build/cuda/vector_add
./build/cuda/tiled_matmul
```

Beklenen doğruluk çıktıları sırasıyla `0 of 1048576 elements wrong` ve
`0 of 65536 elements wrong`. Bunlar beklenen sonuçlardır; henüz ölçülmüş sonuç değildir.
