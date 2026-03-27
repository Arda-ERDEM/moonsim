import tifffile
import os
from pathlib import Path

 
tif_path = 'img/default_moon_dem.tif'
print(f"Dosya: {tif_path}")
print(f"Boyut: {os.path.getsize(tif_path)} byte")
print("\n" + "="*70)

with tifffile.TiffFile(tif_path) as tif:
    # Genel bilgiler
    arr = tif.asarray()
    print(f"Görüntü boyutu: {arr.shape}")
    print(f"Veri tipi: {arr.dtype}")
    print(f"Min/Max değerler: {arr.min()} / {arr.max()}")
    
    print("\n" + "="*70)
    print("TIFF PAGES VE TAGS:")
    print("="*70)
    
    for page_idx, page in enumerate(tif.pages):
        print(f"\nSayfa {page_idx}:")
        if hasattr(page, 'tags'):
            if page.tags:
                for tag_name, tag_obj in page.tags.items():
                    print(f"  {tag_name}: {tag_obj.value}")
            else:
                print("  Tag yok")
        else:
            print("  Tags özelliği yok")
        
        # Page özellikleri
        print(f"  Shape: {page.asarray().shape}")
        print(f"  Dtype: {page.asarray().dtype}")

    # Tüm properties
    print("\n" + "="*70)
    print("OBJECT ÖZELLİKLERİ:")
    print("="*70)
    if tif.pages:
        attrs = [a for a in dir(tif.pages[0]) if not a.startswith('_')]
        for attr in attrs[:20]:
            try:
                val = getattr(tif.pages[0], attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
            except:
                pass
