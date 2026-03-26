# 🚀 Hızlı Başlangıç - Moon Generator

Uygulamayı çalıştırmanın 3 kolay yolu:

## 1️⃣ **En Basit: Batch Dosyası (Çift tıkla)**
```
run.bat
```
Masaüstünde `run.bat` dosyasını çift tıkla - GUI anında açılır!

## 2️⃣ **Terminal: Kısa Komut**

### PowerShell'de:
```powershell
.\run.ps1
```

### Bash'de (Linux/macOS):
```bash
./run.sh
```

## 3️⃣ **VS Code: Command Palette**
1. `Ctrl+Shift+P` aç
2. `Run Moon Generator GUI` yazıp Enter'e bas
3. GUI açılır ve output panel'de görünür

---

## 📝 Ne Yaptı?

✅ `run.bat` - Windows için çift tıkla launcher  
✅ `run.ps1` - PowerShell script  
✅ `run.sh` - Linux/macOS için script  
✅ `.vscode/tasks.json` - VS Code task integration  

Artık `python -m moon_gen` yazman gerekmez! 

---

## Alternativler

Eğer shortest alias istersen:
```bash
# .bashrc veya .profile'ye ekle (Linux/macOS):
alias moon="cd ~/Desktop/moonsim && python -m moon_gen"

# PowerShell $PROFILE'ye ekle:
Set-Alias -Name moon -Value .\run.ps1
```

Sonra sadece `moon` yaz! 🌙
