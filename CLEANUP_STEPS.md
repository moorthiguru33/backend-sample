# 🧹 Cleanup Steps — Duplicate Files நீக்குவது எப்படி

## நிலைமை (Current Status)

| விவரம் | எண்ணிக்கை |
|--------|-----------|
| forpsd.com மொத்த files | ~6,918 |
| rename_log.xlsx valid files | **4,702** |
| Drive-ல் மொத்த files (estimate) | ~10,644 |
| Duplicate/extra files | **~5,942** |
| Counter gaps (big jumps) | 4 இடங்கள் |

## Counter Gaps — எங்கே தவறு ஆனது

```
tamilpsd-2303 → tamilpsd-2830  (526 extra files)
tamilpsd-3143 → tamilpsd-5452  (2,308 extra files)  ← மிகவும் பெரியது
tamilpsd-7097 → tamilpsd-7754  (656 extra files)
tamilpsd-7765 → tamilpsd-10167 (2,401 extra files)  ← மிகவும் பெரியது
```

இந்த gaps-ல் உள்ள files — **Drive-ல் உள்ளன ஆனால் rename_log-ல் இல்லை** → Delete செய்ய வேண்டும்.

---

## Step 1: Files ZIP Push செய்யவும்

இந்த ZIP-ல் உள்ள files-ஐ repo-ல் replace செய்யவும்:

| File | என்ன மாறியது |
|------|--------------|
| `.github/workflows/automation.yml` | Schedule நீக்கப்பட்டது, auto-trigger நீக்கப்பட்டது |
| `.github/workflows/psd_pipeline.yml` | Schedule நீக்கப்பட்டது |
| `.github/workflows/cleanup.yml` | 🆕 புதிய cleanup workflow |
| `scripts/cleanup_duplicates.py` | 🆕 Duplicate delete script |
| `state.json` | `total_processed: 5025` சரி செய்யப்பட்டது |

---

## Step 2: Dry Run — என்ன delete ஆகும் என்று பாருங்கள்

```
GitHub → Actions → Cleanup Duplicate Files
→ Run workflow
→ dry_run: true   ← இப்போது இதை தேர்ந்தெடுங்கள்
→ Run workflow
```

Log-ல் காட்டும்:
- Drive-ல் எத்தனை files valid / extra
- preview_image/-ல் எத்தனை files valid / extra
- **ஒன்றும் delete ஆகாது** — பார்க்கமட்டும்

---

## Step 3: Actual Delete

Dry run result திருப்தியாக இருந்தால்:

```
GitHub → Actions → Cleanup Duplicate Files
→ Run workflow
→ dry_run: false  ← இப்போது இதை தேர்ந்தெடுங்கள்
→ Run workflow
```

இது செய்யும்:
1. ✅ Drive-ல் rename_log-இல் இல்லாத ~5,942 files delete
2. ✅ preview_image/-ல் 407 extra .webp files delete
3. ✅ Git commit + push (preview_image/ clean ஆகும்)

---

## Step 4: Future Runs — இனி duplicate வராது

இனி pipeline run செய்ய:
```
GitHub → Actions → PSD Automation Pipeline → Run workflow
```
- Schedule இல்லை — நீங்களே run செய்ய வேண்டும்
- Item limit default: 50 (ஒரு run-ல் 50 files)
- Auto restart இல்லை — run முடிந்தவுடன் நிற்கும்

---

## rename_log.xlsx — Source of Truth

| Original (forpsd) | TamilPSD Name |
|-------------------|---------------|
| 6327.psd | tamilpsd-0001.psd |
| 6326.psd | tamilpsd-0002.psd |
| ... | ... |
| 1758.psd | tamilpsd-10644.psd |

**Next file:** tamilpsd-10645
