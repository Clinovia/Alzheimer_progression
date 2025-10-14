import pandas as pd
import numpy as np

# -------------------------
# 1️⃣ Load datasets
# -------------------------
adnimerge = pd.read_csv("ADNIMERGE.csv", low_memory=False)
apoe = pd.read_csv("apoe4_count.csv")
gdscale = pd.read_csv("GDSCALE.csv")
dxsum = pd.read_csv("DXSUM.csv")

# -------------------------
# 2️⃣ Filter ADNIMERGE and DXSUM
# -------------------------
valid_dx_bl = ['CN', 'LMCI', 'AD']
adnimerge = adnimerge[adnimerge['DX_bl'].isin(valid_dx_bl)]

valid_diagnosis = [1, 2, 3]
dxsum = dxsum[dxsum['DIAGNOSIS'].isin(valid_diagnosis)]

# -------------------------
# 3️⃣ Core features from ADNIMERGE
# -------------------------
core_features = ['RID','VISCODE','EXAMDATE','AGE','PTGENDER','PTEDUCAT',
                 'ADAS13','MOCA','CDRSB','FAQ']
adnimerge_core = adnimerge[core_features].copy()

# -------------------------
# 4️⃣ Merge APOE4_count
# -------------------------
adnimerge_core = adnimerge_core.merge(apoe[['RID','APOE4_count']], on='RID', how='left')

# -------------------------
# 5️⃣ Prepare GDScale
# -------------------------
gdscale['VISDATE'] = pd.to_datetime(gdscale['VISDATE'], errors='coerce')
gdscale = gdscale.dropna(subset=['VISDATE'])
gdscale_sub = gdscale[['RID','VISCODE','VISDATE','GDTOTAL']]

adnimerge_core['EXAMDATE'] = pd.to_datetime(adnimerge_core['EXAMDATE'], errors='coerce')
adnimerge_core = adnimerge_core.dropna(subset=['EXAMDATE'])

# -------------------------
# 6️⃣ Merge GDScale using nearest date
# -------------------------
adnimerge_core = pd.merge_asof(
    adnimerge_core.sort_values('EXAMDATE'),
    gdscale_sub.sort_values('VISDATE'),
    by='RID',
    left_on='EXAMDATE',
    right_on='VISDATE',
    direction='nearest',
    tolerance=pd.Timedelta(days=90)
)
adnimerge_core.drop(columns=['VISDATE'], inplace=True)

# -------------------------
# 7️⃣ Merge DXSUM for target computation
# -------------------------
dxsum['EXAMDATE'] = pd.to_datetime(dxsum['EXAMDATE'], errors='coerce')
dxsum = dxsum.dropna(subset=['EXAMDATE'])
dxsum_sub = dxsum[['RID','VISCODE','EXAMDATE','DIAGNOSIS']]

adnimerge_core = pd.merge_asof(
    adnimerge_core.sort_values('EXAMDATE'),
    dxsum_sub.sort_values('EXAMDATE'),
    by='RID',
    left_on='EXAMDATE',
    right_on='EXAMDATE',
    direction='nearest',
    tolerance=pd.Timedelta(days=90)
)

# -------------------------
# 8️⃣ Generate multi-horizon progression targets
# -------------------------
adnimerge_core['AD_binary'] = adnimerge_core['DIAGNOSIS'].apply(lambda dx: 1 if dx == 2 else 0)
adnimerge_core = adnimerge_core.sort_values(['RID','EXAMDATE'])

horizons = [1, 2, 3, 5]  # years
target_cols = [f'Target_{h}yr' for h in horizons]

for rid, group in adnimerge_core.groupby('RID'):
    visits = group['EXAMDATE'].values
    ad_flags = group['AD_binary'].values
    for i, baseline_date in enumerate(visits):
        future_indices = np.where(visits > baseline_date)[0]
        for horizon, col in zip(horizons, target_cols):
            horizon_days = horizon * 365.25
            converted = False
            for j in future_indices:
                delta_days = (visits[j] - baseline_date) / np.timedelta64(1, 'D')
                if delta_days <= horizon_days and ad_flags[j] == 1:
                    converted = True
                    break
            adnimerge_core.loc[group.index[i], col] = int(converted)

# -------------------------
# 9️⃣ Save core model dataset
# -------------------------
core_columns = core_features + ['APOE4_count','GDTOTAL','DIAGNOSIS'] + target_cols
df_core = adnimerge_core[core_columns]
df_core.to_csv("ADNI_core_model_dataset.csv", index=False)
print("Core dataset ready: ADNI_core_model_dataset.csv")

# -------------------------
# 🔟 Extended model: merge CSF and PET features
# -------------------------
# Adjust these if your CSV uses _bl suffix, e.g., 'ABETA_bl'
csf_pet_features = ['ABETA','TAU','PTAU','FDG','PIB','AV45','FBB']

# Keep only CSF/PET columns + RID + EXAMDATE for merging
csf_pet_data = adnimerge[['RID','EXAMDATE'] + csf_pet_features].copy()
csf_pet_data['EXAMDATE'] = pd.to_datetime(csf_pet_data['EXAMDATE'], errors='coerce')
csf_pet_data = csf_pet_data.dropna(subset=['EXAMDATE'])

# Merge with core dataset
df_extended = pd.merge_asof(
    adnimerge_core.sort_values('EXAMDATE'),
    csf_pet_data.sort_values('EXAMDATE'),
    by='RID',
    left_on='EXAMDATE',
    right_on='EXAMDATE',
    direction='nearest',
    tolerance=pd.Timedelta(days=90)
)

df_extended.to_csv("ADNI_extended_model_dataset.csv", index=False)
print("Extended dataset ready: ADNI_extended_model_dataset.csv")
