import pandas as pd
import numpy as np

# -------------------------
# 1️⃣ Load datasets
# -------------------------
adnimerge = pd.read_csv("ADNIMERGE.csv")
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
# 5️⃣ Merge GDScale using nearest date
# -------------------------
gdscale['VISDATE'] = pd.to_datetime(gdscale['VISDATE'], errors='coerce')
gdscale = gdscale.dropna(subset=['VISDATE'])
gdscale_sub = gdscale[['RID','VISCODE','VISDATE','GDTOTAL']]

adnimerge_core['EXAMDATE'] = pd.to_datetime(adnimerge_core['EXAMDATE'])

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
# 6️⃣ Merge DXSUM for target computation
# -------------------------
dxsum['EXAMDATE'] = pd.to_datetime(dxsum['EXAMDATE'])
dxsum_sub = dxsum[['RID','VISCODE','EXAMDATE','DXSUM']]

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
# 7️⃣ Generate multi-horizon progression targets
# -------------------------
adnimerge_core['AD_binary'] = adnimerge_core['DXSUM'].apply(lambda dx: 1 if dx == 2 else 0)
adnimerge_core = adnimerge_core.sort_values(['RID','EXAMDATE'])

horizons = [1, 2, 3, 5]  # in years
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
                if (visits[j] - baseline_date).astype('timedelta64[D]').item() <= horizon_days and ad_flags[j] == 1:
                    converted = True
                    break
            adnimerge_core.loc[group.index[i], col] = int(converted)

# -------------------------
# 8️⃣ Save core model dataset
# -------------------------
core_columns = core_features + ['APOE4_count','GDTOTAL','DXSUM'] + target_cols
df_core = adnimerge_core[core_columns]
df_core.to_csv("ADNI_core_model_dataset.csv", index=False)
print("Core dataset ready: ADNI_core_model_dataset.csv")

# -------------------------
# 9️⃣ Extended model: add CSF and PET features
# -------------------------
csf_pet_features = ['ABETA','TAU','PTAU','FDG','PIB','AV45','FBB']
df_extended = adnimerge_core[core_columns + csf_pet_features]
df_extended.to_csv("ADNI_extended_model_dataset.csv", index=False)
print("Extended dataset ready: ADNI_extended_model_dataset.csv")
