import os
import pandas as pd

# 1. Load Data
base_path = r"C:\Users\gunar\OneDrive\Desktop\fraud detection\Data"
bene_df = pd.read_csv(os.path.join(base_path, "Train_Beneficiarydata.csv"))
ip_df = pd.read_csv(os.path.join(base_path, "Train_Inpatientdata.csv"))
op_df = pd.read_csv(os.path.join(base_path, "Train_Outpatientdata.csv"))
tgt_df = pd.read_csv(os.path.join(base_path, "Train_Target.csv"))

# 2. Label admission type
ip_df = ip_df.assign(**{"Admitted?": 1})
op_df = op_df.assign(**{"Admitted?": 0})

# 3. Combine inpatient & outpatient
combined = pd.concat([ip_df, op_df], ignore_index=True)

# 4. Merge beneficiary & target
merged = (
    combined
    .merge(bene_df, on="BeneID", how="inner")
    .merge(tgt_df, on="Provider", how="inner")
)

# 5. Convert date columns
dates = ["DOB","DOD","ClaimStartDt","ClaimEndDt","AdmissionDt","DischargeDt"]
merged[dates] = merged[dates].apply(pd.to_datetime, format="%Y-%m-%d")

# 6. Fill missing DOD, compute age
max_dod = merged["DOD"].max()
merged["DOD"] = merged["DOD"].fillna(max_dod)
merged["Bene_Age"] = ((merged["DOD"] - merged["DOB"]).dt.days/365).round(1)

# 7. Durations
merged["Claim_Duration"] = (merged["ClaimEndDt"] - merged["ClaimStartDt"]).dt.days
merged["Admitted_Duration"] = (
    merged["DischargeDt"] - merged["AdmissionDt"]
).dt.days.fillna(0).astype(int)

# 8. Alive flag
merged["Is_Alive?"] = merged["DOD"].ne(max_dod).astype(int)

# 9. Physician claim counts
def count_transform(df, group, col):
    return df.groupby(group)[col].transform("count")

for phy, colname in [
    ("AttendingPhysician","Att_Phy_tot_claims"),
    ("OperatingPhysician","Opr_Phy_tot_claims"),
    ("OtherPhysician","Oth_Phy_tot_claims")
]:
    merged[colname] = count_transform(merged, phy, "ClaimID").fillna(0).astype(int)
merged["Att_Opr_Oth_Phy_Tot_Claims"] = (
    merged["Att_Phy_tot_claims"] +
    merged["Opr_Phy_tot_claims"] +
    merged["Oth_Phy_tot_claims"]
)

# 10. Prepare DOB year
df = merged
merged['DOB_Year'] = df['DOB'].dt.year
prov = merged.groupby("Provider")

# 11. Provider-level aggregates
merged["Prv_Tot_Att_Opr_Oth_Phys"] = (
    prov["AttendingPhysician"].transform("count") +
    prov["OperatingPhysician"].transform("count") +
    prov["OtherPhysician"].transform("count")
)
merged["PRV_Tot_Admit_DCodes"] = prov["ClmAdmitDiagnosisCode"].transform("nunique")
merged["PRV_Tot_DGrpCodes"] = prov["DiagnosisGroupCode"].transform("nunique")
merged["PRV_Tot_Unq_DOB_Years"] = prov["DOB_Year"].transform("nunique")
merged["PRV_Bene_Age_Sum"] = prov["Bene_Age"].transform("sum")
merged["RenalDiseaseIndicator"] = merged["RenalDiseaseIndicator"].eq("Y").astype(int)
merged["PRV_Tot_RKD_Patients"] = prov["RenalDiseaseIndicator"].transform("sum")
# annual/deductible sums
for src, tgt in {
    "DeductibleAmtPaid":"PRV_CoPayment",
    "IPAnnualReimbursementAmt":"PRV_IP_Annual_ReImb_Amt",
    "IPAnnualDeductibleAmt":"PRV_IP_Annual_Ded_Amt",
    "OPAnnualReimbursementAmt":"PRV_OP_Annual_ReImb_Amt",
    "OPAnnualDeductibleAmt":"PRV_OP_Annual_Ded_Amt",
    "Admitted_Duration":"PRV_Admit_Duration",
    "Claim_Duration":"PRV_Claim_Duration"
}.items():
    merged[tgt] = prov[src].transform("sum")

# 12. Generic aggregations
def make_agg(df, grp, prefix, cols):
    out = { }
    g = df.groupby(grp)
    for c in cols:
        if c in df:
            out[f"{prefix}_{c}"] = g[c].transform("sum")
    return pd.DataFrame(out, index=df.index)

num_cols = [
    "InscClaimAmtReimbursed","DeductibleAmtPaid","IPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt","OPAnnualReimbursementAmt","OPAnnualDeductibleAmt",
    "Admitted_Duration","Claim_Duration"
]
frames = [merged]
for grp,prefix in [
    ("BeneID","BENE"), ("AttendingPhysician","ATT_PHY"),
    ("OperatingPhysician","OPT_PHY"), ("OtherPhysician","OTH_PHY"),
    ("ClmAdmitDiagnosisCode","Claim_Admit_Diag_Code"),
    ("DiagnosisGroupCode","Diag_GCode")
]:
    frames.append(make_agg(merged, grp, prefix, num_cols))
for i in range(1,11): frames.append(make_agg(merged, f"ClmDiagnosisCode_{i}", f"Claim_DiagCode{i}", num_cols))
for i in range(1,4): frames.append(make_agg(merged, f"ClmProcedureCode_{i}", f"Claim_ProcCode{i}", num_cols))
merged = pd.concat(frames, axis=1)

# 13. Detailed ClmCount blocks
count_frames = []
basic = [
    ["Provider"], ["Provider","BeneID"], ["Provider","AttendingPhysician"],
    ["Provider","OtherPhysician"], ["Provider","OperatingPhysician"],
    ["Provider","ClmAdmitDiagnosisCode"]
] + [["Provider",f"ClmProcedureCode_{i}"] for i in range(1,4)]
for grp in basic:
    name = "ClmCount_" + "_".join(grp)
    count_frames.append(merged.groupby(grp)["ClaimID"].transform("count").rename(name))
for i in range(1,11):
    grp = ["Provider", f"ClmDiagnosisCode_{i}"]
    count_frames.append(merged.groupby(grp)["ClaimID"].transform("count").rename(f"ClmCount_Provider_ClmDiagnosisCode_{i}"))
for phy in ["AttendingPhysician","OtherPhysician","OperatingPhysician"]:
    base = ["Provider","BeneID",phy]
    count_frames.append(merged.groupby(base)["ClaimID"].transform("count").rename(f"ClmCount_Provider_BeneID_{phy}"))
    for j in range(1,4): count_frames.append(merged.groupby(base+[f"ClmProcedureCode_{j}"])["ClaimID"].transform("count").rename(f"ClmCount_Provider_BeneID_{phy}_ClmProcedureCode_{j}"))
    for i in range(1,11): count_frames.append(merged.groupby(base+[f"ClmDiagnosisCode_{i}"])["ClaimID"].transform("count").rename(f"ClmCount_Provider_BeneID_{phy}_ClmDiagnosisCode_{i}"))
for j in range(1,4): count_frames.append(merged.groupby(["Provider","BeneID",f"ClmProcedureCode_{j}"])["ClaimID"].transform("count").rename(f"ClmCount_Provider_BeneID_ClmProcedureCode_{j}"))
for i in range(1,11): count_frames.append(merged.groupby(["Provider","BeneID",f"ClmDiagnosisCode_{i}"])["ClaimID"].transform("count").rename(f"ClmCount_Provider_BeneID_ClmDiagnosisCode_{i}"))
for i in range(1,11):
    for j in range(1,4): count_frames.append(
        merged.groupby(["Provider","BeneID",f"ClmDiagnosisCode_{i}",f"ClmProcedureCode_{j}"])["ClaimID"].transform("count").rename(f"ClmCount_Provider_BeneID_ClmDiagnosisCode_{i}_ClmProcedureCode_{j}")
    )
merged = pd.concat([merged] + count_frames, axis=1)

# 14. Defragment DataFrame
merged = merged.copy()

# 15. Drop unwanted columns
drop_cols = [
    'BeneID','ClaimID','ClaimStartDt','ClaimEndDt','AttendingPhysician',
    'OperatingPhysician','OtherPhysician','AdmissionDt','DischargeDt',
    'ClmAdmitDiagnosisCode','DiagnosisGroupCode','DOB_Year',
    'ClmProcedureCode_4','ClmProcedureCode_5','ClmProcedureCode_6',
    'NoOfMonths_PartACov','NoOfMonths_PartBCov'
] + [f'ClmDiagnosisCode_{i}' for i in range(1,11)] + [f'ClmProcedureCode_{i}' for i in range(1,7)] + ['DOB','DOD','State','County']
merged.drop(columns=[c for c in drop_cols if c in merged], inplace=True)

# 16. Final encoding & output
merged = merged.assign(**{ 'DeductibleAmtPaid': merged['DeductibleAmtPaid'].fillna(0) })
merged['Gender'] = (merged['Gender'] != 2).astype(int)
merged['PotentialFraud'] = merged['PotentialFraud'].eq('Yes').astype(int)
merged = pd.get_dummies(merged, columns=['Gender','Race','Admitted?','Is_Alive?'], drop_first=True)
merged.fillna(0, inplace=True)

# 17. Final grouping
final = merged.groupby(['Provider','PotentialFraud'], as_index=False).sum()
final.to_csv("final_merged_and_preprocessed_data.csv", index=False)
final.drop(columns=['Provider','PotentialFraud']).to_csv("Features.csv", index=False)
final['PotentialFraud'].to_csv("Target.csv", index=False)
