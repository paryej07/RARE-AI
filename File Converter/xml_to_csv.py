# convert_orphadata_xml.py
import xml.etree.ElementTree as ET
import pandas as pd

tree = ET.parse("/Users/paryejalam/Downloads/en_product4.xml")
root = tree.getroot()

rows = []
for disorder in root.iter("Disorder"):
    orpha_code = disorder.findtext("OrphaCode")
    disease_name = disorder.findtext("Name")
    
    hpo_list = disorder.find("HPODisorderAssociationList")
    if hpo_list:
        for assoc in hpo_list.findall("HPODisorderAssociation"):
            hpo_id = assoc.findtext(".//HPOId")
            hpo_term = assoc.findtext(".//HPOTerm")
            frequency = assoc.findtext(".//HPOFrequency/Name")
            diagnostic_criteria = assoc.findtext(".//DiagnosticCriteria/Name")
            
            rows.append({
                "OrphaCode": orpha_code,
                "DiseaseName": disease_name,
                "HPO_ID": hpo_id,
                "HPO_Term": hpo_term,
                "Frequency": frequency,
                "DiagnosticCriteria": diagnostic_criteria
            })

df = pd.DataFrame(rows)
df.to_csv("/Users/paryejalam/Downloads/en_product4.csv", index=False)
print(f"Done — {len(df)} rows, {df['OrphaCode'].nunique()} diseases")
print(df.head())