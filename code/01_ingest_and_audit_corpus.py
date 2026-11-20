import os
import re
import glob
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def extract_year(val):
    if val is None or pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in ["", "unknown", "none", "nan", "historic", "unrecorded"]:
        return np.nan
    matches = [int(y) for y in re.findall(r"\b([3-9]\d{2}|1[0-6]\d{2})\b", s)]
    return float(np.mean(matches)) if matches else np.nan


def classify_icon(text: str) -> str:
    t = str(text).lower()
    if any(k in t for k in ["avalokiteshvara", "tara", "maitreya", "manjushri", "bodhisattva", "padmapani"]):
        return "Bodhisattva_Cult"
    if any(k in t for k in ["medicine", "bhaisajyaguru", "healing", "dharani", "tantric", "vajra"]):
        return "Protective_Esoteric"
    if any(k in t for k in ["reliquary", "stupa", "votive", "tsatsa", "creed", "monk", "offering"]):
        return "Ritual_Votive_Relic"
    if any(k in t for k in ["sima", "sasana", "charter", "land", "grant", "temple", "punya", "monastery"]):
        return "Monastic_Endowment"
    return "Buddha_Canonical"


def classify_region(reg: str, part: str) -> str:
    r = str(reg).lower()
    if part == "DHARMA" or any(k in r for k in ["java", "sumatra", "bali", "vietnam", "champa", "indonesia", "sea", "maritime", "malacca", "kedah", "thailand", "bangkok"]):
        return "Maritime_Mainland_SEA"
    if any(k in r for k in ["tibet", "nepal", "himalaya", "kashmir", "swat", "ladakh"]):
        return "Trans_Himalayan"
    if any(k in r for k in ["india", "nalanda", "bihar", "bengal", "gandhara", "south asia"]):
        return "South_Asia_Core"
    if any(k in r for k in ["china", "japan", "korea", "east asia"]):
        return "East_Asia"
    return "Unassigned_Contact_Zone"


def classify_material(mat: str, part: str) -> str:
    m = str(mat).lower()
    if part == "DHARMA":
        return "Inscribed_Copper_Plate" if ("copper" in m or "bronze" in m) else "Lithic_Stone_Stele"
    if "gilt" in m or "gold" in m:
        return "Gilt_Copper_Alloy"
    if "brass" in m:
        return "Cast_Brass"
    return "Cast_Bronze_Copper"


def run_ingestion(raw_dir="../data/raw_data", proc_dir="../data/processed_data", tables_dir="../tables"):
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    records = []
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    # 1. ERC DHARMA TEI-XML
    for fp in sorted(glob.glob(os.path.join(raw_dir, "dharma_xml", "**", "*.xml"), recursive=True)):
        try:
            root = ET.parse(fp).getroot()
            body_node = root.find(".//tei:body", ns)
            body_text = " ".join(body_node.itertext()) if body_node is not None else " ".join([e.text for e in root.iter() if e.text])
            body_text = " ".join(body_text.split())
            if len(body_text) <= 35:
                continue
            fid = os.path.splitext(os.path.basename(fp))[0]
            t_node = root.find(".//tei:titleStmt/tei:title", ns)
            title = t_node.text.strip() if (t_node is not None and t_node.text) else fid
            d_node = root.find(".//tei:origDate", ns)
            raw_d = d_node.attrib.get("notBefore", "") if d_node is not None else ""
            yr = extract_year(raw_d)
            places = [p.text.strip() for p in root.findall(".//tei:placeName", ns) if p.text]
            records.append({
                "artifact_id": fid,
                "partition": "DHARMA",
                "title": title[:80].replace("\n", " "),
                "raw_region": places[0] if places else "Maritime Southeast Asia",
                "macro_region": classify_region(places[0] if places else "SEA", "DHARMA"),
                "has_origin": True,
                "raw_date": raw_d if raw_d else "Unrecorded",
                "year_ce": yr,
                "is_dated": not np.isnan(yr),
                "material_clean": classify_material("stone", "DHARMA"),
                "iconography_class": classify_icon(f"{title} {body_text[:600]}"),
                "token_count": len(body_text.split()),
            })
        except Exception:
            continue

    # 2. V&A Museum JSON
    va_dir = os.path.join(raw_dir, "va_sculptures")
    if not os.path.exists(va_dir):
        va_dir = os.path.join(raw_dir, "va_portable_buddhist_sculptures")
    for jf in sorted(glob.glob(os.path.join(va_dir, "**", "*.json"), recursive=True)):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            for it in (data if isinstance(data, list) else [data]):
                aid = str(it.get("systemNumber") or it.get("objectNumber") or os.path.splitext(os.path.basename(jf))[0])
                title = str(it.get("_primaryTitle") or it.get("objectType") or "Buddhist portable bronze")
                place = str(it.get("_primaryPlace") or "").strip()
                raw_d = str(it.get("_primaryDate") or "").strip()
                yr = extract_year(raw_d)
                mat_raw = str(it.get("materials") or "bronze")
                desc = str(it.get("summaryDescription") or "")
                records.append({
                    "artifact_id": aid,
                    "partition": "VA",
                    "title": title[:80].replace("\n", " "),
                    "raw_region": place if place else "Unknown",
                    "macro_region": classify_region(place, "VA"),
                    "has_origin": bool(place and place.lower() not in ["unknown", "none", "nan"]),
                    "raw_date": raw_d if raw_d else "Unrecorded",
                    "year_ce": yr,
                    "is_dated": not np.isnan(yr),
                    "material_clean": classify_material(mat_raw, "VA"),
                    "iconography_class": classify_icon(f"{title} {desc}"),
                    "token_count": len(f"{title} {desc}".split()),
                })
        except Exception:
            continue

    # 3. Cleveland Museum of Art JSON
    for jf in sorted(glob.glob(os.path.join(raw_dir, "cma_sculptures", "**", "*.json"), recursive=True)):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("data", [data])
            for it in items:
                aid = str(it.get("accession_number") or it.get("id") or os.path.splitext(os.path.basename(jf))[0])
                title = str(it.get("title") or "Buddhist Sculpture")
                cult = it.get("culture", [])
                place = cult[0] if (isinstance(cult, list) and len(cult) > 0) else str(cult or "").strip()
                raw_d = str(it.get("creation_date") or "").strip()
                yr = extract_year(raw_d)
                tech = str(it.get("technique") or "bronze")
                desc = str(it.get("description") or "")
                records.append({
                    "artifact_id": aid,
                    "partition": "CMA",
                    "title": title[:80].replace("\n", " "),
                    "raw_region": place if place else "Unknown",
                    "macro_region": classify_region(place, "CMA"),
                    "has_origin": bool(place and place.lower() not in ["unknown", "none", "nan", "[]"]),
                    "raw_date": raw_d if raw_d else "Unrecorded",
                    "year_ce": yr,
                    "is_dated": not np.isnan(yr),
                    "material_clean": classify_material(tech, "CMA"),
                    "iconography_class": classify_icon(f"{title} {tech} {desc}"),
                    "token_count": len(f"{title} {tech} {desc}".split()),
                })
        except Exception:
            continue

    df = pd.DataFrame(records)
    median_year = float(df.loc[df["is_dated"], "year_ce"].median()) if df["is_dated"].any() else 900.0
    df["year_ce_imputed"] = df["year_ce"].fillna(median_year)

    corpus_csv = os.path.join(proc_dir, "structured_corpus_audit.csv")
    df.to_csv(corpus_csv, index=False)

    # Generate Table 1: Provenance and Chronological Completeness
    t1_rows = []
    for p, label in [("DHARMA", "DHARMA Project"), ("VA", "Victoria & Albert Museum"), ("CMA", "Cleveland Museum of Art")]:
        sub = df[df["partition"] == p]
        dated_yrs = sub.loc[sub["is_dated"], "year_ce"].dropna()
        d_range = f"{int(dated_yrs.min())}--{int(dated_yrs.max())}" if len(dated_yrs) > 0 else "Unrecorded"
        t1_rows.append({
            "Corpus_Partition": label,
            "Partition_Code": p,
            "Records": len(sub),
            "Origin_Recorded_Pct": round(100.0 * sub["has_origin"].mean(), 1),
            "Dated_Pct": round(100.0 * sub["is_dated"].mean(), 1),
            "Date_Range_CE": d_range,
            "Primary_Media": ", ".join(sub["material_clean"].value_counts().index[:2]),
        })
    pd.DataFrame(t1_rows).to_csv(os.path.join(tables_dir, "Table1_Corpus_Composition.csv"), index=False)
    print(f"[01_ingest_and_audit_corpus] Saved {corpus_csv} (N={len(df)}) and Table1_Corpus_Composition.csv")


if __name__ == "__main__":
    run_ingestion()
