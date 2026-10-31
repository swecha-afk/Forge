"""ICAR Agronomic Knowledge Seeder & FAISS Vector Index Builder.

Ingests verified Indian Council of Agricultural Research (ICAR) protocols into
SQLite WAL storage and generates a faiss.IndexIDMap2(faiss.IndexFlatIP(384))
dense vector index using BAAI/bge-small-en-v1.5.
"""

import os
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.services.rag_engine.db_manager import DEFAULT_DB_PATH, SQLiteKnowledgeManager

DEFAULT_FAISS_PATH = os.path.join("artifacts", "icar_faiss.index")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Verified Indian Council of Agricultural Research (ICAR) Document Chunks
ICAR_SEEDED_PROTOCOLS: list[dict[str, Any]] = [
    # --- POTATO: Late Blight (Phytophthora infestans) ---
    {
        "crop_name": "potato",
        "disease_or_pest": "late_blight",
        "growth_stage": "Tuber Bulking & Canopy Closure",
        "agro_climatic_zone": "Indo-Gangetic Plains & Northern Hills",
        "raw_content": (
            "ICAR-CPRI Protocol for Late Blight (Phytophthora infestans) in Potato: Under cool nights (10-18 deg C) "
            "and persistent relative humidity >85% for 48 hours, latent spore incubation accelerates rapidly. "
            "Curative systemic intervention: Apply Cymoxanil 8% + Mancozeb 64% WP at 2.5 g/L of water (1.0 kg/acre in 400 L water) "
            "or Dimethomorph 50% WP at 1.0 g/L combined with Mancozeb 75% WP at 2.0 g/L. "
            "Pre-Harvest Interval (PHI): 7 days mandatory waiting period before harvest. "
            "Spray precaution: Ensure uniform abaxial foliar coverage. Never apply during rain forecast or high winds (>18 km/h)."
        ),
        "chemical_dosage": {
            "active_ingredient": "Cymoxanil 8% + Mancozeb 64% WP",
            "dilution_dosage": "2.5 g/L of water (1.0 kg/acre)",
            "spray_volume_l_acre": 400,
            "phi_days": 7,
            "toxicity_class": "Class III (Slightly Hazardous)",
            "fungicide_type": "Curative Systemic + Protective Contact",
        },
        "source_doc_title": "ICAR - Central Potato Research Institute (CPRI) Technical Bulletin No. 42: Late Blight Management",
        "page_number": 14,
        "gov_url": "https://cpri.icar.gov.in/technical_bulletins/late_blight_cpri_tb42.pdf",
    },
    {
        "crop_name": "potato",
        "disease_or_pest": "late_blight",
        "growth_stage": "Vegetative & Foliar Emergence",
        "agro_climatic_zone": "All Potato Tracts",
        "raw_content": (
            "ICAR-CPRI Prophylactic Advisory for Potato Late Blight: Prior to canopy closure or upon initial regional IMD disease warnings, "
            "apply protective contact fungicide Mancozeb 75% WP at 2.5 g/L (1.0 kg/acre) or Chlorothalonil 75% WP at 2.0 g/L. "
            "Repeat at 7-10 day intervals if weather remains overcast and humid. "
            "Pre-Harvest Interval (PHI): 14 days for chlorothalonil. Avoid tank mixing with alkaline copper compounds."
        ),
        "chemical_dosage": {
            "active_ingredient": "Mancozeb 75% WP",
            "dilution_dosage": "2.5 g/L of water (1.0 kg/acre)",
            "spray_volume_l_acre": 400,
            "phi_days": 10,
            "toxicity_class": "Class III",
            "fungicide_type": "Protective Contact",
        },
        "source_doc_title": "ICAR-CPRI Potato Crop Protection Compendium (Standard Operating Procedures)",
        "page_number": 28,
        "gov_url": "https://cpri.icar.gov.in/publications/sop_protection_potato.pdf",
    },
    # --- POTATO: Early Blight (Alternaria solani) ---
    {
        "crop_name": "potato",
        "disease_or_pest": "early_blight",
        "growth_stage": "Mid-to-Late Tuber Formation",
        "agro_climatic_zone": "Northern & Eastern Plains",
        "raw_content": (
            "ICAR-CPRI Protocol for Early Blight (Alternaria solani) in Potato: Characterized by concentric necrotic target-board lesions "
            "on lower senescing leaves under warm days (24-30 deg C) and alternating wet-dry conditions. "
            "Recommended fungicide: Spray Azoxystrobin 18.2% + Difenoconazole 11.4% SC at 1.0 mL/L of water (400 mL/acre) "
            "or Chlorothalonil 75% WP at 2.0 g/L. "
            "Pre-Harvest Interval (PHI): 14 days before vine desiccation or harvest."
        ),
        "chemical_dosage": {
            "active_ingredient": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC",
            "dilution_dosage": "1.0 mL/L of water (400 mL/acre)",
            "spray_volume_l_acre": 400,
            "phi_days": 14,
            "toxicity_class": "Class II (Moderately Hazardous)",
            "fungicide_type": "Systemic Strobilurin + Triazole",
        },
        "source_doc_title": "ICAR-CPRI Technical Advisory on Alternaria Blights in Sub-Tropical Potato Cultivation",
        "page_number": 9,
        "gov_url": "https://cpri.icar.gov.in/bulletins/alternaria_potato_guidelines.pdf",
    },
    # --- RICE: Rice Blast (Magnaporthe oryzae) ---
    {
        "crop_name": "rice",
        "disease_or_pest": "rice_blast",
        "growth_stage": "Tillering to Panicle Emergence",
        "agro_climatic_zone": "Eastern & Southern Rice Ecosystems",
        "raw_content": (
            "ICAR-NRRI Cuttack Protocol for Rice Blast (Magnaporthe oryzae - Leaf and Neck Blast): Outbreaks occur with night dew, "
            "relative humidity >90%, and temperatures between 20-26 deg C with high nitrogen top-dressing. "
            "First-line intervention: Foliar spray of Tricyclazole 75% WP at 0.6 g/L of water (240 g/acre in 400 L water) "
            "or Isoprothiolane 40% EC at 1.5 mL/L of water. "
            "Pre-Harvest Interval (PHI): 21 days. Do not top-dress additional urea when active blast lesions are detected."
        ),
        "chemical_dosage": {
            "active_ingredient": "Tricyclazole 75% WP",
            "dilution_dosage": "0.6 g/L of water (240 g/acre)",
            "spray_volume_l_acre": 400,
            "phi_days": 21,
            "toxicity_class": "Class II",
            "fungicide_type": "Melanin Biosynthesis Inhibitor (Systemic)",
        },
        "source_doc_title": "ICAR-NRRI Cuttack Standard Rice Disease Management Guidelines (Bulletin 108)",
        "page_number": 33,
        "gov_url": "https://icar-nrri.gov.in/disease_bulletins/blast_control_nrri_b108.pdf",
    },
    # --- RICE: Bacterial Leaf Blight (Xanthomonas oryzae pv. oryzae) ---
    {
        "crop_name": "rice",
        "disease_or_pest": "bacterial_leaf_blight",
        "growth_stage": "Tillering to Booting Stage",
        "agro_climatic_zone": "All Major Irrigated Rice Tracts",
        "raw_content": (
            "ICAR-NRRI Cuttack Management of Bacterial Leaf Blight (BLB - Xanthomonas oryzae pv. oryzae): Characterized by wavy, "
            "water-soaked yellowish lesions progressing from leaf tips downward with milky bacterial ooze under humid rainy weather. "
            "Bactericidal treatment: Spray Streptocycline (Streptomycin sulphate 90% + Tetracycline hydrochloride 10%) at 6 g in 100 L water "
            "(approx 0.06 g/L, 100 ppm) tank-mixed with Copper Oxychloride 50% WP at 2.5 g/L of water. "
            "Pre-Harvest Interval (PHI): 15 days. Drain standing water from the field if excessive and suspend nitrogen fertilization."
        ),
        "chemical_dosage": {
            "active_ingredient": "Streptocycline 90:10 (6g/100L) + Copper Oxychloride 50% WP (2.5 g/L)",
            "dilution_dosage": "6g Streptocycline + 250g COC in 100 L water",
            "spray_volume_l_acre": 400,
            "phi_days": 15,
            "toxicity_class": "Class III",
            "fungicide_type": "Bactericide + Protective Contact Copper",
        },
        "source_doc_title": "ICAR-NRRI Bacterial Diseases of Rice: Diagnosis, Epidemiology and Chemical Containment",
        "page_number": 22,
        "gov_url": "https://icar-nrri.gov.in/bulletins/blb_management_nrri.pdf",
    },
    # --- WHEAT: Yellow Rust / Stripe Rust (Puccinia striiformis) ---
    {
        "crop_name": "wheat",
        "disease_or_pest": "yellow_rust",
        "growth_stage": "Vegetative to Flag Leaf Stage",
        "agro_climatic_zone": "North-Western Plains Zone (NWPZ) & Northern Hills",
        "raw_content": (
            "ICAR-IIWBR Karnal Advisory for Yellow Rust (Stripe Rust - Puccinia striiformis): Distinct yellow powdery pustules arranged in stripes "
            "on leaves favored by cool, moist weather (night temp 8-15 deg C, dew formation). "
            "Immediate eradication spray: Propiconazole 25% EC at 1.0 mL/L of water (200 mL/acre in 200 L water) or Tebuconazole 25.9% EC at 1.0 mL/L. "
            "One spray at initial yellow stripe appearance arrests disease spread. Pre-Harvest Interval (PHI): 30 days. "
            "Spray with flat-fan nozzles ensuring complete flag leaf interception."
        ),
        "chemical_dosage": {
            "active_ingredient": "Propiconazole 25% EC",
            "dilution_dosage": "1.0 mL/L of water (200 mL/acre)",
            "spray_volume_l_acre": 200,
            "phi_days": 30,
            "toxicity_class": "Class II",
            "fungicide_type": "Systemic Ergosterol Biosynthesis Inhibitor (Triazole)",
        },
        "source_doc_title": "ICAR-IIWBR Karnal Yellow Rust Containment Protocol in Wheat (Advisory Circular 51)",
        "page_number": 5,
        "gov_url": "https://iiwbr.icar.gov.in/wheat_protection/yellow_rust_circular51.pdf",
    },
    # --- WHEAT: Powdery Mildew (Blumeria graminis f. sp. tritici) ---
    {
        "crop_name": "wheat",
        "disease_or_pest": "powdery_mildew",
        "growth_stage": "Tillering and Stem Elongation",
        "agro_climatic_zone": "Himalayan Foothills & Northern Plains",
        "raw_content": (
            "ICAR-IIWBR Karnal Management of Powdery Mildew in Wheat: White cottony fungal patches on leaf sheaths and blades, "
            "prevalent under dense crop stands, high relative humidity (70-85%), and moderate temperatures (15-22 deg C). "
            "Recommended protocol: Foliar spray of Wettable Sulphur 80% WP at 3.0 g/L of water (1.2 kg/acre) or Triadimefon 25% WP at 1.0 g/L. "
            "Pre-Harvest Interval (PHI): 20 days. Ensure spray penetrates the dense lower canopy."
        ),
        "chemical_dosage": {
            "active_ingredient": "Wettable Sulphur 80% WP",
            "dilution_dosage": "3.0 g/L of water (1.2 kg/acre)",
            "spray_volume_l_acre": 400,
            "phi_days": 20,
            "toxicity_class": "Class IV (Practically Non-Toxic)",
            "fungicide_type": "Contact Inorganic Protectant",
        },
        "source_doc_title": "ICAR-IIWBR Manual of Wheat Fungal Diseases and Control Measures",
        "page_number": 19,
        "gov_url": "https://iiwbr.icar.gov.in/manuals/powdery_mildew_wheat.pdf",
    },
    # --- TOMATO: Tomato Leaf Curl Virus (ToLCV) & Whitefly Vector ---
    {
        "crop_name": "tomato",
        "disease_or_pest": "tomato_leaf_curl",
        "growth_stage": "Nursery to Flowering & Early Fruiting",
        "agro_climatic_zone": "Peninsular, Central and Northern Tomato Tracts",
        "raw_content": (
            "ICAR-IIHR Bangalore IPM Protocol for Tomato Leaf Curl Virus (ToLCV): Transmitted by the whitefly vector (Bemisia tabaci). "
            "Characterized by severe upward leaf curling, puckering, vein thickening, and stunting. "
            "Vector containment intervention: Foliar spray of Imidacloprid 17.8% SL at 0.3 mL/L of water (120 mL/acre) or "
            "Diafenthiuron 50% WP at 1.0 g/L. Install yellow sticky traps (15-20 per acre) at canopy level. "
            "Pre-Harvest Interval (PHI): 3 days for Imidacloprid on tomato. Rogue out infected seedlings immediately."
        ),
        "chemical_dosage": {
            "active_ingredient": "Imidacloprid 17.8% SL",
            "dilution_dosage": "0.3 mL/L of water (120 mL/acre)",
            "spray_volume_l_acre": 400,
            "phi_days": 3,
            "toxicity_class": "Class II",
            "fungicide_type": "Systemic Neonicotinoid Insecticide (Vector Control)",
        },
        "source_doc_title": "ICAR-IIHR Bangalore IPM Package of Practices for Protected and Field Tomato Cultivation",
        "page_number": 41,
        "gov_url": "https://iihr.res.in/ipm_packages/tomato_leaf_curl_iihr.pdf",
    },
    # --- TOMATO: Early Blight (Alternaria solani) ---
    {
        "crop_name": "tomato",
        "disease_or_pest": "early_blight",
        "growth_stage": "Vegetative and Fruit Development",
        "agro_climatic_zone": "All Tomato Growing Regions",
        "raw_content": (
            "ICAR-IIHR Bangalore Advisory for Tomato Early Blight (Alternaria solani): Produces brown necrotic concentric rings on leaves and "
            "sunken dark leathery cankers at stem bases and fruit calyx. "
            "Fungicidal spray: Mancozeb 75% WP at 2.5 g/L of water or Difenoconazole 25% EC at 1.0 mL/L. "
            "Apply at 10-14 day intervals upon disease onset. Pre-Harvest Interval (PHI): 7 days for Mancozeb on tomato."
        ),
        "chemical_dosage": {
            "active_ingredient": "Mancozeb 75% WP",
            "dilution_dosage": "2.5 g/L of water (1.0 kg/acre)",
            "spray_volume_l_acre": 400,
            "phi_days": 7,
            "toxicity_class": "Class III",
            "fungicide_type": "Protective Dithiocarbamate",
        },
        "source_doc_title": "ICAR-IIHR Horticultural Crop Disease Bulletin: Solanaceous Fungal Pathogens",
        "page_number": 16,
        "gov_url": "https://iihr.res.in/publications/early_blight_tomato.pdf",
    },
    # --- HEALTHY CROP MAINTENANCE PROTOCOLS (For Baseline Agronomic Guidance) ---
    {
        "crop_name": "potato",
        "disease_or_pest": "healthy",
        "growth_stage": "Tuber Initiation & Growth",
        "agro_climatic_zone": "Pan-India",
        "raw_content": (
            "ICAR-CPRI Good Agronomic Practices for Healthy Potato Crop: Maintain soil moisture between 65-75% field capacity. "
            "Do not allow standing water to prevent tuber rot and hypoxia. Apply recommended NPK 150:100:120 kg/ha in split doses. "
            "Scout weekly for aphid and whitefly vectors using yellow water pan traps. No fungicide application needed under low humidity (<70%)."
        ),
        "chemical_dosage": {
            "active_ingredient": "None (Preventive Agronomic Maintenance)",
            "dilution_dosage": "N/A",
            "spray_volume_l_acre": 0,
            "phi_days": 0,
            "toxicity_class": "N/A",
            "fungicide_type": "Cultural Management",
        },
        "source_doc_title": "ICAR-CPRI Good Agricultural Practices (GAP) for Potato Production in Sub-Tropics",
        "page_number": 2,
        "gov_url": "https://cpri.icar.gov.in/gap_potato_cpri.pdf",
    },
    {
        "crop_name": "rice",
        "disease_or_pest": "healthy",
        "growth_stage": "Vegetative Tillering",
        "agro_climatic_zone": "Pan-India",
        "raw_content": (
            "ICAR-NRRI Good Agronomic Practices for Healthy Paddy: Adopt Alternate Wetting and Drying (AWD) water management "
            "to save 25-30% irrigation water and reduce methane emissions. Maintain water depth of 2-5 cm during critical tillering. "
            "Apply balanced fertilizer using Leaf Color Chart (LCC) to avoid excessive vegetative growth and disease susceptibility."
        ),
        "chemical_dosage": {
            "active_ingredient": "None (Cultural AWD Management)",
            "dilution_dosage": "N/A",
            "spray_volume_l_acre": 0,
            "phi_days": 0,
            "toxicity_class": "N/A",
            "fungicide_type": "Water & Nutrient Management",
        },
        "source_doc_title": "ICAR-NRRI Rice Production Technology: Nutrient and Water Optimization",
        "page_number": 8,
        "gov_url": "https://icar-nrri.gov.in/gap_rice_water.pdf",
    },
    {
        "crop_name": "wheat",
        "disease_or_pest": "healthy",
        "growth_stage": "Crown Root Initiation & Jointing",
        "agro_climatic_zone": "Pan-India",
        "raw_content": (
            "ICAR-IIWBR Best Management Practices for Wheat: Ensure timely first irrigation at Crown Root Initiation (CRI) stage (21 days after sowing). "
            "Avoid excessive irrigation to prevent lodging and fungal root rots. Monitor microclimate; if canopy remains dry and humidity is low, "
            "fungicide application is unwarranted."
        ),
        "chemical_dosage": {
            "active_ingredient": "None (Agronomic Maintenance)",
            "dilution_dosage": "N/A",
            "spray_volume_l_acre": 0,
            "phi_days": 0,
            "toxicity_class": "N/A",
            "fungicide_type": "Irrigation Management",
        },
        "source_doc_title": "ICAR-IIWBR Wheat Package of Practices for High Yield and Resource Efficiency",
        "page_number": 4,
        "gov_url": "https://iiwbr.icar.gov.in/wheat_practices.pdf",
    },
    {
        "crop_name": "tomato",
        "disease_or_pest": "healthy",
        "growth_stage": "Vegetative and Flowering",
        "agro_climatic_zone": "Pan-India",
        "raw_content": (
            "ICAR-IIHR Tomato Cultural Practices: Stake plants with bamboo poles to keep foliage and fruit off wet soil, reducing fungal splash. "
            "Use drip irrigation with fertigation to minimize canopy wetting. Inspect weekly for leaf miner and thrips. "
            "No chemical sprays required under balanced crop health."
        ),
        "chemical_dosage": {
            "active_ingredient": "None (Cultural Staking & Drip Management)",
            "dilution_dosage": "N/A",
            "spray_volume_l_acre": 0,
            "phi_days": 0,
            "toxicity_class": "N/A",
            "fungicide_type": "Canopy & Trellis Architecture",
        },
        "source_doc_title": "ICAR-IIHR Tomato Cultivation Guide: Maximizing Quality & Disease Prevention",
        "page_number": 12,
        "gov_url": "https://iihr.res.in/tomato_cultivation_guide.pdf",
    },
]


def build_knowledge_base(
    db_path: str = DEFAULT_DB_PATH,
    faiss_path: str = DEFAULT_FAISS_PATH,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> None:
    """Seeds SQLite WAL database and builds FAISS IndexIDMap2 vector index."""
    print(f"[KnowledgeBase] Initializing SQLite database at: {db_path}")
    db = SQLiteKnowledgeManager(db_path)

    # Clean existing chunks to ensure clean deterministic seeding
    with db.get_connection() as conn:
        conn.execute("DELETE FROM icardocs_chunks;")
        conn.execute("DELETE FROM chunks_fts;")
        conn.commit()

    chunk_ids = []
    contents = []

    print(
        f"[KnowledgeBase] Inserting {len(ICAR_SEEDED_PROTOCOLS)} verified ICAR guidance chunks..."
    )
    for proto in ICAR_SEEDED_PROTOCOLS:
        cid = db.insert_chunk(
            crop_name=proto["crop_name"],
            disease_or_pest=proto["disease_or_pest"],
            raw_content=proto["raw_content"],
            chemical_dosage_json=proto["chemical_dosage"],
            source_doc_title=proto["source_doc_title"],
            page_number=proto["page_number"],
            gov_url=proto["gov_url"],
            growth_stage=proto.get("growth_stage", "All Stages"),
            agro_climatic_zone=proto.get("agro_climatic_zone", "Pan-India"),
        )
        chunk_ids.append(cid)
        contents.append(proto["raw_content"])

    print(f"[KnowledgeBase] Loading embedding model: {model_name}...")
    embedder = SentenceTransformer(model_name)

    print(
        f"[KnowledgeBase] Encoding {len(contents)} document chunks with L2 normalization..."
    )
    embeddings = embedder.encode(
        contents,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    assert dim == 384, f"Expected 384 dimensions from {model_name}, got {dim}"

    print(f"[KnowledgeBase] Creating FAISS IndexIDMap2(IndexFlatIP({dim}))...")
    sub_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(sub_index)

    ids_array = np.array(chunk_ids, dtype=np.int64)
    index.add_with_ids(embeddings, ids_array)

    os.makedirs(os.path.dirname(os.path.abspath(faiss_path)), exist_ok=True)
    faiss.write_index(index, faiss_path)
    print(
        f"[KnowledgeBase] Successfully indexed {index.ntotal} vectors to {faiss_path}"
    )


if __name__ == "__main__":
    build_knowledge_base()
