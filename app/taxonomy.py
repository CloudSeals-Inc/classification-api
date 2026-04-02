"""
IndicWaste Taxonomy — 16 Categories
MIBAKI IP · Project Swachh-AI
CO2e values derived from IPCC AR6 WG3 Chapter 7 + CPCB India Waste Management Report 2021
Severity: 1–10 environmental impact scale (10 = most harmful if landfilled)
"""

from dataclasses import dataclass

@dataclass
class WasteCategory:
    id: int
    code: str
    name: str
    recyclable: bool
    sub_type_route: str          # where this waste goes: SCRAP | COMPOST | RDF | HAZMAT | MEDICAL | INERT
    co2e_avoided_per_tonne: float  # kgCO2e avoided per tonne if diverted from landfill
    severity_score: int          # 1–10 (environmental harm if NOT diverted)
    inr_per_tonne_min: float     # minimum scrap/processing value ₹
    inr_per_tonne_max: float     # maximum scrap/processing value ₹
    token_base_rate: float       # base MIBA tokens issued per kg (before multipliers)
    description: str

INDICWASTE: dict[str, WasteCategory] = {
    "W01": WasteCategory(1,  "W01", "Organic / Food Waste",          False, "COMPOST",  550.0,   5, 200,   800,   0.8,  "Kitchen, food scraps, garden waste"),
    "W02": WasteCategory(2,  "W02", "Rigid Plastic (PET/HDPE)",       True,  "SCRAP",    1850.0,  8, 8000, 18000,  2.2,  "Bottles, containers, jerricans"),
    "W03": WasteCategory(3,  "W03", "Flexible Plastic / Film",        True,  "SCRAP",    1200.0,  9, 3000,  8000,  1.8,  "Bags, wrappers, pouches, sachets"),
    "W04": WasteCategory(4,  "W04", "Metal (Ferrous)",                True,  "SCRAP",    2100.0,  6, 15000, 35000, 2.8,  "Steel, iron cans, containers"),
    "W05": WasteCategory(5,  "W05", "Metal (Non-ferrous / Aluminium)",True,  "SCRAP",    9200.0,  7, 60000,120000, 5.0,  "Aluminium cans, copper wire, brass"),
    "W06": WasteCategory(6,  "W06", "Paper / Cardboard",              True,  "SCRAP",    900.0,   4, 4000, 12000,  1.2,  "Newspapers, cardboard, office paper"),
    "W07": WasteCategory(7,  "W07", "Glass",                          True,  "SCRAP",    300.0,   3, 1000,  4000,  0.6,  "Bottles, jars, flat glass"),
    "W08": WasteCategory(8,  "W08", "Electronic Waste (E-waste)",     True,  "HAZMAT",   4800.0,  10, 30000, 80000, 4.5,  "Phones, PCBs, batteries, cables"),
    "W09": WasteCategory(9,  "W09", "Hazardous / Chemical",           False, "HAZMAT",   6200.0,  10, 0,     0,     0.0,  "Paints, solvents, pesticides"),
    "W10": WasteCategory(10, "W10", "Textile / Clothing",             True,  "SCRAP",    1100.0,  6, 2000,  8000,  1.4,  "Old clothes, fabric, footwear"),
    "W11": WasteCategory(11, "W11", "Rubber / Tyres",                 True,  "RDF",      1400.0,  7, 3000, 10000,  1.6,  "Tyres, rubber products"),
    "W12": WasteCategory(12, "W12", "Construction & Demolition",      False, "INERT",    180.0,   3, 500,   2000,  0.3,  "Concrete, bricks, debris"),
    "W13": WasteCategory(13, "W13", "Medical / Biomedical",           False, "MEDICAL",  3200.0,  10, 0,     0,     0.0,  "Syringes, bandages, pharma waste"),
    "W14": WasteCategory(14, "W14", "Sanitary / Hygiene",             False, "RDF",      680.0,   8, 0,     500,   0.5,  "Diapers, sanitary pads, wipes"),
    "W15": WasteCategory(15, "W15", "Inert / Ash / Soil",             False, "INERT",    90.0,    2, 0,     300,   0.1,  "Ash, dirt, ceramic, stone"),
    "W16": WasteCategory(16, "W16", "Mixed / Unclassified",           False, "RDF",      420.0,   5, 0,     1000,  0.4,  "Multi-material, unsorted waste"),
}

TOKEN_INR_RATES: dict[str, float] = {
    "W01": 0.5,  "W02": 8.0,  "W03": 5.0,  "W04": 12.0, "W05": 50.0,
    "W06": 3.0,  "W07": 1.5,  "W08": 20.0, "W09": 0.0,  "W10": 4.0,
    "W11": 5.0,  "W12": 0.8,  "W13": 0.0,  "W14": 1.0,  "W15": 0.2,  "W16": 1.5,
}

CATEGORY_CODES = list(INDICWASTE.keys())
CATEGORY_NAMES = [v.name for v in INDICWASTE.values()]
RECYCLABLE_CODES = [k for k, v in INDICWASTE.items() if v.recyclable]
HAZARDOUS_CODES = ["W09", "W13"]

def get_by_name(name: str) -> WasteCategory | None:
    for cat in INDICWASTE.values():
        if cat.name.lower() in name.lower():
            return cat
    return None

def get_by_code(code: str) -> WasteCategory | None:
    return INDICWASTE.get(code.upper())
