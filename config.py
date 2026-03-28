# 14 NIH ChestX-ray14 conditions + 10 extended conditions + "no finding" (25 total)
CONDITIONS = [
    # NIH ChestX-ray14
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural effusion",
    "emphysema",
    "fibrosis",
    "hernia",
    "infiltration",
    "mass",
    "nodule",
    "pleural thickening",
    "pneumonia",
    "pneumothorax",
    # Extended conditions
    "tuberculosis",
    "lung abscess",
    "heart failure",
    "aortic aneurysm",
    "pericardial effusion",
    "rib fractures",
    "spine abnormalities",
    "scoliosis",
    "mediastinal mass",
    "hilar enlargement",
    # Normal
    "no finding",
]

# Text prompts fed to BiomedCLIP for zero-shot classification
CONDITION_PROMPTS = {
    # NIH ChestX-ray14
    "atelectasis":        "chest x-ray showing atelectasis with partial lung collapse and volume loss",
    "cardiomegaly":       "chest x-ray showing cardiomegaly with enlarged heart silhouette",
    "consolidation":      "chest x-ray showing lung consolidation with airspace opacity",
    "edema":              "chest x-ray showing pulmonary edema with bilateral interstitial opacities",
    "pleural effusion":   "chest x-ray showing pleural effusion with blunting of costophrenic angle",
    "emphysema":          "chest x-ray showing emphysema with hyperinflation and flattened diaphragm",
    "fibrosis":           "chest x-ray showing pulmonary fibrosis with reticular interstitial markings",
    "hernia":             "chest x-ray showing diaphragmatic hernia with bowel in chest cavity",
    "infiltration":       "chest x-ray showing lung infiltrates with patchy airspace opacification",
    "mass":               "chest x-ray showing a pulmonary mass or large nodule",
    "nodule":             "chest x-ray showing a pulmonary nodule or small rounded opacity",
    "pleural thickening": "chest x-ray showing pleural thickening along the chest wall",
    "pneumonia":          "chest x-ray showing pneumonia with lobar or segmental consolidation",
    "pneumothorax":       "chest x-ray showing pneumothorax with collapsed lung and pleural air line",
    # Extended conditions
    "tuberculosis":       "chest x-ray showing tuberculosis with upper lobe cavitation and infiltrates",
    "lung abscess":       "chest x-ray showing lung abscess with cavitary lesion and air-fluid level",
    "heart failure":      "chest x-ray showing congestive heart failure with cardiomegaly and pulmonary vascular congestion",
    "aortic aneurysm":    "chest x-ray showing aortic aneurysm with widened mediastinum and enlarged aortic knob",
    "pericardial effusion": "chest x-ray showing pericardial effusion with globular enlarged cardiac silhouette",
    "rib fractures":      "chest x-ray showing rib fractures with cortical disruption of ribs",
    "spine abnormalities": "chest x-ray showing vertebral abnormalities or compression fractures of the spine",
    "scoliosis":          "chest x-ray showing scoliosis with lateral curvature of the spine",
    "mediastinal mass":   "chest x-ray showing mediastinal widening or mass in the mediastinum",
    "hilar enlargement":  "chest x-ray showing hilar enlargement with prominent bilateral hilar shadows",
    # Normal
    "no finding":         "normal chest x-ray with clear lungs and no significant abnormalities",
}

# BiomedCLIP model (Microsoft, trained on PubMed + MIMIC)
BIOMED_CLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# BLIP VQA model (CPU-friendly base model)
BLIP_VQA_MODEL = "Salesforce/blip-vqa-base"

# Confidence threshold — with 15 conditions the score is more spread out,
# so we use a slightly lower threshold than the 6-condition setup
CONFIDENCE_THRESHOLD = 0.2
