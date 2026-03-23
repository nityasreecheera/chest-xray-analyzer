CONDITIONS = [
    "pneumonia",
    "pleural effusion",
    "cardiomegaly",
    "atelectasis",
    "pneumothorax",
    "no finding",
]

# Text prompts fed to BiomedCLIP for zero-shot classification
CONDITION_PROMPTS = {
    "pneumonia": "chest x-ray showing pneumonia with lung consolidation and opacity",
    "pleural effusion": "chest x-ray showing pleural effusion with fluid accumulation",
    "cardiomegaly": "chest x-ray showing cardiomegaly with enlarged heart",
    "atelectasis": "chest x-ray showing atelectasis with partial lung collapse",
    "pneumothorax": "chest x-ray showing pneumothorax with collapsed lung and air in pleural space",
    "no finding": "normal chest x-ray with no significant abnormalities",
}

# BiomedCLIP model (Microsoft, trained on PubMed + MIMIC)
BIOMED_CLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# BLIP-2 model
BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"

# Confidence threshold below which we flag uncertainty
CONFIDENCE_THRESHOLD = 0.3
