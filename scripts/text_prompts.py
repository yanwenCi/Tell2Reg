import random

# Lists of subjects, actions, and descriptors to create diverse prompts
subjects = [
    "A dog", "A cat", "A tree", "A car", "A building", "A person", "A bird",
    "A plane", "A boat", "A river", "A mountain", "A beach", "A forest", 
    "A street", "A bridge", "A bike", "A laptop", "A smartphone", "A book",
    "A table", "A chair", "A cup", "A bottle", "A flower", "A fruit", "A vegetable",
    "A fish", "A horse", "A cow", "A sheep", "A goat", "A chicken", "A bus",
    "A train", "A truck", "A motorcycle", "A skateboard", "A snowboard", "A skier",
    "A swimmer", "A runner", "A cyclist", "A basketball player", "A soccer player",
    "A tennis player", "A doctor", "A nurse", "A teacher", "A student", "A chef",
    "A waiter", "A pilot", "A firefighter", "A policeman", "A soldier", "A singer",
    "A dancer", "A musician", "A painter", "A writer", "A scientist", "A programmer",
    "A gamer", "A youtuber", "A blogger", "A tourist", "A photographer"
]

actions = [
    "sitting on a bench", "standing in the park", "running on the street", "jumping over a hurdle",
    "flying in the sky", "swimming in the pool", "riding a bike", "driving a car", "reading a book",
    "writing on a notebook", "talking on the phone", "taking a photo", "cooking in the kitchen",
    "eating at a restaurant", "shopping in a mall", "playing with a ball", "holding an umbrella",
    "watering the plants", "playing a guitar", "singing a song", "dancing on stage",
    "painting a picture", "drawing on paper", "coding on a laptop", "testing an experiment",
    "teaching a class", "studying for exams", "running a marathon", "playing chess",
    "surfing on waves", "skating on ice", "skiing down a slope", "hiking up a mountain",
    "fishing in a lake", "camping in the woods", "sunbathing on the beach", "walking a dog",
    "feeding birds", "throwing a frisbee", "building a sandcastle", "catching a train",
    "boarding a plane", "waiting for a bus", "selling vegetables", "buying groceries",
    "carrying a backpack", "holding a baby", "drinking coffee", "sipping tea", "brushing hair",
    "wearing a hat", "putting on gloves", "tying shoes", "wearing a scarf", "checking time",
    "playing video games", "watching TV", "sitting on a couch", "talking to friends",
    "laughing at a joke", "smiling for a photo"
]

descriptors = [
    "on a sunny day", "in the evening", "during the night", "in a crowded place", "in a quiet area",
    "near a river", "by the sea", "in the mountains", "under a tree", "next to a car", "in front of a building",
    "beside a lake", "on a hill", "in a forest", "on the beach", "in a park", "at home", "at school",
    "at work", "in a city", "in a village", "in a small town", "in a large city", "at the mall",
    "at the airport", "in a station", "in a classroom", "in an office", "in a kitchen", "in a living room",
    "in a bedroom", "in a bathroom", "in a garden", "in a backyard", "in a field", "on a farm", "on a boat",
    "on a train", "on a plane", "on a bus", "in a car", "on a motorcycle", "on a bicycle", "on a skateboard",
    "on a surfboard", "on a snowboard", "on skis", "on a hike", "on a run", "on a walk", "at a concert",
    "at a festival", "at a party", "at a wedding", "at a funeral", "at a meeting", "at a conference",
    "at a workshop", "at a seminar", "in a museum", "in a gallery", "in a theater", "in a cinema",
    "in a library", "in a bookstore"
]



# def generate_prompts():
#     subject = random.choice(subjects)
#     action = random.choice(actions)
#     descriptor = random.choice(descriptors)
#     prompt = f"{subject} {action} {descriptor}."
#     return prompt   

# List of terms related to prostate lesions
terms = [
    "prostate lesion", "prostate cancer", "biopsy", "MRI scan", "ultrasound",
    "Gleason score", "PSA level", "prostatic intraepithelial neoplasia", 
    "adenocarcinoma", "tumor", "malignant", "benign", "screening", 
    "prostatectomy", "radiation therapy", "hormone therapy", "active surveillance", 
    "watchful waiting", "localized", "metastatic", "digital rectal exam", 
    "urinary symptoms", "hematuria", "nocturia", "bone scan", "CT scan", 
    "MRI-guided biopsy", "transrectal ultrasound", "multiparametric MRI", 
    "histopathology", "tumor staging", "prostate-specific antigen", 
    "androgen deprivation therapy", "chemotherapy", "cryotherapy", "HIFU", 
    "robot-assisted surgery", "nerve-sparing surgery", "quality of life", 
    "survival rate", "recurrence", "follow-up", "immunotherapy", "genetic testing", 
    "family history", "diet and prostate cancer", "risk factors", "prevention", 
    "early detection", "treatment options", "clinical trials", "advanced prostate cancer", 
    "localized prostate cancer", "PSMA PET scan", "castration-resistant prostate cancer", 
    "prostate health", "BPH", "enlarged prostate", "inflammation", "prostatitis", 
    "active monitoring", "MRI fusion biopsy", "high-grade PIN", "atypical small acinar proliferation",
    "surgical margins", "pathology report", "urooncology", "precision medicine", 
    "targeted therapy", "pain management", "sexual health", "erectile dysfunction", 
    "urinary incontinence", "pelvic floor exercises", "kegel exercises", 
    "radiation side effects", "hormone therapy side effects", "PSA doubling time", 
    "biomarkers", "biochemical recurrence", "cancer stem cells", "immune response", 
    "genomic analysis", "biopsy results", "MRI findings", "tumor progression", 
    "clinical symptoms", "diagnosis", "treatment planning", "second opinion", 
    "patient support", "oncology consultation", "radiologist report", "pathologist findings",
    "medical imaging", "urology", "oncologist", "treatment guidelines", "healthcare provider"
]

# # Function to generate text prompts
# def generate_prompts():
#     term1 = random.choice(terms)
#     term2 = random.choice(terms)
#     term3 = random.choice(terms)
#     while term2 == term1:
#         term2 = random.choice(terms)
#     prompts=f"{term3} in {term1} on {term2}."
#     return prompts


def generate_prompts():
    organ_positions = [
        "at the top of the image",
        "at the bottom of the image",
        "on the left side of the image",
        "on the right side of the image",
        "in the center of the image",
        "towards the top left corner",
        "towards the top right corner",
        "towards the bottom left corner",
        "towards the bottom right corner"
    ]
    
    organ_shapes = [
        "in a circular shape",
        "in an oval shape",
        "in an irregular shape",
        "in a rectangular shape",
        "with a smooth border",
        "with a rough border"
    ]
    
    organs = [
        "bladder",
        "liver",
        "spleen",
        "pancreas",
        "kidney",
        "gallbladder",
        "intestines",
        "uterus",
        "ovaries",
        "prostate"
    ]
    
    conditions = [
        "showing signs of inflammation",
        "with no visible abnormalities",
        "displaying abnormal growth",
        "indicating possible cysts",
        "with a noticeable mass",
        "showing signs of obstruction",
        "with fluid accumulation",
        "indicating potential stones",
        "with irregular texture",
        "appearing enlarged"
    ]

    prompt_templates = [
        "The {organ} is located {position}, {shape}, {condition}.",
        "Observe the {organ} {position}, {shape}, {condition}.",
        "In this MRI, the {organ} can be seen {position}, {shape}, {condition}.",
        "The {organ}, which is {position}, appears {shape} and is {condition}.",
        "Located {position}, the {organ} is {shape} and {condition}."
    ]
    
    organ = random.choice(organs)
    position = random.choice(organ_positions)
    shape = random.choice(organ_shapes)
    condition = random.choice(conditions)
    prompt_template = random.choice(prompt_templates)
    
    prompt1 = prompt_template.format(organ=organ, position=position, shape=shape, condition=condition)
    prompt2 = prompt_template.format(organ=organ, position=position, shape=shape, condition=condition)
    prompt3 = prompt_template.format(organ=organ, position=position, shape=shape, condition=condition)
    return prompt1+" "+prompt2+" "+prompt3


# Function to generate random combinations of descriptions for a pancreas MRI image
def generate_random_combination_prompts():
    descriptions = {
        'Pancreas': [
            "The pancreas resembles a slender fish swimming horizontally.",
            "It appears like a smooth lake with medium intensity.",
            "The pancreatic head is a rounded pebble nestled against the duodenum.",
            "The pancreas is shaped like a banana, lying in the middle of the image.",
            "It looks like a delicate ribbon with subtle variations in texture."
        ],
        'Bladder': [
            "The bladder is a perfectly round ball with a bright top.",
            "It looks like a polished marble, well-defined and smooth.",
            "The bladder resembles a glowing orb in the pelvic cavity.",
            "It appears like a buoy floating, full and round.",
            "The bladder is a smooth, shiny fruit, perfectly shaped."
        ],
        'Prostate': [
            "The prostate resembles a cluster of chestnuts in the center.",
            "It looks like a smooth walnut, normal in size and shape.",
            "The prostate is akin to a small acorn, well-defined.",
            "It appears as a rounded mass, slightly larger than a marble.",
            "The prostate is shaped like a chocolate truffle, soft and compact."
        ],
        'Rectum': [
            "The rectum looks like a gently curving tube.",
            "It resembles a smooth, flexible straw.",
            "The rectum is akin to a winding path through a garden.",
            "It appears as a clear channel, unobstructed and open.",
            "The rectum is shaped like a subtle archway, inviting and clear."
        ],
        'Bone': [
            "The bones resemble sturdy tree branches, well-defined.",
            "They look like strong pillars, intact and supportive.",
            "The bone structure appears like solid rock, unyielding.",
            "They resemble the ribs of a ship, strong and well-formed.",
            "The bones look like ancient columns, holding the structure."
        ],
        'Muscles': [
            "The muscles are like rippling waves, flowing and dynamic.",
            "They resemble taut ropes, strong and flexible.",
            "The muscle fibers appear like interwoven threads, resilient.",
            "They look like smooth bands of silk, stretching and contracting.",
            "The muscles are akin to coiled springs, ready to move."
        ]
    }
    
    # Randomly select one description from each structure
    selected_descriptions = {structure: random.choice(desc) for structure, desc in descriptions.items()}
    
    # Create a random combination prompt
    combination_prompt = "Random MRI Descriptions:\n"
    for structure, description in selected_descriptions.items():
        combination_prompt += f"{structure}: {description}\n"
    
    return combination_prompt

    # Generate and print the random combination descriptions
    # random_combination = generate_random_combination_prompts()
# print(random_combination)
