import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Download required resources
nltk.download("punkt")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("averaged_perceptron_tagger")

piano_class_text = (
    "The future of India will be shaped by today’s younger generation who need quality education through digital literacy, making them productive and self-reliant citizens."
    "'Digital literacy' is the skills required to achieve digital competence and use of Information and Communication Technology (ICT) for work, leisure, learning and communication."
    "It does not replace traditional forms of literacy, instead complements and amplifies the skills that form the foundation of traditional forms."
    "Learners can benefit from the knowledgebase and experience of 4 decades of Infosys as an enterprise."
    "We also bring the quality content from our partners and leading universities across the world."
    "Content is aligned with New Education policy 2020."
    "In Mayfair or the City of London and has world-class piano instructors."
    "Includes soft skills and vocational skills."
    "Infosys Springboard is a Digital literacy program launched as part of the Infosys ESG Tech for Good charter."
    "It aims to enable students and associated communities from early education to lifelong learners by imparting digital life skills through curated content & interventions, free of cost."
)

# Step 1: Tokenize
tokens = word_tokenize(piano_class_text)

# Step 2: POS tagging
pos_tags = pos_tag(tokens)

# Step 3: Named Entity Recognition
ne_tree = ne_chunk(pos_tags)

# Extract entities
def extract_entities(tree):
    entities = []
    for subtree in tree:
        if isinstance(subtree, Tree):  # If subtree is a named entity
            entity = " ".join([token for token, pos in subtree.leaves()])
            entity_type = subtree.label()
            entities.append((entity, entity_type))
    return entities

entities = extract_entities(ne_tree)

# Print extracted entities
for entity, label in entities:
    print(f"Entity: {entity}, Label: {label}")

# Optional: Visualize the NE tree
ne_tree.draw()
