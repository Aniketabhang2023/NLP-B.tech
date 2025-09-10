import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

piano_class_text = (
    "The future of India will be shaped by today’s younger generation who need quality education through digital literacy, making them productive and self-reliant citizens."
    "'Digital literacy' is the skills required to achieve digital competence and use of Information and Communication Technology (ICT) for work, leisure, learning and communication."
    "It does not replace traditional forms of literacy, instead complements and amplifies the skills that form the foundation of traditional forms."
    "Learners can benefit from the knowledgebase and experience of 4 decades of Infosys as an enterprise."
    "We also bring the quality content from our partners and leading universities across the world."
    "Content is aligned with New Education policy 2020." 
    " In Mayfair or the City of London and has"
    " world-class piano instructors."
    "Includes soft skills and vocational skills."
    "Infosys Springboard is a Digital literacy program launched as part of the Infosys ESG Tech for Good charter." 
    "It aims to enable students and associated communities from early education to lifelong learners by imparting digital"
    "life skills through curated content & interventions, free of cost."
    
)
piano_class_doc = nlp(piano_class_text)

# Show visualization in browser
displacy.serve(piano_class_doc, style="ent", auto_select_port=True)


# Print entity details
for ent in piano_class_doc.ents:
    print(
        f"""
{ent.text = }
{ent.start_char = }
{ent.end_char = }
{ent.label_ = }
spacy.explain('{ent.label_}') = {spacy.explain(ent.label_)}"""
    )