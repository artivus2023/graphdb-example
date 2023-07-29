from transformers import pipeline

triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')

# Function to parse the generated text and extract the triplets
def extract_triplets(text):
    extracted_text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(
        text, return_tensors=True, return_text=False)[0]["generated_token_ids"]]
    )
    # extracted_triplets = extract_triplets(extracted_text[0])
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    extracted_text = extracted_text[0].strip()
    current = 'x'
    for token in extracted_text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets