import spacy
import base64
from spacy.tokens import Span
from spacy import displacy
import medspacy
from medspacy.preprocess import PreprocessingRule, Preprocessor
from medspacy.ner import TargetRule
from medspacy.context import ConTextItem
from medspacy.section_detection import Sectionizer
from medspacy.postprocess import PostprocessingRule, PostprocessingPattern, Postprocessor
from medspacy.postprocess import postprocessing_functions
from medspacy.visualization import visualize_ent, visualize_dep

import re
import streamlit as st

def visualize_ent(doc, context=True, sections=True, colors=None):

    ents_data = []

    for target in doc.ents:
        ent_data = {
            "start": target.start_char,
            "end": target.end_char,
            "label": target.label_.upper(),
        }
        ents_data.append((ent_data, "ent"))

    if context:
        visualized_modifiers = set()
        for target in doc.ents:
            for modifier in target._.modifiers:
                if modifier in visualized_modifiers:
                    continue
                ent_data = {
                    "start": modifier.span.start_char,
                    "end": modifier.span.end_char,
                    "label": modifier.category,
                }
                ents_data.append((ent_data, "modifier"))
                visualized_modifiers.add(modifier)
    if sections:
        for section_tup in doc._.sections:
            title, header = section_tup[:2]
            if title is None:
                continue
            ent_data = {
                "start": header.start_char,
                "end": header.end_char,
                "label": f"<< {title.upper()} >>",
            }
            ents_data.append((ent_data, "section"))
    if len(ents_data) == 0:  # No data to display
        viz_data = [{"text": doc.text, "ents": []}]
        st.write("ents_data is null")
        options = dict()
    else:
        ents_data = sorted(ents_data, key=lambda x: x[0]["start"])

        # If colors aren't defined, generate color mappings for each entity
        # and modifier label and set all section titles to a light gray
        if colors is None:
            labels = set()
            section_titles = set()
            for (ent_data, ent_type) in ents_data:
                if ent_type in ("ent", "modifier"):
                    labels.add(ent_data["label"])
                elif ent_type == "section":
                    section_titles.add(ent_data["label"])
            colors = _create_color_mapping(labels)
            for title in section_titles:
                colors[title] = "#dee0e3"
        ents_display_data, _ = zip(*ents_data)
        viz_data = [{"text": doc.text, "ents": ents_display_data,}]
        #print(viz_data)
        options = {
            "colors": colors,
        }
    return displacy.render(
        viz_data, style="ent", page = True, manual = True,options=options 
    )

def _create_color_mapping(labels):
    mapping = {}
    color_cycle = _create_color_generator()
    for label in labels:
        if label not in mapping:
            mapping[label] = next(color_cycle)
    return mapping
def _create_color_generator():
    """Create a generator which will cycle through a list of
    default matplotlib colors"""
    from itertools import cycle

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return cycle(colors)

def visualize_dep(doc):
    """Create a dependency-style visualization for
    ConText targets and modifiers in doc. This will show the
    relationships between entities in doc and contextual modifiers.
    """
    token_data = []
    token_data_mapping = {}
    for token in doc:
        data = {"text": token.text, "tag": "", "index": token.i}
        token_data.append(data)
        token_data_mapping[token] = data

    # Merge phrases
    targets_and_modifiers = [*doc._.context_graph.targets]
    targets_and_modifiers += [
        mod.span for mod in doc._.context_graph.modifiers
    ]
    for span in targets_and_modifiers:
        first_token = span[0]
        data = token_data_mapping[first_token]
        data["tag"] = span.label_

        if len(span) == 1:
            continue

        idx = data["index"]
        for other_token in span[1:]:
            # Add the text to the display data for the first word
            # and remove the subsequent token
            data["text"] += " " + other_token.text
            # Remove this token from the list of display data
            token_data.pop(idx + 1)

        # Lower the index of the following tokens
        for other_data in token_data[idx + 1 :]:
            other_data["index"] -= len(span) - 1

    dep_data = {"words": token_data, "arcs": []}
    # Gather the edges between targets and modifiers
    for target, modifier in doc._.context_graph.edges:
        target_data = token_data_mapping[target[0]]
        modifier_data = token_data_mapping[modifier.span[0]]
        dep_data["arcs"].append(
            {
                "start": min(target_data["index"], modifier_data["index"]),
                "end": max(target_data["index"], modifier_data["index"]),
                "label": modifier.category,
                "dir": "right" if target > modifier.span else "left",
            }
        )
    return displacy.render(dep_data, manual=True)

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

st.header("MedSpaCy")
from PIL import Image
image = Image.open('medspacy.jpg')
st.sidebar.image(image, caption = "MedSpaCy")
nlp = medspacy.load("en_info_3700_i2b2_2012",disable = "target_matcher")
#Processing Text
ner = nlp.get_pipe('ner')
DEFAULT_TEXT = "Write Your Text Here"
text = st.text_area("Text to analyze", DEFAULT_TEXT, height=200)
doc = nlp(text)

#st.write(doc)


#Preprocessing
preprocessor = Preprocessor(nlp.tokenizer)
nlp.tokenizer = preprocessor
preprocess_rules = [
	PreprocessingRule(
        re.compile("dx'd"), repl="Diagnosed", 
                  desc="Replace abbreviation"
    ),
    
    PreprocessingRule(
        re.compile("tx'd"), repl="Treated", 
                  desc="Replace abbreviation"
    ),]
preprocessor.add(preprocess_rules)

#ConText
context = nlp.get_pipe('context')
item_data = [
ConTextItem("diagnosed in <YEAR>", "HISTORICAL", 
	pattern=[
	{"LOWER": "diagnosed"},
	{"LOWER": "in"},
	{"LOWER": {"REGEX": "(19|20)\d\d"}}
	])
]
context.add(item_data)

#Section Detection
sectionizer = Sectionizer(nlp, patterns = "default")
nlp.add_pipe(sectionizer)

#Post Processing
postprocessor = Postprocessor(debug=False)
nlp.add_pipe(postprocessor)
postprocess_rules = [
    PostprocessingRule(
        patterns=[
            PostprocessingPattern(condition=lambda ent: ent.lower_ == "married"),
        ],
        action=postprocessing_functions.remove_ent,
        description="Remove a specific misclassified span of text."
    ),
    
]
postprocessor.add(postprocess_rules)


doc = nlp(text)
radio = st.sidebar.radio("Select",["ent","dep"])
if radio == "ent":
	obj1 = visualize_ent(doc)
	st.markdown(str(obj1),unsafe_allow_html = True)
else:
	from medspacy.sentence_splitting import PyRuSHSentencizer
	sentencizer = PyRuSHSentencizer(rules_path="rush_rules.tsv")
	name,component = nlp.remove_pipe('sentencizer')	
	nlp.add_pipe(sentencizer,before ='parser')
	doc = nlp(text)
	for sent in doc.sents:
		doc = nlp(sent.text)
		obj2 = visualize_dep(doc)
		render_svg(obj2)


