# Awesome Ancient Greek NLP

A curated list of awesome Ancient Greek language processing software, models, datasets, and resources. Inspired by [awesome-buryat-nlp](https://github.com/donothinger/awesome-buryat-nlp) and [awesome-kyrgyz-nlp](https://github.com/alexeyev/awesome-kyrgyz-nlp).

The main focus is on **open source** tools, **downloadable** data, and research **papers with code**.

If you want to contribute to this list (please do), send a pull request. Also, a listed repository should be tagged as deprecated if:
- Repository owners explicitly say that "this library is not maintained"
- Not committed to for a long time (2-3 years)

## Table of Contents

- [Datasets](#datasets)
  - [Corpora](#corpora)
  - [Treebanks (Annotated Corpora)](#treebanks-annotated-corpora)
  - [Parallel Corpora](#parallel-corpora)
  - [Named Entity Recognition](#named-entity-recognition)
  - [Text Classification](#text-classification)
  - [Question Answering](#question-answering)
  - [Machine-readable Dictionaries and Lexical Resources](#machine-readable-dictionaries-and-lexical-resources)
  - [Papyri and Manuscripts](#papyri-and-manuscripts)
- [Pretrained Models](#pretrained-models)
  - [Language Models](#language-models)
  - [Word Embeddings](#word-embeddings)
  - [Specialized Models](#specialized-models)
- [Methods/Software](#methodssoftware)
  - [NLP Pipelines](#nlp-pipelines)
  - [Morphological Analysis](#morphological-analysis)
  - [Syntactic Parsing](#syntactic-parsing)
  - [Text Classification and Genre Detection](#text-classification-and-genre-detection)
  - [OCR and HTR](#ocr-and-htr)
  - [Other Tools](#other-tools)
- [Script Types and Orthography](#script-types-and-orthography)
- [Online Demos and Resources](#online-demos-and-resources)
- [Communities and Research Groups](#communities-and-research-groups)

---

## Datasets

### Corpora

#### Unannotated Corpora
- **[Perseus Digital Library](https://www.perseus.tufts.edu/)** - One of the oldest and most comprehensive digital libraries for Ancient Greek and Latin texts. Contains extensive Greek corpus from Homer to post-Classical period. CC licensed.
- **[Thesaurus Linguae Graecae (TLG)](https://stephanus.tlg.uci.edu/)** - The most comprehensive digital corpus of Ancient Greek literature from Homer (8th century BC) to the 19th century. Founded in 1972. Contains virtually all surviving Greek texts. Subscription-based.
- **[First1KGreek](https://opengreekandlatin.github.io/First1KGreek/)** - Part of the Open Greek and Latin Project. Openly licensed Ancient Greek texts.
- **[Diorisis Ancient Greek Corpus](https://figshare.com/articles/dataset/The_Diorisis_Ancient_Greek_Corpus/6187256)** - Large-scale corpus for computational analysis and semantic change studies. Available on Figshare.
- **[GLAUx (Greek Language Automated)](https://github.com/alekkeersmaekers/glaux)** - Large corpus of 20M+ tokens covering Ancient Greek from 8th century BC to roughly 4th century AD.

#### Annotated Corpora (Publication-Ready)
Most treebanks listed below are publication-ready with proper citations and licenses.

### Treebanks (Annotated Corpora)

#### Dependency Treebanks
- **[Ancient Greek and Latin Dependency Treebank (AGLDT)](http://perseusdl.github.io/treebank_data/)** - The earliest and largest dependency treebank for Ancient Greek (~550K tokens). Started at Tufts University in 2006. Includes Archaic poetry, Classical poetry and prose. Morphological and syntactic annotation. [GitHub](https://github.com/PerseusDL/treebank_data) | License: CC BY-SA 3.0
  
- **[Universal Dependencies - Ancient Greek Perseus](https://github.com/UniversalDependencies/UD_Ancient_Greek-Perseus)** - Automatic conversion of AGLDT to UD format. Part of Universal Dependencies project.

- **[Universal Dependencies - Ancient Greek PROIEL](https://github.com/UniversalDependencies/UD_Ancient_Greek-PROIEL)** - Part of the PROIEL (Pragmatic Resources in Old Indo-European Languages) treebank. Focuses on New Testament Greek and other early texts. UD-compliant.

- **[PROIEL Treebank](https://proiel.github.io/)** - Ancient Indo-European parallel treebank including Ancient Greek (New Testament, Herodotus). Morphological and syntactic annotation.

- **[Pedalion Project Treebanks](https://perseids.org/)** - Expanding Ancient Greek treebank data through the Perseids collaborative annotation platform. Compatible with Arethusa editor.

- **[PapyGreek Treebanks](https://papygreek.com/)** - Dataset of linguistically annotated Greek documentary papyri. Unique focus on non-literary Greek.

#### Annotation Tools
- **[Arethusa](https://www.perseids.org/tools/arethusa/app/#/)** - Web-based collaborative treebanking environment for Ancient Greek and Latin. Supports morphology, syntax, and advanced syntax annotation.

- **[Alpheios](https://alpheios.net/)** - Local treebanking environment and reading tools for Greek and Latin.

### Parallel Corpora

- **[Ancient Greek-English Parallel Corpus](https://jcls.io/article/id/100/)** - Aligned translations for Ancient Greek texts with English. Used for translation studies and comparative analysis.

- **[Ancient Greek-Persian Parallel Corpus](https://aclanthology.org/2023.alp-1.21/)** - Parallel corpus with 400,000 sentences across Ancient Greek, English, Persian, and other languages. Focus on Homer and classical texts.

- **[Ancient Greek Multilingual Parallel Bible Corpus](https://opus.nlpl.eu/)** - Ancient Greek New Testament aligned with translations in 100+ languages. Available via OPUS.

### Named Entity Recognition

- **[Digital Athenaeus NER Dataset](https://digitalathenaeus.org/)** - Named entities from Athenaeus' Deipnosophistae annotated with PER, LOC, ORG, and derivative categories. Includes lemmatization and CTS URNs.

- **[WikiANN Ancient Greek](https://huggingface.co/datasets/wikiann)** - Part of the WikiANN multilingual NER dataset. Contains PER, LOC, ORG entities from Wikipedia.

- **[UGARIT NER Dataset](https://huggingface.co/UGARIT/grc-ner-xlmr)** - Training data for Ancient Greek NER with PER and LOC categories.

- **[GLAUx TEST Dataset](https://github.com/alekkeersmaekers/glaux)** - Manually annotated NER dataset with PERS, LOC, GRP scheme for evaluating models across diverse Ancient Greek literature.

### Text Classification

- **[Ancient Greek Genre Classification Dataset](https://github.com/QuantitativeCriticismLab/ancient_greek_genre_classification)** - Dataset for classifying Ancient Greek texts by genre (prose vs. verse, epic vs. drama). Contains stylometric features for almost all surviving classical Greek literature.

### Question Answering

Currently no native Ancient Greek QA datasets exist. Reference datasets in Modern Greek:
- **[GreekQA](https://cgi.di.uoa.gr/~ys02/frontistiria2022/GreekQA.pdf)** - Modern Greek reading comprehension dataset based on Wikipedia (1,000+ questions). Could serve as a template for Ancient Greek QA development.

### Machine-readable Dictionaries and Lexical Resources

- **[Liddell-Scott-Jones (LSJ) Greek-English Lexicon](https://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.04.0057)** - The standard Greek-English lexicon, digitized and searchable via Perseus and Logeion. Covers Ancient Greek from Homer to Byzantine period.

- **[Logeion](https://logeion.uchicago.edu/)** - Unified interface for multiple Greek lexicons including LSJ, Middle Liddell, Autenrieth's Homeric Dictionary, Slater's Pindar Lexicon, and more. Multilingual (Greek-English, Greek-French, Greek-German, Greek-Spanish, Greek-Dutch).

- **[Perseus Morphological Lexicon](http://www.perseus.tufts.edu/hopper/resolveform)** - Complete morphological forms database for Ancient Greek words.

- **[Morpheus](https://github.com/PerseusDL/morpheus)** - Morphological analysis engine and lexicon for Ancient Greek. Open source.

- **[BDAG](https://www.logos.com/product/3878/a-greek-english-lexicon-of-the-new-testament-and-other-early-christian-literature-bdag)** - Greek-English Lexicon of the New Testament and Other Early Christian Literature. The standard for Koine Greek. Commercial.

- **[Cunliffe Homeric Lexicon](http://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.04.0073)** - Specialized lexicon for Homeric Greek.

- **[Lampe Patristic Lexicon](https://patristic.proxied.lsadc.org/)** - Lexicon of Patristic Greek, designed to supplement LSJ.

### Papyri and Manuscripts

- **[Papyri.info](https://papyri.info/)** - The largest collection of digital papyri. Integrates multiple papyrological databases (Duke Databank, Heidelberger Gesamtverzeichnis, etc.). Includes Greek, Latin, Demotic, and Coptic papyri with full text and metadata.

- **[Trismegistos](https://www.trismegistos.org/)** - Interdisciplinary portal for ancient world texts and related information. Links papyri, ostraca, and tablets across databases.

- **[DCLP (Digital Corpus of Literary Papyri)](http://www.litpap.info/)** - Literary papyri from antiquity with scholarly editions.

---

## Pretrained Models

### Language Models

- **[Ancient-Greek-BERT](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT)** - The first Ancient Greek subword BERT model. State-of-the-art performance on POS tagging and morphological analysis. Based on bert-base architecture.

- **[Greek-BERT](https://github.com/nlpaueb/greek-bert)** - BERT model for Greek (primarily Modern Greek, but can transfer to Ancient Greek). Trained on 29GB dataset including Greek Wikipedia.

- **[GreekBART](https://arxiv.org/abs/2304.00869)** - First Seq2Seq model based on BART-base architecture for Greek. Pretrained on large-scale Greek corpus. Evaluated on generative and discriminative tasks.

- **[XLM-RoBERTa](https://huggingface.co/xlm-roberta-base)** - Multilingual model supporting Ancient Greek. Widely used for transfer learning on Ancient Greek tasks.

### Word Embeddings

- **[Ancient Greek fastText Embeddings](https://zenodo.org/records/7630945)** - 300-dimensional fastText word embeddings trained on 1GB of Ancient Greek texts. Produced for social network and social semantic analysis. Available on Zenodo.

- **[fastText Multilingual Embeddings](https://fasttext.cc/docs/en/crawl-vectors.html)** - Includes Ancient Greek (el) vectors trained on Common Crawl and Wikipedia. 300 dimensions.

- **[Polyglot Embeddings](https://sites.google.com/site/rmyeid/projects/polyglot)** - Includes Ancient Greek word embeddings.

### Specialized Models

- **[Ancient Greek NER Models (XLM-R)](https://huggingface.co/UGARIT/grc-ner-xlmr)** - Fine-tuned XLM-RoBERTa for Ancient Greek Named Entity Recognition (PER, LOC).

- **[OdyCy](https://aclanthology.org/2023.latechclfl-1.14/)** - General-purpose NLP pipeline for Ancient, Koine, and Medieval Greek. Achieves state-of-the-art performance in POS tagging, morphological analysis, and dependency parsing. Based on Ancient-Greek-BERT.

- **[greCy](https://github.com/jmyerston/greCy)** - Ancient Greek language models for spaCy. Trained on Perseus and PROIEL UD corpora. Includes tokenization, POS tagging, and dependency parsing.

- **[Stanza Ancient Greek Pipeline](https://stanfordnlp.github.io/stanza/)** - Pipeline called 'ktmu'. Supports tokenization, POS tagging, lemmatization, and dependency parsing. Use with caution for bracket processing issues.

---

## Methods/Software

### NLP Pipelines

- **[spaCy Ancient Greek Support](https://spacy.io/)** - Basic support for Ancient Greek including tokenization and stopwords. Requires custom models like greCy for full functionality.

- **[greCy](https://github.com/jmyerston/greCy)** - spaCy models specifically trained for Ancient Greek. Includes:
  - Tokenization
  - POS tagging
  - Lemmatization
  - Dependency parsing
  - Trained on Perseus and PROIEL treebanks

- **[OdyCy](https://github.com/jkostkan/odycy)** - Comprehensive Ancient Greek NLP pipeline built on spaCy. Supports Classical, Koine, and Medieval Greek. State-of-the-art performance across multiple tasks.

- **[Stanza](https://stanfordnlp.github.io/stanza/)** - Stanford NLP toolkit with Ancient Greek pipeline (ktmu). Trained on KTMU's UD Treebank.

- **[CLTK (Classical Language Toolkit)](https://github.com/cltk/cltk)** - Python library for processing Ancient Greek and other classical languages. Includes:
  - Tokenization
  - POS tagging
  - Lemmatization
  - Named entity recognition
  - Phonological transcription
  - Syllabification

- **[UDPipe](https://lindat.mff.cuni.cz/services/udpipe/)** - Universal dependency parser with Ancient Greek models trained on UD treebanks.

### Morphological Analysis

- **[Morpheus](https://github.com/PerseusDL/morpheus)** - Perseus morphological analysis engine. Analyzes and generates Greek word forms. Core component of Perseus Digital Library. Open source.

- **[Akhos](https://github.com/AshleyGrant/akhos)** - Modern Greek & Latin morphology tool. Web-based interface.

- **[Eulexis](https://github.com/PhVerkerk/Eulexis)** - Morphological analyzer and lemmatizer for Ancient Greek. Desktop application.

- **[Greek Word Study Tool (Perseus)](https://www.perseus.tufts.edu/hopper/morph)** - Online morphological analysis integrated with Perseus texts.

### Syntactic Parsing

- **[UDPipe Ancient Greek Models](https://lindat.mff.cuni.cz/services/udpipe/)** - Dependency parsers trained on UD Ancient Greek treebanks.

- **[Stanza Ancient Greek Parser](https://stanfordnlp.github.io/stanza/)** - Neural dependency parser for Ancient Greek.

- **[Turku Neural Parser](https://turkunlp.org/Turku-neural-parser-pipeline/)** - Used in Pedalion project with 90% LAS on Ancient Greek.

- **[DendroSearch](https://dendrosearch.informatik.uni-leipzig.de/)** - User-friendly tool for querying Greek treebanks with complex searches through visual interface.

### Text Classification and Genre Detection

- **[Ancient Greek Genre Classifier](https://github.com/QuantitativeCriticismLab/ancient_greek_genre_classification)** - Stylometric classifier distinguishing prose vs. verse, and epic vs. drama with >97% accuracy. Uses custom heuristic features without syntactic parsing.

### OCR and HTR

- **[Ancient Greek OCR (Tesseract-based)](https://ancientgreekocr.org/)** - Free software to convert scans of printed Ancient Greek into Unicode text. Based on Tesseract OCR engine, tailored for Ancient Greek typography. Version 2.0 available. Supports Windows, macOS, Linux, Android.

- **[Kraken](https://kraken.re/)** - Trainable OCR/HTR system optimized for historical documents and non-Latin scripts. Superior performance on Ancient Greek manuscripts when properly trained. Open source.

- **[Transkribus](https://readcoop.eu/transkribus/)** - HTR platform with GUI. Successfully used for Greek manuscripts (10th-14th c. AD). CER under 20% achievable. Commercial platform with free tier.

- **[eScriptorium](https://escriptorium.fr/)** - Open-source HTR platform based on Kraken. Used in digital humanities projects for Ancient Greek manuscripts.

- **[Tesseract Ancient Greek Models](https://github.com/tesseract-ocr/tessdata)** - Generic Tesseract models. Less accurate than specialized Ancient Greek OCR.

- **[i2OCR Ancient Greek](https://www.i2ocr.com/free-online-ancient-greek-ocr)** - Online OCR tool for Ancient Greek. Recognizes polytonic diacritics. Free for single images.

### Other Tools

- **[Perseids Platform](https://www.perseids.org/)** - Collaborative environment for creating and editing Ancient Greek annotations, including treebanking and translation alignment.

- **[Canonical Text Services (CTS)](http://cite-architecture.github.io/ctsurn/)** - Citation protocol for identifying passages in Ancient Greek texts. Used across Perseus and other digital libraries.

- **[Beta Code Converter](https://www.tlg.uci.edu/encoding/)** - Tools for converting between Beta Code and Unicode Greek. Essential for working with legacy databases.

---

## Script Types and Orthography

### Writing Systems
Ancient Greek has been written in several scripts throughout history:

1. **Greek Alphabet (Primary)**
   - **Polytonic orthography** - Standard for Ancient and Medieval Greek. Uses five diacritics:
     - Acute, grave, circumflex accents (pitch accent)
     - Rough and smooth breathing marks (aspiration)
     - Iota subscript
     - Diaeresis
   - **Monotonic orthography** - Introduced in 1982 for Modern Greek. Uses only two diacritics (tonos and diaeresis). Not used for Ancient Greek texts.

2. **Linear B** (Mycenaean Greek, ~1450-1200 BC)
   - Syllabic script
   - Limited corpus
   - Separate Unicode block (U+10000–U+1007F)

3. **Beta Code** (Legacy encoding)
   - ASCII-based transliteration system
   - Still used by TLG and older databases
   - Convertible to Unicode

### Unicode Ranges
- **Greek and Coptic**: U+0370–U+03FF (Modern Greek)
- **Greek Extended**: U+1F00–U+1FFF (Polytonic characters for Ancient Greek)
- **Ancient Greek Musical Notation**: U+1D200–U+1D24F
- **Ancient Greek Numbers**: U+10140–U+1018F

### Important Notes for NLP Processing
- **Diacritics are essential** for Ancient Greek - models must handle polytonic orthography
- **Case normalization is complex** - upper/lower case differences interact with diacritics
- **Multiple Unicode representations** - same word can be encoded differently (NFD vs NFC normalization)
- **Manuscript variations** - papyri and inscriptions may use different letterforms

---

## Online Demos and Resources

### Digital Libraries
- **[Perseus Digital Library](https://www.perseus.tufts.edu/)** - Most comprehensive open digital library. Includes texts, lexicons, morphological tools.
- **[TLG Online](https://stephanus.tlg.uci.edu/)** - The complete TLG corpus with online search. Requires subscription.
- **[Scaife Viewer](https://scaife.perseus.org/)** - Modern interface for Perseus texts. Part of Perseus 5.0 development.

### Lexicons and Dictionaries
- **[Logeion](https://logeion.uchicago.edu/)** - Unified lexicon interface. Multiple dictionaries in one search.
- **[Perseus Word Study Tool](https://www.perseus.tufts.edu/hopper/morph)** - Morphological analysis and lexicon lookup.
- **[Lsj.gr](https://www.lsj.gr/)** - LSJ search in multiple languages (Greek, Latin, English, French, German, Spanish, Russian, Chinese).

### Annotation and Analysis Tools
- **[Arethusa Treebank Editor](https://www.perseids.org/tools/arethusa/app/#/)** - Collaborative treebanking.
- **[Alpheios Reading Tools](https://alpheios.net/)** - Browser extension for reading Ancient Greek with instant morphological analysis and vocabulary support.
- **[DendroSearch](https://dendrosearch.informatik.uni-leipzig.de/)** - Query interface for Ancient Greek treebanks.

### Educational Resources
- **[Ancient Greek for Everyone](https://pressbooks.pub/ancientgreek/)** - Open textbook including guide to polytonic orthography.

---

## Communities and Research Groups

### International Communities
- **[Digital Classicist](https://digitalclassicist.org/)** - Decentralized international community for digital methods in classical studies. Regular seminars and wiki with resources.

- **[Open Greek and Latin Project](https://www.opengreekandlatin.org/)** - International collaboration for open educational resources. Includes corpus development and deep-reading tools.

- **[Perseids Project](https://www.perseids.org/)** - Collaborative platform for classics scholarship. Supports language acquisition, document annotation, and research tools.

- **[CLARIN](https://www.clarin.eu/)** - European research infrastructure for language resources. Hosts Ancient Greek corpora and tools.

### Research Centers and Labs

#### Europe
- **[Center for Hellenic Studies (Harvard)](https://chs.harvard.edu/)** - Digital methods for Ancient Greek and Latin texts. Sponsors projects like CTS protocol development.

- **[Leipzig University - Digital Humanities for Ancient Languages](https://www.dh.uni-leipzig.de/)** - AGLDT maintenance, treebank development, Pedalion project, DendroSearch.

- **[Tufts University - Perseus Project](https://sites.tufts.edu/perseusupdates/)** - Original home of Perseus Digital Library and AGLDT.

- **[University of Groningen - Classics in the Digital Age](https://www.rug.nl/research/research-let/onderzoek-per-vakgebied/griekse-en-latijnse-taal-en-cultuur/classics-in-the-digital-age)** - Research on NLP for Ancient Greek, including GLAUx corpus and OdyCy pipeline.

- **[KU Leuven - Pedalion Project](https://perseids.org/)** - Creating and enriching Ancient Greek treebanks with state-of-the-art NLP technology.

- **[National and Kapodistrian University of Athens - Digital Humanities Lab](https://dhl.phil.uoa.gr/)** - Laboratory for management of Greek and Latin digital resources.

- **[Furman University - D-scribes Project](https://d-scribes.philhist.unibas.ch/)** - Digital paleography for Greek and Coptic papyri.

#### North America
- **[University of Chicago - Classics Department](https://classics.uchicago.edu/)** - Chicago Homer project and digital humanities initiatives. Hosts Logeion.

- **[Northwestern University - Classics Digital Humanities](https://www.classics.northwestern.edu/research/digital-humanities-projects.html)** - Multiple digital projects including Ancient Rome in Chicago.

- **[UC Irvine - Thesaurus Linguae Graecae](https://stephanus.tlg.uci.edu/)** - Home of the TLG since 1972.

- **[University of San Diego - Diogenet Project](https://diogenet.ucsd.edu/)** - Social network analysis and NLP for Ancient Greek. Produced fastText embeddings.

#### UK
- **[The Alan Turing Institute - Computational Models of Meaning Change](https://www.turing.ac.uk/research/research-projects/computational-models-meaning-change-ancient-greek)** - Bayesian computational models for semantic change in Ancient Greek.

### Enthusiast Communities
- **Reddit: [r/AncientGreek](https://www.reddit.com/r/AncientGreek/)** - Active community for learners and scholars. Discussions on tools and resources.
- **Reddit: [r/GREEK](https://www.reddit.com/r/GREEK/)** - Modern and Ancient Greek community.
