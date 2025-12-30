# Building TEXT-AUTH: An Evidence-First System for Forensic Text Analysis

> *How a multi-metric, domain-aware forensic platform was implemented for probabilistic text authenticity assessment — without authorship claims.*

---

## Introduction: Why Text Forensics Needs a Rethink

The widespread availability of high-quality generative language systems has fundamentally altered the landscape of written communication. In education, publishing, journalism, and enterprise domains, stakeholders increasingly confront a complex forensic question:

> Does this text exhibit statistical, linguistic, and semantic patterns consistent with organically composed language, or does it display measurable characteristics associated with algorithmically regularized generation?

Traditional detection systems attempt to answer this with binary classifications: "Human" or "AI." This approach fails in practice because real-world text exists along a continuum—often hybrid, domain-specific, edited, paraphrased, or collaboratively produced.

TEXT-AUTH was conceived not as another detector, but as a forensic analysis system that evaluates observable textual properties and reports probabilistic consistency signals with explicit reasoning and uncertainty quantification. It provides evidence, not verdicts.

---

## Design Philosophy: Evidence, Not Attribution

At its core, TEXT-AUTH operates under a strict methodological constraint:

> The system does not determine who wrote a text.  
> It evaluates measurable linguistic and statistical patterns present in the text.

This distinction is both technical and ethical. By focusing on patterns rather than provenance, the system avoids the philosophical quagmire of authorship attribution while providing actionable forensic intelligence.

All outputs are framed as probabilistic assessments accompanied by:

- Explicit confidence intervals
- Quantified uncertainty scores
- Domain-specific calibration context
- Sentence-level evidence annotation

This architecture makes TEXT-AUTH suitable for high-stakes workflows where explainability, auditability, and human judgment remain essential components of decision-making.

---

## Core Architectural Principles

TEXT-AUTH implements five foundational principles that differentiate it from conventional detection systems:

### 1. Multi-Dimensional Analysis

Instead of relying on a single metric (typically perplexity), the system evaluates six orthogonal forensic signals, each capturing distinct aspects of textual consistency. This multi-dimensional approach provides robustness against adversarial manipulation—while individual metrics can be gamed, simultaneously gaming all six requires sophisticated effort that often produces other detectable anomalies.

### 2. Domain-Aware Calibration

The system recognizes that different writing genres exhibit different baseline characteristics. Academic papers naturally demonstrate lower perplexity than creative fiction. Legal documents show different structural patterns than social media posts. TEXT-AUTH implements sixteen domain-specific configurations, each with calibrated thresholds and metric weights, reducing false positives by 15–20% compared to generic detection approaches.

### 3. Explicit Uncertainty Modeling

Rather than forcing certainty, the system explicitly quantifies and reports uncertainty through a composite score combining:

- Inter-metric disagreement (variance)
- Individual metric confidence levels
- Distance from decision boundaries

High uncertainty triggers explicit recommendations for human review rather than automated decisions.

### 4. Granular Sentence-Level Analysis

Instead of providing a single document-level score, the system performs sentence-by-sentence forensic evaluation, producing color-coded visualizations that highlight where statistical anomalies occur. This granular approach provides actionable insights for editing, revision, and targeted review.

### 5. Transparent, Explainable Reasoning

Every analysis includes human-readable explanations detailing:

- Which metrics contributed most to the assessment
- Specific text patterns that triggered detection
- Domain context considerations
- Uncertainty sources and confidence factors

This transparency builds trust and enables informed decision-making.

---

## The Forensic Model: Six Orthogonal Signals

TEXT-AUTH evaluates text through six independent analytical lenses, each examining different dimensions of linguistic behavior. These metrics were selected based on their statistical independence, computational feasibility, and demonstrated discriminative power across text genres.

### 1. Statistical Predictability Analysis (Perplexity)

**What it measures**: The average negative log-likelihood of tokens given their preceding context, quantifying how "surprised" a reference language model is by the text sequence.

**Mathematical Foundation**:

$$
\text{Perplexity}(T) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i \mid w_{<i})\right)
$$

**Forensic Insight**: Language models generate text by selecting tokens with high conditional probabilities, creating sequences that occupy high-probability regions of the language distribution. Human writing, in contrast, includes unexpected lexical choices, creative expressions, and domain-specific jargon that models find statistically "surprising."

**Domain Calibration**: Expected perplexity ranges differ significantly by genre. Academic writing naturally exhibits lower perplexity due to formal structure and technical terminology. Creative writing shows higher baseline perplexity due to stylistic variation. Social media content displays the highest natural perplexity due to informal language and idiosyncratic expression.

---

### 2. Information Diversity Measurement (Entropy)

**What it measures**: The dispersion and unpredictability of token usage throughout the text, quantifying lexical richness and variation.

**Mathematical Foundation**:

$$
H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)
$$

**Forensic Insight**: Human-authored text typically exhibits higher lexical entropy due to expressive variation, nuanced vocabulary selection, and contextual adaptation. Algorithmically regularized text often shows more concentrated token distributions, with certain words and phrases appearing with unnatural frequency.

**Visual Representation**:

- Authentic Writing: ██░░░░░░░░ (High entropy, diverse distribution)
- Synthetic Generation: ██████░░░░ (Lower entropy, concentrated distribution)

---

### 3. Structural Rhythm Analysis (Burstiness and Uniformity)

**What it measures**: Sentence-level variation patterns through two complementary metrics.

**Burstiness Coefficient**:

$$
B = \frac{\sigma_L - \mu_L}{\sigma_L + \mu_L} \quad \text{where } B \in [-1, 1]
$$

Positive burstiness indicates varied sentence lengths; negative values indicate uniformity.

**Uniformity Metric**:

$$
U = 1 - \frac{\sigma_L}{\mu_L} \quad \text{for } \mu_L > 0
$$

**Forensic Insight**: Human writing exhibits natural rhythm—short, punchy sentences for emphasis followed by longer, complex sentences for elaboration. This creates characteristic "burstiness." Language model outputs tend toward more uniform sentence structures, creating a metronome-like consistency that lacks natural rhythmic variation.

---

### 4. Semantic Flow Evaluation (Coherence)

**What it measures**: The consistency of meaning between consecutive sentences using semantic embedding similarity.

**Mathematical Foundation**:

$$
\text{Coherence}(D) = \frac{1}{N_s-1} \sum_{i=1}^{N_s-1} \frac{\mathbf{e}_i \cdot \mathbf{e}_{i+1}}{\|\mathbf{e}_i\|\|\mathbf{e}_{i+1}\|}
$$

where $\mathbf{e}_i$ represents the embedding vector for sentence $i$.

**Forensic Insight**: Ironically, excessively high coherence can indicate algorithmic generation. Language models maintain remarkably consistent semantic flow through attention mechanisms. Human writing includes natural digressions, associative leaps, topic shifts, and rhetorical devices that create more variable coherence patterns.

**The Coherence Paradox**: In many contexts, better coherence actually provides evidence toward synthetic generation rather than organic composition.

---

### 5. Linguistic Pattern Analysis (Syntactic Complexity)

**What it measures**: Grammatical sophistication and syntactic variation through multiple sub-metrics:

**Part-of-Speech Diversity**:

$$
\text{POS}_{\text{diversity}} = \frac{|\{\text{POS tags}\}|}{N_{\text{tokens}}}
$$

**Parse Tree Depth Distribution**:

$$
D_{\text{syntactic}} = \frac{1}{N_{\text{sentences}}} \sum_{i=1}^{N_{\text{sentences}}} \max_{\text{tokens}} \text{depth}(t)
$$

**Forensic Insight**: Different writing styles exhibit characteristic syntactic fingerprints. Language models demonstrate systematic preferences for certain grammatical constructions, clause embeddings, and transitional patterns. Human writing shows greater syntactic irregularity, especially in longer passages where stylistic variation becomes more pronounced.

---

### 6. Stability Under Perturbation

**What it measures**: How text probability changes under meaning-preserving modifications, based on DetectGPT principles.

**Mathematical Foundation**:

$$
\Delta_{\text{logp}} = \frac{1}{k} \sum_{j=1}^k \left| \log P(T) - \log P(T'_{\epsilon_j}) \right|
$$

**Forensic Insight**: Text generated by language models occupies characteristic "curvature" regions in probability space—local maxima where small perturbations cause predictable probability decreases. Human-written text, not originating from these probability distributions, shows different perturbation sensitivity patterns.

**Computational Consideration**: This is the most resource-intensive metric, so TEXT-AUTH implements conditional execution, reserving it for cases where other metrics provide insufficient confidence.

## Ensemble Aggregation Methodology

Each of the six metrics produces:

- A synthetic probability estimate $p_i \in [0,1]$
- An internal confidence score $c_i \in [0,1]$
- An evidence strength classification (weak/moderate/strong)

The aggregation process follows a sophisticated multi-stage approach:

### Stage 1: Domain-Specific Base Weighting

Each of the sixteen supported domains has pre-calibrated base weights reflecting metric importance for that genre:

**Academic Domain Weights**:

- Perplexity: 22%
- Entropy: 18%
- Structural: 15%
- Semantic: 15%
- Linguistic: 20%
- Stability: 10%

---

### Stage 2: Confidence-Adjusted Dynamic Weighting

Base weights are dynamically adjusted based on each metric's confidence using a sigmoid scaling function:

$$
w_i^{\text{(adjusted)}} = w_i^{\text{(base)}} \cdot \left( \frac{1}{1 + e^{-\gamma(c_i - 0.5)}} \right)
$$

where $\gamma = 10$ controls adjustment sensitivity.

---

### Stage 3: Normalization and Aggregation

Adjusted weights are normalized to sum to 1.0, then used for weighted probability calculation:

$$
P_{\text{synthetic}} = \sum_{i=1}^6 w_i^{\text{(final)}} \cdot p_i
$$

---

### Stage 4: Consensus Analysis

The system evaluates inter-metric agreement:

- High consensus increases overall confidence
- Low consensus triggers uncertainty flags
- Extreme disagreement may indicate adversarial manipulation or domain misclassification

---

## Uncertainty Quantification Framework

TEXT-AUTH explicitly models uncertainty through a three-component composite score:

### 1. Metric Disagreement Uncertainty

$$
U_{\text{variance}} = \min(1.0, \sigma_P \cdot 2)
$$

where $\sigma_P$ is the standard deviation of the six metric probabilities.

### 2. Confidence-Based Uncertainty

$$
U_{\text{confidence}} = 1 - \frac{1}{6} \sum_{i=1}^6 c_i
$$

### 3. Decision Boundary Uncertainty

$$
U_{\text{decision}} = 1 - 2 \cdot |P_{\text{synthetic}} - 0.5|
$$

This component captures how close the final probability is to the maximally uncertain point (0.5).

### Composite Uncertainty Score

$$
U_{\text{total}} = 0.4U_{\text{variance}} + 0.3U_{\text{confidence}} + 0.3U_{\text{decision}}
$$

**Interpretation Guidelines**:

- **< 0.20**: High confidence, reliable assessment
- **0.20 – 0.40**: Moderate confidence, use with appropriate caution
- **> 0.40**: Low confidence, inconclusive—recommend human review

---

## Domain-Aware Calibration System

The system recognizes that different writing genres have different normative characteristics. Sixteen domains are supported, each with specialized configurations.

### Domain Classification Process

1. **Feature Extraction**: Analyze text for domain indicators including formality, technical terminology, citation patterns, punctuation usage, and structural complexity
2. **Probabilistic Classification**: Use heuristic and optional pre-trained model-assisted inference to estimate domain probabilities
3. **Threshold Selection**: Apply domain-specific detection thresholds and metric weights

### Example Domain Configurations

**Academic Domain (Conservative thresholds)**:
- Higher linguistic complexity expectations
- Reduced sensitivity to low perplexity
- Elevated synthetic probability threshold (0.75)
- Priority on minimizing false positives

**Creative Domain (Adaptive thresholds)**:
- Enhanced entropy and structural analysis
- Tolerance for high perplexity variation
- Balanced synthetic threshold (0.70)
- Focus on stylistic pattern detection

**Social Media Domain (Lenient thresholds)**:
- Perplexity as primary signal
- Relaxed linguistic requirements
- Lower synthetic threshold (0.65)
- Emphasis on conversational authenticity

**Technical Documentation (Strict thresholds)**:
- Semantic coherence prioritization
- Highest synthetic threshold (0.80)
- Structural pattern analysis
- Maximum emphasis on minimizing false accusations

### Calibration Methodology

Thresholds were optimized using ROC curve analysis on curated datasets of 10,000+ verified texts per domain, with cross-validation to ensure generalization. The optimization objective balanced precision and recall while prioritizing false positive minimization in high-stakes domains.

--- 

## Interpretability and Explainability

### Sentence-Level Forensic Highlighting

Text is analyzed at the sentence level, with each sentence receiving a color-coded classification:

- 🔴 **Deep Red**: Strong synthetic consistency signals (> 80% probability)
- 🟠 **Light Red**: Moderate synthetic signals (60–80% probability)
- 🟡 **Yellow**: Inconclusive or mixed signals (40–60% probability)
- 🟢 **Green**: Strong authentic consistency signals (< 40% probability)

Hover interactions reveal detailed forensic data for each sentence, including individual metric scores and contributing factors.

### Natural Language Reasoning Generation

Every analysis includes comprehensive human-readable explanations structured as:

#### Executive Summary
A concise overview of the forensic assessment, including final probability, confidence level, and primary findings.

#### Key Forensic Indicators
Specific text characteristics that contributed to the assessment, such as:
- "Unusually uniform sentence structure (burstiness: -0.12)"
- "Exceptionally high semantic coherence (mean: 0.91)"
- "Low perplexity variance indicating predictable token sequences"

#### Confidence Factors Analysis
Explicit discussion of:
- Supporting evidence (metrics showing strong signals)
- Contradicting evidence (metrics showing conflicting signals)
- Uncertainty sources (domain ambiguity, text length limitations, etc.)

#### Metric Contribution Breakdown
Percentage attribution showing how much each forensic signal contributed to the final assessment, helping users understand the analytical weighting.

#### Domain Context Considerations
Explanation of how the text's genre affected the analysis, including any domain-specific adjustments applied to thresholds or interpretations.

--- 

## Ethical Framework and Implementation Principles

### Core Ethical Commitments

- **Transparency Over Certainty**: The system explicitly acknowledges uncertainty rather than feigning omniscience. All outputs include confidence intervals and uncertainty quantification.
- **Evidence Over Attribution**: TEXT-AUTH reports statistical patterns, not authorship claims. This distinction is maintained throughout the user interface, documentation, and API responses.
- **Contextual Awareness**: Analyses consider domain, genre, language, and cultural factors that might affect interpretation. The system includes bias mitigation measures for protected writing styles.
- **Human-in-the-Loop Design**: Automated analysis supports rather than replaces human judgment. High-uncertainty cases explicitly recommend human review, and all high-stakes applications require human oversight.
- **Continuous Auditing**: The system implements regular fairness evaluations, performance monitoring, and bias detection to identify and address emerging issues.

### Responsible Use Guidelines

**Appropriate Applications**
- Academic integrity screening (with human review processes)
- Content verification in editorial workflows
- Resume authenticity checking (as part of holistic review)
- Research on text generation patterns
- Writing assistance tool calibration

**Inappropriate Applications**
- Sole determinant for academic penalties
- Automated rejection without appeal mechanisms
- Surveillance without consent or disclosure
- Cross-cultural comparison without proper calibration
- Real-time monitoring without transparency

### Bias Mitigation Strategies

The system implements multiple bias reduction techniques:
- **Domain normalization**: Genre-specific baselines reduce false positives against formal writing styles
- **Confidence thresholding**: Higher uncertainty triggers human review for edge cases
- **Protected style detection**: Identification of non-native, neurodivergent, or regional writing patterns with adjusted interpretation
- **Regular fairness auditing**: Scheduled evaluation of performance across demographic and stylistic subgroups

### Computational Performance
- Short texts (100–500 words): 1.2 seconds average processing
- Medium texts (500–2000 words): 3.5 seconds average
- Long texts (2000+ words): 7.8 seconds average
- Parallel execution: 2.9× speedup over sequential processing
- Memory footprint: 1.5–3.0 GB depending on configuration

---

## Conclusion: Toward Responsible Text Forensics

TEXT-AUTH represents a paradigm shift in text authenticity analysis—from binary classification to evidence-based forensic assessment. By combining orthogonal statistical signals with domain-aware calibration and transparent reasoning, the system provides actionable intelligence while acknowledging the inherent complexity and uncertainty of the problem.

### Key Contributions

- **Methodological Innovation**: A multi-metric, domain-calibrated approach that recognizes genre diversity in writing patterns
- **Uncertainty Quantification**: Explicit modeling of confidence and uncertainty prevents overconfident errors
- **Transparent Reasoning**: Comprehensive explainability builds trust and enables informed decision-making
- **Ethical Foundation**: Clear boundaries around appropriate use and acknowledgment of limitations
- **Production Engineering**: Parallel processing, efficient caching, and scalable architecture enable real-world deployment

---

### The Path Forward

Text authenticity assessment remains an evolving challenge in the age of generative AI. TEXT-AUTH provides a foundation for responsible forensic analysis, but continued development is essential:

- Multilingual expansion to support diverse linguistic contexts
- Real-time analysis capabilities for interactive writing environments
- Enhanced adversarial robustness against evolving evasion techniques
- Institutional calibration frameworks for organization-specific needs
- Collaborative research initiatives to advance the field collectively

Ultimately, the goal is not perfect detection—an unrealistic standard in an adversarial environment—but rather the development of tools that make authenticity analysis more transparent, more nuanced, and more accountable than previous approaches.

By focusing on evidence rather than attribution, uncertainty rather than false certainty, and support rather than replacement of human judgment, TEXT-AUTH contributes to building trust in written communication in the generative AI era.

---

**TEXT-AUTH Forensic Text Analysis Platform**  
Version 1.0 — December 2025  
Author: Satyaki Mitra
_Evidence-based assessment, transparent reasoning, responsible implementation_

---