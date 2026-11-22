# Point-by-point response to the ASMUS 2026 reviewer comments

This file records how the comments in `ASMUS2026_review_comments.pdf` were handled in the SPIE Medical Imaging 2027 abstract materials. The revised files are `Abstract_v0.docx` and `Supplementary_v0.docx`.

No additional experiments or validation runs were performed. Comments requiring new results are therefore handled by narrowing claims, reorganizing existing results, adding methodological detail, or explicitly stating the remaining limitation.

Status labels:

- **Addressed**: resolved in the revised text, organization, table, figure caption, or disclosure.
- **Partially addressed**: the writing issue was resolved, but the underlying experimental limitation remains.
- **Not experimentally addressed**: the requested experiment was not run; the limitation is disclosed and related claims were narrowed.

## Meta-review

| Reviewer concern | Status | Response in the SPIE materials |
|---|---|---|
| Cross-model ranking mixed fundamentally different prompt types. | **Addressed** | Results are organized by prompt family. Table 1 is explicitly labeled as prompt-stratified and states that it is not a cross-prompt leaderboard. The abstract also avoids a single overall model ranking. |
| Results lacked variance or significance testing. | **Partially addressed** | No inferential statistics were added. All values are described as descriptive point estimates, language implying statistical significance was removed, and the absence of confidence intervals and hypothesis tests is stated in the abstract, Table 1 caption, Figure 2 caption, and Limitations. |
| A promised metric was missing from Results. | **Addressed in writing** | Signed relative area error is no longer promised or discussed in the SPIE materials. Dice is the primary summary metric. Boundary distances are described as dataset-level only and are not pooled across datasets. |
| UltraSAM train/test contamination or dataset overlap undermined the “unseen” claim. | **Addressed in framing; provenance remains incomplete** | “Unseen dataset” and strict zero-shot claims were removed. The materials describe released-checkpoint behavior and disclose known dataset-level overlap for UltraSAM and SAMUS. |
| Novelty was limited or overstated. | **Addressed** | The work is explicitly presented as an ultrasound-specific consolidation and technology-assessment framework, not as a new architecture or a new general benchmarking paradigm. |

## Reviewer vq7i: major weaknesses

### 1. Limited novelty and insufficient positioning against prior benchmarks

**Status: Addressed.**

The revised materials state that the contribution is not a new architecture and that promptable-model benchmarking is an established paradigm. Novelty is limited to the ultrasound-specific consolidation of 13 public datasets across 11 anatomical categories, the provenance-preserving registry-decoder framework, prompt-family organization, wording and perturbation analyses, and target-morphology failure characterization. Relevant model and validation references are included in the supplemental file.

### 2. Cross-model ranking confounded by heterogeneous prompts

**Status: Addressed.**

The mixed-prompt overall ranking was removed. Table 1 separates box, point, and text/hybrid settings and warns against combining them into a single leaderboard. Comparisons in the prose are made only within the same prompt family or within MedSAM3 when comparing text with text-plus-box.

### 3. Likely pretraining/evaluation overlap and unsupported “unseen datasets” claim

**Status: Partially addressed.**

Strict zero-shot and unseen-dataset language was removed. The Limitations section discloses that UltraSAM's US-43d development corpus contains public datasets also used in the benchmark, including BUS-BRA, GIST514-DB, and MMOTU-2D, and that SAMUS's US30K includes BUSI and CAMUS. The revised interpretation is limited to released-checkpoint behavior. A complete per-model provenance matrix could not be produced from the supplied materials and is not claimed.

### 4. Robustness and morphology analyses were performed only for UltraSAM

**Status: Not experimentally addressed.**

No second model was added. The supplemental file now identifies these analyses as UltraSAM-specific descriptive analyses and states that they cannot establish a universal property of promptable ultrasound segmentation models.

### 5. Six curated datasets and signed relative area error were absent from the comparative results

**Status: Addressed in reporting; no missing experiments were added.**

The cross-model table is explicitly described as a seven-dataset reporting subset. The six remaining datasets are described as part of the broader collection and the UltraSAM perturbation analysis, not as complete cross-model evidence. Signed relative area error was removed from the reported-method claims because it was not available in the supplied results.

### 6. Point estimates were reported without uncertainty or significance tests

**Status: Partially addressed.**

No new statistical analysis was performed. The revised documents consistently use “descriptive,” avoid “significant,” and disclose that confidence intervals and hypothesis tests are unavailable. Figure 2 and its correlations are not presented as inferential evidence.

### 7. The averaging scheme and pooled pixel-scale ASSD were difficult to interpret

**Status: Addressed.**

Macro Dice is defined as the arithmetic mean of seven dataset-level mean Dice values, with each dataset receiving equal weight. The text explains that this prevents UltraBones100k from dominating because of its sample count. Cross-dataset ASSD averaging was removed because physical spacing is inconsistent or unavailable.

### 8. The single-point protocol was weak and could bias point-versus-box conclusions

**Status: Addressed methodologically; no new point sweep was run.**

The point protocol is now specified as selecting the middle element after enumerating all foreground pixels in row-major order across the complete mask; disconnected components are not handled separately. Point results are explicitly described as protocol-specific, and the revised materials do not claim that point prompting is generally inferior to box prompting.

## Reviewer vq7i: minor comments

| Comment | Status | Response |
|---|---|---|
| Grammar and consistency errors, including duplicated words and inconsistent UltraSAM capitalization. | **Addressed** | The SPIE materials were rewritten and proofread; UltraSAM capitalization is standardized and the cited grammar errors are absent. |
| Document which datasets each model saw during pretraining. | **Partially addressed** | Known overlap for UltraSAM and SAMUS is disclosed. The materials do not claim a complete provenance matrix where the released documentation is insufficient. |
| Specify how the middle foreground point is chosen for disconnected masks. | **Addressed** | The Methods section states that foreground pixels across the complete mask are pooled in row-major order and the middle element is selected; components are not processed separately. |
| Report whether translated or scaled boxes still contain the target. | **Not experimentally addressed** | Containment was not recorded in the supplied results. Figure 2 and Limitations now state this explicitly and warn that a displaced box may change the localization task rather than represent boundary noise alone. |
| Explain how Figure 3 examples were selected. | **Addressed** | The caption states that cases span the seven reported anatomies and were selected to illustrate prespecified error categories supported by dataset-level trends; they are not used to estimate failure prevalence. |

## Reviewer tiAX: experimental scope and comparison comments

| Reviewer concern or question | Status | Response in the SPIE materials |
|---|---|---|
| Include parameter count, memory footprint, inference time, or other computational comparisons. | **Not experimentally addressed** | Model complexity, latency, and memory are listed as unevaluated limitations. No efficiency claim is made. |
| Consider tuning non-ultrasound models for a fairer comparison. | **Not experimentally addressed** | The scope is explicitly released checkpoints without fine-tuning, test-time adaptation, or dataset-specific threshold optimization. Claims are limited to out-of-the-box released-checkpoint behavior. |
| Evaluate every model-prompt combination. | **Not experimentally addressed** | The revised text notes that released interfaces differ and that every model-prompt combination was not evaluated. Results are stratified by the prompt modes that each released system supports. |
| Include all 13 datasets in the comparative table. | **Partially addressed** | The seven-dataset subset is now named and the remaining six datasets are not implied to have complete cross-model results. No new model runs were added. |
| Add standard deviations or significance tests. | **Partially addressed** | No statistics were added; all results are labeled descriptive and the limitation is repeated wherever comparative numbers are presented. |
| Explain why only B-mode ultrasound was evaluated. | **Addressed as scope limitation** | The collection is explicitly defined as 2D B-mode, and the Limitations section states that the evidence is limited to B-mode images. Other ultrasound modes are therefore outside the stated scope. |
| Explain why Dice and ASSD were selected when other metrics were described. | **Addressed** | Dice is the primary summary metric. Boundary distances are restricted to dataset-level interpretation because common physical spacing is unavailable; unreported metrics are not promised. |
| Clarify why UltraSAM alone was used for morphology and perturbation analysis. | **Addressed in framing** | The circular “best-performing model” justification was removed. The analysis is presented as an existing UltraSAM-specific diagnostic study, and its inability to support model-general conclusions is stated explicitly. |

## Reviewer tiAX: clarity and formatting comments

| Comment | Status | Response |
|---|---|---|
| Present contributions as an unnumbered list if no ranking is intended. | **Addressed** | Novel aspects are presented as prose rather than a ranked numbered list. |
| Start references on a separate page. | **Addressed** | References begin on a separate page and are outside the four technical-content pages. |
| Enlarge fonts in tables and figures where possible. | **Addressed for document layout** | The document follows SPIE typography: 10-point body text, 9-point captions and table text, and figures placed at near-full text width. Source figure artwork is reused without modification. |
| Make the bottom lane of Figure 1 flow left to right. | **Not changed in this draft** | At the author's direction, the original ASMUS Figure 1 is reused unchanged and no replacement figure is introduced. |
| Reduce redundancy between Experiments and Results. | **Addressed** | Experimental setup is consolidated under Methods; outcomes and interpretation are presented under Results and Prompt Robustness and Failure Modes. |
| Place MedSAM3 evaluations together in the results table. | **Addressed** | The text/hybrid block places the three MedSAM3 settings consecutively, followed by the two Medical SAM3 settings. |
| Move decoder, annotation-format, and caching details to supplementary material. | **Addressed by document scope** | The short technical abstract omits implementation details. The supplemental review file retains only a concise registry-decoder description needed to understand the assessment design. |
| Clarify the originality relative to previous work. | **Addressed** | The materials explicitly identify the work as a consolidation of an established paradigm and enumerate the ultrasound-specific scope and diagnostic analyses that distinguish it. |

## Items that remain for author confirmation

1. Confirm the final author list. The latest local `main.tex` includes Matthias Seibold, while the prior OpenReview record does not.
2. Confirm that the statement “No concurrent submission is disclosed in the supplied materials” remains accurate at the time of SPIE submission.
3. If a complete per-model training-provenance matrix becomes available, add it before submission; the current wording intentionally avoids claiming complete provenance.
