# Phase Zero 

## Phenomenon: Retrieval behavior 

### Experiment: 
Picked up 18 samples from Benchmark questions of raGROW Phase 1 to understand the below
- Why retrieval succeeds & where it fails
- Categorize retrieval behavior based on initial observation across all 18 samples
- Perform Evidence profiling for each unique causal factor (7) to understand the retrieval behavior pattern
- Formulate causal hypothesis and arrive at emperical taxonomy labels for each all the 7 cases
- Arrive at theoritcal taxonomy labels based on observation and create further theortical taxonomy labels 

### Observed: 
1. Observe the query and the retrieval result 
Compare the query with each of Top-K retrieved chunks and their similarity scores.
2. Classify the observed retrieval behaviour
Assign the appropriate taxonomy category without trying to explain the cause yet.
3. Profile the evidence
Examine the retrieved evidence, focusing on the chunks material to the observed behaviour.
4. Formulate the causal hypothesis
Combine Observed retrieval behaviour + Evidence Profile + Corpus context where necessary to formulate candidate causal factors.
5. Arrive at theoritial taxonomy labels
Based on observation of the corpus and retrieval behavior, created further labels to baseline the future experiments


### Interpretation: 
1. At the highest level there can be only successful and failed retrievals. While successful retrievals may help in generation based on the threshold limitations, failure may have multiple reasons
2. Broadly failures can be due to absence of information in the corpus or failures inspite of availeble information in the corpus. The latter can be due to multiple causes and lead to varied taxonomy labels
3. We can analyze the retrieval behavior from two perspectives - (a) Causal Analysis leads to taxonomy which tells us what happened ; (b) evidence profiling helps constrain why it may have happened. These two are complimentary to understand retrieval mechanism
4. Retrieval behaviour is a distinct analysis before we start reviewing generation outcomes; Similarly threshold decisions is a seperate mechanism that impacts retrieval results.
5. When we repeat this cycle "Observe → Classify → Evidence Profile → Formulate Causal Hypothesis", it leads to "Retrieval Diagnostic Framework" which is re-usable for better understanding and evaluation of retrieval mechanism.

### Technical conclusion: 
Retrieval quality is determined not only by semantic similarity, but by the interaction between corpus information, chunk composition, evidence distribution, and similarity measurement. Retrieval diagnostics therefore require examining the relationship between the query, Top-K evidence, similarity scores, and corpus coverage rather than relying on the top-ranked result alone

## Capability demonstrated: 
Ability to systematically diagnose retrieval behavior by separating retrieval outcomes from their causal factors, profiling retrieved evidence, and developing empirical and theoretical taxonomy labels that can be reused for subsequent retrieval experiments.

## Evidence: 
Retrieval experiments, benchmark samples, evidence profiles, and taxonomy analysis artifacts are maintained in:
`/home/surendran/LangChain/Semantics@Suren/S@S_Module Zero Experiments`/

# Phase 1

## Phenomenon: (1) Perturbation of Queries

### Experiment:
Conducted controlled query-perturbation experiments using the same underlying question and corpus to observe how retrieval behavior changes under:
* Synonym substitution
* Increased specificity
* Reordering of query terms
* Negation
* Addition of positive and negative constraints
Compared Top-K chunks, cosine scores, ranking changes, and Top-1 dominance before and after perturbation across MXBAI and BGE-M3.

### Observed:
1. Compare the original query with its perturbed version and examine the Top-K retrieved chunks, cosine scores, and ranking.
2. Observe whether the answer-bearing Top-1 chunk is retained, replaced, or repositioned after perturbation.
3. Compare score movements and the gap between Top-1 and Top-2 to assess changes in retrieval confidence and ranking stability.
4. Compare the same perturbation across MXBAI and BGE-M3 to identify model-specific differences in retrieval behavior.

### Interpretation:
1. Retrieval is sensitive to the formulation of a query even when its underlying intent remains unchanged; however, the degree and nature of this sensitivity varies by perturbation type and embedding model.
2. Query perturbation can alter cosine scores and the relative ordering of retrieved chunks without necessarily changing the answer-bearing Top-1 chunk.
3. Synonym substitution, specificity, reordering, and negation do not produce a uniform retrieval response; the same perturbation can affect score magnitude, ranking stability, and Top-1 dominance differently.
4. Adding information that is present in the corpus can strengthen the answer-bearing chunk, whereas adding information unavailable in the corpus can alter the overall similarity landscape without providing useful evidence.
5. The experiments demonstrate that cosine similarity should be interpreted comparatively rather than as an absolute measure of retrieval correctness or confidence.

### Technical conclusion:
Query formulation is an active variable in dense retrieval. Semantically equivalent or related queries can produce materially different similarity scores and Top-K rankings, with the magnitude and stability of the effect depending on both the perturbation and the embedding model. Therefore, retrieval evaluation should include controlled query perturbation rather than relying only on performance against the original query formulation.

### Capability demonstrated:
Ability to design controlled query-perturbation experiments and systematically evaluate embedding-model sensitivity through Top-K stability, cosine-score movement, rank changes, and Top-1/Top-2 separation.

### Evidence:
`/home/surendran/LangChain/Semantics@Suren/S@S_Module 1 Experiments/phase1_perturbations_query.ods`

# Phase 1

## Phenomenon: (2) Perturbation of Chunks

### Experiment:
Conducted controlled chunk-perturbation experiments using the same query and corpus to understand how changes to the composition, wording, density, and sentence order of a chunk influence its retrieval score and ranking.
The experiments covered wording modification, density reduction, sentence-order changes, and addition/reduction of noise, with comparisons across MXBAI and BGE-M3.

### Observed:
1. Compare the original chunk with its perturbed version while keeping the query and other corpus chunks unchanged.
2. Observe changes in the perturbed chunk's cosine similarity and Top-K ranking.
3. Examine whether changes to one chunk affect the scores or ranking of the other retrieved chunks.
4. Compare the magnitude and direction of these effects across MXBAI and BGE-M3.

### Interpretation:
1. Perturbing a chunk produces a primarily local change in its similarity score; the scores of the other retrieved chunks remain largely unchanged.
2. Changes to wording and density generally produced relatively small changes in retrieval behavior, while sentence-order and noise perturbations could produce more substantial ranking changes.
3. Removing surrounding information does not necessarily increase similarity to the query. In some cases, retaining only the apparently critical information actually reduced the chunk's similarity score and caused a significant ranking deterioration.
4. The same chunk perturbation can produce different score and ranking responses across embedding models, demonstrating that the representation of a chunk is model-dependent.
5. Chunk-level perturbation can therefore change retrieval ranking even when the underlying relevant information has not been removed.

### Technical conclusion:
Dense retrieval similarity is a property of the **entire chunk representation**, not simply a measure of the presence of the most relevant sentence. Changes to chunk wording, density, sentence order, or surrounding information can alter its embedding and consequently its similarity score and ranking. The impact is predominantly local to the perturbed chunk, but its changed score can indirectly alter its position relative to the other retrieved chunks.

## Capability demonstrated:
Ability to design controlled chunk-perturbation experiments and assess the sensitivity of dense retrieval to changes in chunk composition, while separating local similarity-score changes from broader ranking effects and comparing embedding-model behavior.

## Evidence:
`/home/surendran/LangChain/Semantics@Suren/S@S_Module 1 Experiments/phase1_perturbations_chunks.ods`

## Phenomenon: (3) Perturbation of Chunk Boundaries

### Experiment:
Conducted controlled experiments by changing how corpus information is divided into chunks while keeping the underlying information and retrieval queries unchanged.
The experiments covered overlap, window size, atomic chunking and merged chunking, with retrieval behaviour compared across MXBAI and BGE-M3.

### Observed:
Compare retrieval behaviour under different chunk-boundary configurations using Top-1 identity, cosine score, Top-K ranking and score separation.
Observe whether the relevant information remains retrievable, changes position, or becomes distributed across multiple chunks.
Examine the effect of changing the amount of surrounding information through different window sizes and merged chunks, and the effect of isolating information into atomic chunks.
Compare the retrieval response across MXBAI and BGE-M3 for the different chunking configurations.

### Interpretation:
Retrieval behaviour is sensitive to how information is distributed across chunk boundaries; changing boundaries can alter both retrievability and ranking without changing the underlying corpus information.
Introducing overlap can recover information that was previously difficult to retrieve when relevant content lies near a chunk boundary, but can also introduce competing neighbouring representations.
Changing window size and grouping information into larger chunks can alter the amount of contextual and unrelated material represented together, while atomic chunking can localise the relevant evidence.
Isolating information does not automatically guarantee higher similarity or better retrieval; the retrieval response depends on how the embedding model represents the resulting chunk.
The effect of chunk-boundary changes varies across embedding models, indicating that chunking strategy and embedding representation interact in determining retrieval behaviour.

### Technical conclusion:
Chunk boundaries are an active variable in dense retrieval. The same underlying information can produce different retrieval outcomes depending on how it is partitioned, overlapped, isolated or merged. Therefore, chunking strategy is not merely a preprocessing decision; it is part of the retrieval mechanism and needs to be evaluated empirically.

## Capability demonstrated:
Ability to design controlled chunk-boundary experiments and evaluate how overlap, window size and information grouping influence dense retrieval behaviour across different embedding models.

## Evidence:
`//home/surendran/LangChain/Semantics@Suren/S@S_Module 1 Experiments/phase1_perturbations_chunk_boundaries.ods`/

# Module 2

## Phenomenon: Embedding representation does not map directly to textual quantity

### Experiment
We performed controlled textual perturbation experiments using a fixed query and modified versions of a chunk. The purpose was to observe how the embedding representation changes when the underlying text is substantially changed.
The measurements included embedding norm, query–chunk cosine similarity, and normalized Euclidean distance.

### Observed
A textual perturbation produces a new embedding representation.
The new representation can differ in magnitude, direction, or both.
Substantial removal of unrelated text from a chunk did not necessarily produce a proportional reduction in embedding magnitude.
The original mixed-content chunk and the revised policy-focused chunk remained strongly directionally aligned despite the textual modification.
Cosine similarity and normalized Euclidean distance changed consistently because, for normalized vectors, they represent the same underlying angular relationship in different mathematical forms.
Repeating the experiment under the same embedding conditions is important for distinguishing reproducible behaviour from an accidental observation.

### Interpretation
The embedding should not be interpreted as a direct numerical representation of text length or the quantity of information present in the text.
In particular:
Embedding magnitude cannot be equated with character count or amount of textual information.
The geometric measurements tell us what happened to the representation, but they do not by themselves explain why the embedding model produced that representation.

### Technical conclusion
A textual change can alter an embedding without providing a simple, proportional relationship between textual quantity and geometric change. Cosine similarity, norm and normalized Euclidean distance are measurements of the resulting representation; they are not explanations of the representation-generation process.

### Capability demonstrated
Ability to design a controlled textual-perturbation experiment and use embedding geometry to investigate representation behaviour rather than treating retrieval scores as self-explanatory.

### Evidence
`//home/surendran/LangChain/Semantics@Suren/S@S_Module 2 Experiments/`

# Module 3
Phenomenon: Local geometry of embedding representations

## Experiment:
Investigated embedding geometry beyond individual query–chunk similarity by examining relationships among retrieved candidates and the structure of their local neighbourhood.
The experiments examined inter-chunk similarity, local density, retrieval margin/competition, geometric grouping, semantic grouping, and the intuition of a local manifold.

## Observed:
Query-to-chunk cosine similarity describes the relationship between a query and an individual candidate representation.
Inter-chunk similarity describes how the candidate representations relate to one another. A Top-K retrieval therefore provides a small observable window into the local geometric neighbourhood surrounding the retrieved representations.
Retrieval margin and local density are distinct geometric properties. Margin describes the separation between the query's winner and its competitors, whereas density describes the relationships among the neighbouring representations.
A false-positive case demonstrated that the correct semantic information can exist inside a retrieved chunk while unrelated or weakly related content within the same chunk influences its embedding. Decomposing the chunk into atomic statements and comparing those representations with the query exposed the relevant semantic component more clearly. This established intra-chunk noise as a representation-level retrieval factor.
A compressed-ranking case showed that several retrieved chunks can occupy a tightly packed similarity region, with very small differences in query-to-chunk cosine scores. High inter-chunk similarity among these candidates indicates that the retrieval competition is occurring within a geometrically similar neighbourhood rather than between clearly separated alternatives.
Retrieval ambiguity was demonstrated through cases where multiple candidates were simultaneously close to the query and to one another. In such situations, cosine ranking alone provides limited separation between competing candidates.
Geometric grouping was not treated as proof of semantic grouping. A geometric relationship establishes similarity in the representation space; semantic equivalence or relevance requires independent evidence.
The concept of a local manifold was introduced as an intuition for studying the structured region surrounding a representation rather than attempting to reason about the entire embedding space at once.

### Interpretation:
Retrieval can be examined at two complementary geometric levels: the relationship between the query and each candidate, and the relationships among the candidates themselves.
The neighbourhood therefore matters because the behaviour of a retrieved result is influenced not only by its absolute similarity to the query, but also by the geometric competition surrounding it.
The experiments provided three concrete retrieval insights. False positives can arise from intra-chunk noise; compressed rankings can arise when competing candidates occupy a tightly packed local region; and retrieval ambiguity occurs when several candidates are geometrically difficult to separate.
These observations also reinforce the distinction between geometric evidence and semantic interpretation. Geometry can reveal why retrieval behaves a certain way, but it does not by itself establish which candidate contains the correct answer.

### Technical conclusion:
Embedding retrieval is a neighbourhood-level geometric phenomenon, not merely a sequence of query-to-chunk cosine scores.
Cosine similarity, inter-chunk similarity, local density, and retrieval margin provide complementary measurements of local structure. Together, they can expose competition, compression, ambiguity, and representation-level noise that are not apparent from the Top-1 score alone.

### Capability demonstrated:
Ability to analyse embedding representations through local neighbourhood structure, distinguish query-centred similarity from neighbourhood-centred geometry, and critically evaluate the relationship between observed geometric structure and semantic relationships.

### Evidence:
`//home/surendran/LangChain/Semantics@Suren/S@S_Module 3 Experiments/`

# Module 4
Global Geometry Behavior

## Introduction
The experiments examined whether global transformations of the embedding space — Raw, Mean-Centered, Whitened and Intrinsic-Dimensionality-Reduced — materially alter dense retrieval behaviour and whether such geometric changes translate into meaningful retrieval improvement.
Two corpus settings were examined: a synthetic corpus with controlled semantic variation and B01, the earlier HR corpus with relatively homogeneous information. The analysis focused on changes in Top-1 evidence, Top-K neighbourhood, similarity scores, ranking and score separation while keeping the retrieval setup otherwise consistent.

### Experiment 1: Synthetic Corpus — Global Transformations
Conducted a controlled experiment using the same 25 benchmark questions across Raw, Mean-Centered, Whitened and Reduced representations to understand how global transformation of the embedding space influences retrieval behaviour.

### Observed:
Compared Raw against Centered, Whitened and Reduced retrieval for the same questions, examining Top-1 evidence, Top-K composition, ranking and similarity scores.
Mean-centering generally produced lower cosine similarity scores than Raw, while the subsequent transformations tended to increase scores relative to the Centered representation.
Top-1 retrieval remained largely stable for simple/direct questions, while the Top-K neighbourhood showed more substantial changes in scores and, in some cases, ranking.
Questions requiring multiple chunks showed mixed behaviour across transformations, with relevant evidence not consistently appearing as Top-1 across all four representations.
Some transformations produced substantially larger Top-1/Top-2 score gaps without necessarily changing the underlying correctness of the retrieved evidence.

### Interpretation:
Global transformations can substantially modify the retrieval geometry even when the actual Top-1 retrieval decision remains unchanged.
A larger Top-1/Top-2 score gap should not by itself be interpreted as retrieval improvement; the retrieved evidence must also be correct.
Intrinsic dimensionality reduction serves a different objective from mean-centering and whitening: it primarily seeks to reduce representation dimensionality while preserving the useful information required for retrieval.
Hubness was kept outside the scope of these experiments because it was not observed to be sufficiently significant in this corpus to warrant treatment as a primary explanatory factor.
LLM support can be useful for identifying the correct answer-bearing chunk(s) for validation; however, for a controlled experiment of this size, manually comparing the retrieval results across transformations is simpler and provides direct validation.
The results suggest that the value of global transformations is likely to be condition-dependent, particularly for complex or distributed-answer retrieval, rather than uniformly beneficial across all queries.

### Technical conclusion:
Global geometric transformations can reshape similarity scores, score separation and the Top-K neighbourhood without necessarily improving retrieval correctness. The primary effect observed is therefore geometric rather than consistently substantive in terms of retrieval accuracy. Mean-centering and whitening are relevant to investigating geometric discrimination, while intrinsic dimensionality reduction addresses representation efficiency subject to preservation of retrieval behaviour.

### Capability demonstrated:
Ability to design controlled global-geometry experiments and distinguish changes in embedding geometry from actual retrieval improvement, while evaluating Top-1 correctness, Top-K behaviour, score separation and dimensionality preservation.

## Experiment 2: B01 — Global Transformations
Repeated the global-geometry analysis on B01, the earlier HR corpus with relatively homogeneous information, using representative retrieval cases across Raw, Mean-Centered, Whitened and Reduced representations to determine whether global transformations produced meaningful retrieval improvement.

### Observed:
Across the four representative cases, transformations changed similarity scores and, in some cases, Top-1/Top-2 score separation.
The Top-1 evidence remained largely stable, with no consistent movement towards a demonstrably better retrieval outcome.
Individual cases showed substantial changes in score gaps without corresponding improvement in the correctness of the retrieved evidence.
The transformations therefore produced more visible changes in retrieval geometry and competitive neighbourhood than in actual retrieval accuracy.

### Interpretation:
For the B01 cases examined, global transformations did not demonstrate a consistent retrieval improvement.
The results reinforce the distinction between geometric change and substantive retrieval improvement.
The relatively homogeneous nature of B01 provides limited evidence that global geometric transformation alone can address its retrieval exceptions.
Applying a global intervention to address difficult cases therefore requires regression validation against the existing successful retrieval population.

### Technical conclusion:
For the B01 cases examined, global transformations altered the numerical retrieval landscape but did not provide evidence of consistent improvement in retrieval correctness. Their practical value therefore appears dependent on the characteristics of the retrieval problem rather than inherently beneficial as a global intervention.

### Capability demonstrated:
Ability to validate a global-geometry hypothesis across different corpus characteristics and critically distinguish score/geometry changes from actual retrieval improvement, including consideration of potential regression when applying a global transformation.

### Evidence:
`/home/surendran/LangChain/Semantics@Suren/S@S_Module 4 Experiments/Output_Files`

# Module 5
Representation Learning

## Introduction

Representation learning studies how an embedding model learns to encode semantic relationships into its representation space. The module focused on understanding how different learning objectives can deliberately modify an existing embedding model and how those modifications affect local retrieval behaviour and the broader embedding geometry.

### Experiment 1 : B01 Corpus & Benchmark
The B01 corpus is the HR-domain corpus used as the primary environment for the representation-learning intervention experiments. A benchmark of 150 HR queries was used to examine retrieval behaviour across the corpus. Five targeted retrieval cases were selected from this benchmark for representation intervention, while the complete 150-query set was retained for evaluating the broader impact of each modified representation.

A pretrained `all-MiniLM-L6-v2` embedding model was used as the E1 baseline. Five targeted B01 retrieval cases were selected where representation-related intervention was considered appropriate.
Five independent representation-learning interventions were tested:
- Contrastive Learning
- Triplet Loss
- InfoNCE
- Hard Negative
- Angular Margin
Each intervention was applied independently to the same selected cases. The modified model was then used to regenerate embeddings for the complete B01 corpus, and the same 150-query benchmark was used for evaluation.
Synthetic data was kept outside the training process and reserved for subsequent held-out validation.

### Observed

**Contrastive Learning**  
Local: Strong correction of targeted rankings; positive chunks gained substantially in cosine. Negative and neighbouring Top-5 chunks also moved.  
Global: High impact — noticeable upward shift in overall cosine geometry.

**Triplet Loss**  
Local: Targeted positive–negative separation was achieved with the 0.10 margin. Other Top-5 chunks still showed some movement.  
Global: Low impact — overall geometry remained comparatively stable.

**InfoNCE**  
Local: Correct chunks generally strengthened and ranking improved, but movement was relatively modest.  
Global: Lowest observed impact — global cosine geometry remained comparatively stable.

**Hard Negative**  
Local: Explicitly penalised competitors moved away, often out of the Top-5; other candidates consequently reshuffled.  
Global: High impact — noticeable downward shift in overall cosine geometry.

**Angular Margin**  
Local: Very strong strengthening of targeted positive relationships; correct chunks moved decisively to Rank 1. Other chunks also moved.  
Global: Very high impact — substantial upward shift in overall cosine geometry.

**Overall observation:**  
All five interventions achieved their intended local objective to varying degrees, but none remained purely local. Changes propagated to other chunks and to the broader representation space. The magnitude and direction of this propagation differed by learning objective.

The interventions successfully changed the intended local query–chunk relationships, including ranking corrections and increases in positive cosine similarity.
The changes were not confined to the targeted relationships. Other chunks within the local Top-5 neighbourhood also changed in score and/or ranking, and the overall B01 cosine distribution changed to different degrees depending on the intervention.
The five objectives produced visibly different patterns of local and global movement. In this experiment, Contrastive Learning and Angular Margin produced strong geometric movement, Hard Negative produced pronounced displacement of explicitly penalised competitors, while Triplet and InfoNCE produced comparatively more restrained global movement.

### Interpretation

Fine-tuning the embedding model modifies its shared parameters rather than individual embedding vectors. Consequently, a targeted semantic intervention can propagate through the representation space and affect embeddings and retrieval relationships that were not directly included in the training examples.
Different learning objectives impose different geometric pressures on the shared representation. Therefore, improving a targeted retrieval relationship does not imply that the broader representation space remains unchanged.
The experiment also demonstrated that cosine-score magnitude and retrieval quality are not synonymous. A substantial increase in cosine can occur 
together with substantial global geometric movement.

### Technical conclusion

Representation learning is a model-level intervention. The correct unit of change is the embedding model and its parameters; the resulting embeddings and vector index are regenerated artifacts of that model.
A representation intervention should therefore be evaluated at both local and global levels:
- whether the intended semantic relationship improved; and
- what collateral movement occurred in the surrounding neighbourhood and broader embedding space.
For a production/BAU setting, a modified model should be treated as a candidate model and validated offline against the existing benchmark before being promoted. Its corresponding embeddings must be regenerated and maintained with the model version to preserve representation lineage and auditability.

#### Experiment 2 — Synthetic Corpus & Benchmark

The synthetic corpus is a small, controlled corpus designed to provide a common evaluation environment for comparing representation behaviour. It contains 50 corpus chunks and 25 benchmark questions, including both direct-answer and distributed-evidence cases. The corpus was kept outside the representation-learning process and used only as a held-out evaluation environment to observe how the five modified representations behaved relative to the original all-MiniLM-L6-v2 representation.

#### Observations : 

The six representations produced substantially different cosine-score distributions - because of the change to global geometry though the synthetic corpus was not target database on the modifications that were carried out.

| Representation        | Overall Cosine Mean | Δ vs Base | Rank-1 Mean | Rank-2 Mean | Rank-3 Mean | Rank-4 Mean | Rank-5 Mean |
| --------------------- | ------------------: | --------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| **Base — all-MiniLM** |            0.439788 |         — |    0.715787 |    0.472698 |    0.380585 |    0.328760 |    0.301111 |
| **Contrastive**       |            0.480909 | +0.041121 |    0.749020 |    0.520058 |    0.425508 |    0.368402 |    0.341558 |
| **Triplet**           |            0.437893 | −0.001895 |    0.703612 |    0.476965 |    0.383553 |    0.325102 |    0.300235 |
| **InfoNCE**           |            0.446053 | +0.006265 |    0.723412 |    0.483201 |    0.391851 |    0.328708 |    0.303091 |
| **Hard Negative**     |            0.416874 | −0.022914 |    0.681919 |    0.454126 |    0.361823 |    0.305426 |    0.281073 |
| **Angular Margin**    |            0.552291 | +0.112503 |    0.789862 |    0.588668 |    0.506056 |    0.453193 |    0.423677 |

The interventions therefore did not produce a uniform change in representation geometry. Contrastive Learning and Angular Margin produced upward shifts across the Top-5 score profile, with Angular Margin producing the largest shift. Triplet remained very close to the base geometry, while InfoNCE produced a comparatively modest upward movement. Hard Negative produced a downward shift across the score distribution.
The number of Top-1 results crossing the 0.75 cosine threshold also varied:

| Representation     | Top-1 ≥ 0.75 | Percentage |
| ------------------ | -----------: | ---------: |
| **Base**           |      12 / 25 |        48% |
| **Contrastive**    |      17 / 25 |        68% |
| **Triplet**        |      11 / 25 |        44% |
| **InfoNCE**        |      13 / 25 |        52% |
| **Hard Negative**  |      10 / 25 |        40% |
| **Angular Margin** |      20 / 25 |        80% |

Angular Margin therefore produced the largest number of Top-1 scores above the threshold, while Hard Negative produced the fewest.
However, the individual retrieval results showed that cosine-score magnitude did not directly determine retrieval correctness. Hard Negative, for example, reduced the overall cosine distribution while still correcting many local Top-1 retrieval decisions.
A recurring pattern across the intervention outputs was the displacement of semantically neighbouring chunks. In the base representation, several questions were resolved in favour of a neighbouring chunk; the intervention representations frequently moved the answer-bearing chunk into Rank 1. This indicates that the interventions affected local semantic discrimination, rather than simply increasing the confidence of an already-correct retrieval.
The distributed-answer questions also showed a change in Top-K evidence coverage. The intervention representations generally brought more answer-bearing chunks into the Top-5 neighbourhood, although complete coverage was not achieved consistently for every question.

#### Interpretation 

This secondary experiment demonstrates that the effect of representation learning extends beyond the specific positive-negative relationships used during intervention.
When the modified models were evaluated against an independent synthetic corpus, their altered representation spaces produced different retrieval rankings, score distributions, and local neighbourhoods.
The experiment therefore provides evidence of representation-space propagation: a model-level intervention applied to selected examples can alter retrieval behaviour for queries and chunks outside those specific training relationships.
At the same time, the experiment reinforces an important distinction between geometric movement and retrieval improvement.
An increase in cosine score does not by itself establish better retrieval. Conversely, a reduction in cosine magnitude does not necessarily imply worse retrieval. What matters operationally is whether the relevant evidence is ranked appropriately and whether the surrounding retrieval neighbourhood remains acceptable.
The experiment also exposed a persistent local-neighbour phenomenon. Some base retrieval errors were not arbitrary failures; the representation frequently selected a semantically adjacent chunk instead of the answer-bearing chunk. The interventions changed this local competition in different ways.

#### Technical conclusion

The secondary evaluation demonstrates that targeted representation learning can produce measurable changes beyond the training examples used for intervention.
The resulting effect should therefore be evaluated at three levels:
Local relationship — did the intended positive-negative relationship improve?
Retrieval behaviour — did Top-1 identity, ranking, and Top-K evidence coverage improve?
Global geometry — how did the intervention alter the broader cosine-score distribution and retrieval neighbourhood?
The experiment also establishes that cosine-score elevation should not be used as a standalone indicator of representation quality. Score magnitude, ranking behaviour, evidence correctness, and collateral movement provide different views of the modified representation.
The synthetic corpus therefore serves as a controlled secondary validation environment for observing whether a representation intervention remains local or propagates into broader retrieval behaviour.

### Capability demonstrated

Demonstrated the ability to:

- identify retrieval failures where representation intervention is appropriate;
- formulate targeted representation-learning objectives;
- apply different learning objectives to a pretrained embedding model;
- observe local ranking and cosine changes after intervention;
- evaluate collateral effects on the broader embedding space;
- distinguish representation-level intervention from retrieval/control-layer changes; and
- maintain model-to-embedding lineage for controlled model evolution.

### Evidence

Experiment outputs and retrieval benchmark results:
`/home/surendran/LangChain/Semantics@Suren/S@S_Module 5 Experiments/Output_Files`

# Module 6 : Retrieval behavior

## Experiment 1: ColBERT-style / Late-Interaction Retrieval

A controlled 12-query experiment was conducted to understand ColBERT-style late-interaction retrieval and how token-level query–chunk interaction differs from conventional single-vector dense retrieval.
The same controlled employee-policy corpus and query set were evaluated using dense cosine retrieval, MaxSim-based retrieval, and the final ColBERT-style Top-3 retrieval.
The experiment focused on Top-3 evidence behaviour, rather than Top-1 correctness alone, to determine whether answer-bearing and supporting evidence was retained within the retrieval neighbourhood and whether late interaction changed the ordering of semantically related chunks.

### Observed:
Dense retrieval, MaxSim and ColBERT-style retrieval frequently retrieved the same general evidence neighbourhood, but their ordering of individual chunks differed.
Several queries demonstrated fine-grained ranking changes where a more precise answer-bearing chunk was promoted over a semantically related neighbouring chunk.
Q05 and Q07 provided clear examples where ColBERT promoted the more specific answer-bearing chunk to Rank 1.
Q11 demonstrated the value of Top-3 evaluation for a multi-component query: ColBERT brought both required evidence components into the Top-3 neighbourhood.
Q12 showed that ColBERT restored two directly answer-bearing chunks to the first two positions after the intermediate MaxSim ranking had separated them.
ColBERT did not improve every retrieval. In Q08 and Q09, the correct answer-bearing chunk remained within Top-3 but was displaced from Rank 1 by supporting or semantically related evidence.
Q02 demonstrated that the final ColBERT ranking can differ from the intermediate MaxSim result and may lose evidence that was present in the earlier Top-3.
Top-3 evidence coverage provided a more informative evaluation than Top-1 alone for queries requiring multiple or supporting pieces of evidence.

### Interpretation:
Late interaction changes retrieval behaviour by allowing finer-grained query–chunk interaction than a single query embedding compared against a single chunk embedding.
The experiment showed that this additional interaction can help distinguish between semantically related chunks when broad topical similarity is insufficient. However, the additional interaction does not guarantee that the most directly answer-bearing chunk will always receive the highest rank.
The usefulness of ColBERT was therefore query-dependent. Its clearest value appeared where semantic neighbourhoods were crowded or where multiple pieces of evidence contributed to the answer.

### Technical conclusion:
ColBERT-style late interaction should be evaluated not only through Top-1 accuracy but also through Top-K evidence coverage, evidence ordering, and fine-grained discrimination between competing chunks.
The experiment demonstrated that late interaction can alter the composition and ordering of the retrieval neighbourhood, but does not inherently guarantee better retrieval for every query.
LLM-based evaluation was used as a downstream qualitative check. It showed that retrieval differences do not map deterministically to answer differences because generation introduces an additional interpretation variable. Therefore, retrieval quality was evaluated primarily at the evidence level, using Top-K answer-bearing and supporting evidence coverage; LLM responses were treated as secondary evidence rather than the retrieval metric.

### Capability demonstrated:
Ability to analyse late-interaction retrieval at the evidence level by comparing dense, MaxSim and ColBERT-style retrieval; distinguish Top-1 correctness from Top-K evidence coverage; and identify where token-level interaction changes retrieval behaviour.

## Experiment 2: Lexical Retrieval — TF-IDF vs BM25 as Foundations for Hybrid Retrieval
A controlled lexical-retrieval experiment was conducted over the same 161-chunk corpus using two classical lexical retrieval methods: TF-IDF and BM25.
Both methods used the same corpus, chunk IDs, chunk text, tokenizer, queries and Top-K value. Only the retrieval/scoring method changed.
The experiment also exposed term frequency, document frequency, IDF and positional inverted-index information to understand the lexical evidence contributing to retrieval.
The experiment was intentionally designed as a foundation for understanding hybrid retrieval, rather than as a hybrid retriever itself. The script does not combine lexical and dense scores; it isolates the lexical retrieval component.

## Observed:
TF-IDF and BM25 operate over the same lexical representation but use different scoring mechanisms.
TF-IDF represents the query and chunks through weighted lexical vectors and retrieves using cosine similarity.
BM25 performs lexical relevance scoring using term frequency, inverse document frequency and document-length normalization.
The experiment exposed query terms, corpus document frequency, IDF values and term positions to make the lexical retrieval process inspectable.
The same Top-K retrieval procedure was applied to both methods, allowing their ranking behaviour to be compared without changing the corpus or preprocessing.
The positional inverted index was maintained separately as a diagnostic structure. Its existence did not imply that TF-IDF or BM25 automatically used term proximity in their scoring.

### Interpretation:
The experiment established that lexical retrieval is not a single mechanism. Different lexical scoring functions can produce different rankings even when they operate on the same tokens and corpus.
TF-IDF and BM25 therefore provide two different views of lexical relevance. Understanding this distinction is important before combining lexical retrieval with dense semantic retrieval.
The experiment also established an important architectural separation: index representation, scoring mechanism and retrieval ranking are related but distinct concepts.

### Technical conclusion:
TF-IDF and BM25 provide complementary lexical retrieval mechanisms that can serve as the lexical component of a hybrid retrieval architecture.
However, this experiment by itself does not demonstrate hybrid retrieval because no lexical and dense scores were combined and no hybrid ranking strategy was evaluated.
The experiment therefore establishes the lexical retrieval foundation required to understand a subsequent hybrid architecture in which lexical and semantic retrieval signals may be combined.

### Capability demonstrated:
Ability to construct a controlled lexical-retrieval experiment, maintain common experimental controls across TF-IDF and BM25, inspect lexical term statistics and inverted-index evidence, and distinguish lexical retrieval mechanisms from the broader hybrid-retrieval architecture.

### Evidence:
Experiment outputs and retrieval benchmark results:
`/home/surendran/LangChain/Semantics@Suren/S@S_Module 6 Experiments/`

# Module 7 : Beyond Euclidean Thinking

## Experiment 1: Dimensionsality reduction techniques
A controlled dimensionality-reduction experiment was conducted using the existing employee-policy embedding corpus to investigate whether 2-D projections could provide useful insight into the geometric structure of high-dimensional semantic representations.
The experiment was designed around a practical retrieval question rather than visualization alone. A query and the available policy chunks were projected into 2-D using t-SNE and UMAP, and the resulting spatial relationships were compared with the original high-dimensional cosine-similarity relationships and retrieval rankings.
The purpose was not to determine whether a visually attractive projection could be produced, but to investigate whether the projection provided additional interpretable evidence about semantic neighbourhoods, local structure, broader organization, or difficult retrieval cases that could not be obtained directly from the original embedding-space measurements.
The experiment also deliberately maintained the distinction between the original embedding space and its 2-D projection. The Top-K retrieval ranking was determined from the original high-dimensional embedding space; the t-SNE and UMAP plots were treated only as analytical views of the same representations.

## Observed:
The 2-D projections did not consistently reproduce the original cosine-similarity ranking.
In one retrieval case (Q01) , a chunk that ranked fifth by cosine similarity in the original 1024-dimensional MXBAI embedding space appeared relatively distant from the query in both t-SNE and UMAP. This demonstrated directly that apparent 2-D proximity cannot be used to reconstruct or infer the original retrieval ranking.
The experiment also showed that t-SNE and UMAP can produce different apparent spatial organizations from the same underlying embedding set. This reinforced the distinction between the original embedding geometry and the properties emphasized by a particular dimensionality-reduction method.
The t-SNE projection emphasized local neighbourhood relationships. This made it useful for asking whether semantically related points appeared within a local neighbourhood, but the broader arrangement of points could not be interpreted as a faithful representation of all global distances.
UMAP also exposed local neighbourhood relationships while providing a different view of broader structural organization. However, apparent regional or cluster-level separation in the UMAP projection could not automatically be interpreted as proof of equivalent separation in the original embedding space.
A second retrieval case (Q11) involving a multi-component annual-leave query further demonstrated that cosine ranking, semantic completeness of the retrieved chunk, and apparent 2-D proximity are distinct properties. Chunks that appeared useful from a semantic-answer perspective did not necessarily occupy the same positions that cosine similarity alone would suggest.
The experiments therefore showed that dimensionality reduction can expose relationships that are useful for inspection, but the visual relationships must not be substituted for the original similarity measurements.
The existing HR corpus also revealed an important limitation: with a very small number of chunks and a narrow semantic domain, the visualizations did not consistently provide enough additional information to justify treating them as a primary analytical tool for retrieval.
This raised a broader experimental question: whether the limited usefulness observed so far was primarily a consequence of the corpus and task, rather than an inherent limitation of t-SNE or UMAP.

## Interpretation:
The experiment established that dimensionality reduction should be treated as an analytical lens rather than a reconstruction of the original semantic space.
PCA, t-SNE and UMAP each emphasize different properties of the underlying representation. PCA is oriented toward directions of maximum variance; t-SNE emphasizes local neighbourhood relationships; UMAP preserves local relationships while attempting to retain more broader structural organization.
Consequently, a projection can preserve one property of the original representation while distorting another.
The experiment also reinforced an important distinction between representation geometry and visualization of geometry. The actual computational geometry remains in the original high-dimensional embedding space, where cosine similarity, distances and neighbourhood relationships can be measured directly. The 2-D projection is an additional interpretive layer intended to make selected aspects of that geometry more accessible to human inspection.
Therefore, a disagreement between 2-D proximity and cosine ranking is not evidence that either representation is necessarily incorrect. They are measurements or representations at different levels, with different preservation objectives.
The experiments also established that a visible cluster in a 2-D projection should not automatically be interpreted as proof of an equivalent semantic cluster in the original embedding space.
The initial HR-corpus experiment therefore did not justify the broad conclusion that t-SNE or UMAP are inherently unhelpful. Instead, it exposed a more precise question for further experimentation:
Under what dataset structure and analytical task do t-SNE and UMAP provide useful information beyond direct inspection of the original embedding geometry?
This question became the basis for the next controlled experiment using the 20 Newsgroups dataset, where multiple related semantic categories provide a richer environment for examining local neighbourhoods and broader structural organization.

## Technical conclusion:

The first experiment demonstrated that t-SNE and UMAP should not be used as substitutes for the original high-dimensional embedding space or for retrieval metrics.
Their primary value in this context is diagnostic: they can provide a human-interpretable view of selected structural properties of an embedding space and can help formulate questions about local neighbourhoods, regional organization and geometric behaviour.
However, the experiment also demonstrated that 2-D proximity does not imply original cosine proximity, retrieval rank, or complete semantic similarity.
The experiment therefore established the following technical principle:
A dimensionality-reduction projection preserves selected properties of an embedding space; it does not preserve the entire original geometry.
The initial HR-corpus results were insufficient to determine the broader practical utility of t-SNE and UMAP. A larger and more structurally varied dataset is required before drawing conclusions about when one technique provides information that the other does not.
The experiment remains open for further evidence.

## Capability demonstrated:

Ability to analyse high-dimensional embedding geometry through dimensionality-reduction techniques without confusing projected geometry with the original embedding space.
Ability to distinguish:
original cosine similarity from 2-D proximity,
retrieval ranking from visual neighbourhood,
local structure from broader organization,
variance-oriented analysis from neighbourhood-oriented analysis,
semantic interpretation from geometric visualization.
Ability to recognize when an experiment has demonstrated a limitation of the current dataset or analytical setup rather than prematurely generalizing that limitation to the underlying technique.
Ability to design a follow-up experiment specifically to test whether a richer semantic dataset exposes additional utility from t-SNE and UMAP.

## Experiment 2: t-SNE and UMAP on a Richer Real-World Corpus
TDimensionality Reduction on 20 Newsgroups — Comparing t-SNE and UMAP as Geometric Lenses

### Introduction
A richer real-world corpus was introduced to extend the earlier small HR-policy dimensionality-reduction experiment. The 20 Newsgroups dataset contained 2,343 documents from four categories:
comp.graphics
comp.sys.ibm.pc.hardware
comp.sys.mac.hardware
sci.electronics
The same frozen MXBAI embedding model and 1,024-dimensional representation were used throughout the experiment. t-SNE and UMAP were applied to the same embeddings using cosine distance, allowing the two methods to be compared as different analytical lenses over the same underlying representation.
The experiment deliberately followed an observation-first protocol. The initial projections were inspected without category labels or predefined expectations. Metadata was revealed only after the raw geometry had been observed.

### Observed
The richer corpus made the difference between the two dimensionality-reduction methods much more visible than the earlier small HR corpus.
The unlabeled t-SNE projection produced a broad, relatively continuous-looking 2-D organization with substantial local structure and overlap.
The UMAP projection produced a more compact and regionally structured organization. A small, visually isolated group appeared far away from the main UMAP structure.
Rather than immediately interpreting that group as a semantic outlier, the corresponding points were traced back to the original documents. The isolated group consisted of documents containing effectively null/no-information content, including empty strings, whitespace/newline content, or separator-like content. Importantly, these documents came from all four metadata categories.
This became an important experimental observation:
A visually isolated region in a dimensionality-reduction projection does not automatically represent a semantic cluster. It can expose a data or preprocessing artifact.
The category labels were subsequently revealed.
The four categories did not form four clean geometric compartments. Instead, they showed substantial spatial overlap while also exhibiting some regional tendencies. This was consistent with the nature of human discussion data, where nominally different categories can contain overlapping concepts and contexts.
The category-focused t-SNE views further showed that each metadata category occupies a broad region rather than a single compact island. No category could reasonably be treated as equivalent to one isolated semantic region.
The final t-SNE/UMAP comparison made the distinction between the two lenses much clearer:
Same 1,024-D representation → different dimensionality-reduction objective → visibly different 2-D organization.

### Interpretation
The experiment provided direct evidence for the earlier conceptual distinction:
t-SNE → emphasizes local neighbourhood structure.
UMAP → preserves local structure while also retaining broader/regional organization.
The richer corpus made these differences much easier to see than the small HR experiment.
The experiment also reinforced a critical principle:
2-D projection ≠ original embedding space.
The apparent area occupied by a projection has no direct interpretation as “amount of information.” UMAP occupying a compact portion of its plotting canvas does not mean that the corpus represents a small fraction of some larger semantic universe. Likewise, t-SNE occupying a larger visual area does not mean that it contains more information.
The axes of the projections are newly constructed coordinates. Their primary value is relational, not as absolute semantic measurements.
The category overlays also demonstrated another important distinction:
Metadata taxonomy ≠ learned geometric organization.
The four dataset categories provide external labels, but the embedding representation does not reproduce those labels as four perfectly separated regions.
The null-data island provided the strongest unexpected evidence in the experiment. What initially looked like a possible semantic structure turned out, after tracing the points to their source documents, to be associated with the content quality of the underlying data.

### Technical Conclusion
The experiment did not establish that t-SNE or UMAP can diagnose retrieval errors directly.
This distinction is important.
A retrieval problem may first be identified through suspicious Top-K behaviour. A t-SNE or UMAP projection can then provide an additional geometric lens for investigating that behaviour.
The chain is:
Suspicious retrieval behaviour → projection → additional geometric observation → hypothesis → return to original high-dimensional geometry and document evidence
A disagreement between a 2-D projection and original retrieval ranking does not prove that the retrieval system is wrong. The original 1,024-dimensional representation remains the computational space governing retrieval.
Therefore:
t-SNE/UMAP can deepen an investigation into suspicious retrieval behaviour, but the projection itself does not establish the diagnosis.
This experiment demonstrated that distinction but did not perform a complete retrieval-failure diagnosis on the 20 Newsgroups corpus.

### Practical Value Demonstrated
The experiment produced three grounded practical benefits.
1. Data-quality diagnosis
The clearest practical result was the discovery of effectively null/no-information documents through an isolated UMAP region.
Embedding corpus → visual anomaly → trace points → inspect source documents → identify data/preprocessing issue
This is useful as a complementary quality-control step before retrieval.
2. Retrieval/representation investigation
If Top-K retrieval behaviour is already suspicious, t-SNE/UMAP can provide another geometric view of the relevant documents and neighbourhoods.
They do not prove the cause. They can help deepen the investigation and formulate hypotheses that must then be checked against the original high-dimensional geometry and document evidence.
3. Understanding metadata versus representation structure
The category overlays showed that externally assigned categories can exhibit spatial tendencies without becoming clean geometric compartments.
This helps prevent the incorrect assumption that: dataset categories → automatically equivalent to semantic clusters.

### Capability Demonstrated

This experiment demonstrated the ability to:
apply t-SNE and UMAP to a substantial real-world embedding corpus;
compare two projections of the same frozen representation;
distinguish projected geometry from original high-dimensional geometry;
inspect raw geometry before revealing metadata;
trace an unexpected geometric structure back to source documents;
recognize a data/preprocessing artifact without mistaking it for semantic structure;
distinguish metadata categories from learned semantic organization;
use dimensionality reduction as a diagnostic lens rather than a retrieval mechanism;
avoid treating a 2-D visualization as proof of original-space relationships;
recognize the practical boundary between visual observation, hypothesis, and actual diagnosis.

### Final learning from experiment 2 :
The most important practical lesson from this experiment was not simply that t-SNE and UMAP produce different pictures.
It was:
A projection can tell us where to look, but the original representation and the underlying data must tell us what is actually happening.
And that completes the richer evidence experiment for Module 7.2.

## Experiment 3: Hyperbolic Representation of a Real-World Hierarchy
Hyperbolic/Poincaré Representation Learning on Google Product Taxonomy — Learning and Validating Hierarchical Geometry

### Introduction

A genuinely hierarchical real-world dataset was introduced to study hyperbolic representation directly rather than artificially constructing a Euclidean-versus-hyperbolic benchmark.
The Google Product Taxonomy contained 5,595 categories organized across depths 0–6, with 21 root categories, 876 parent categories, and 5,574 parent-child relationships. The taxonomy also contained substantial branching, with some parent categories having many direct children.
The experiment began with an important conceptual distinction developed during Module 7.3:
Structure → what relationships exist
Dimensionality → how much representational capacity is available
Geometry → how those relationships are spatially expressed
The taxonomy therefore supplied the hierarchical structure, while hyperbolic geometry provided the geometric framework in which that structure would be represented.
The experiment was deliberately not designed to prove that hyperbolic geometry is universally better than Euclidean geometry. The objective was to understand how a real hierarchical structure can be learned and expressed in hyperbolic space.
The experiment used a Poincaré-ball representation with fixed curvature and trainable 10-dimensional coordinates. Unlike the earlier frozen MXBAI embedding workflow, the representation was learned directly from the hierarchical structure through optimization.

### Observed

The initial taxonomy inspection established that the dataset contained a genuine multi-level branching hierarchy rather than merely a collection of semantically related texts.
The structural inspection showed:
5,595 categories;
21 root categories;
maximum depth of 6;
876 parent categories with children;
5,574 parent-child relationships;
maximum direct branching of 79 children under a single parent;
no duplicate category IDs;
no duplicate category paths;
no missing parent paths.
This provided a structurally appropriate substrate for investigating hyperbolic representation.
The taxonomy was then converted into explicit structural input consisting of nodes and parent-child relationships.
The first major observation from the representation stage was that hyperbolic representation required a learning process rather than simply passing the taxonomy through an already-trained semantic embedding model.
The learning process involved trainable hyperbolic coordinates, hyperbolic distance, a structural training objective, optimization, and repeated epochs.
This created an important distinction from the earlier MXBAI workflow:
Pre-trained semantic embedding: text → pre-trained model → vector
Hyperbolic structural representation: hierarchy → trainable hyperbolic representation → optimization → learned coordinates
The resulting representation had shape: (5595, 10)
The .npy output was only the storage format. The fact that the result was stored as numerical vectors did not make it a conventional Euclidean representation. The geometric framework and training process determined how those numbers were interpreted.
The first validation stage examined parent-child relationships.
Across a sample of 5,000 parent-child pairs, the mean hyperbolic distance was approximately 0.530, with a median of approximately 0.386 and a standard deviation of approximately 0.466.
Sibling relationships were then examined separately. Siblings were defined structurally as categories sharing the same immediate parent rather than merely categories having the same depth.
Across 5,000 sampled sibling pairs, the mean distance was approximately 0.674, with a median of approximately 0.515 and a standard deviation of approximately 0.508.
Same-depth, different-branch relationships were then introduced as a separate structural comparison. This distinction was important because equal depth does not imply siblinghood.
The same-depth cross-branch populations were generally much farther apart than direct parent-child relationships, while also showing increasing variation at deeper levels. Depth 6, for example, showed a broad distribution with a mean of approximately 2.374 and a standard deviation of approximately 1.010.
Cross-level comparisons were then introduced to examine relationships between depth n and n-1, and between depth n and n-2.
A random non-parent/child population was also introduced as a broader reference. Its sampled mean distance was approximately 3.233.
Together, these observations showed a clear aggregate pattern:
direct hierarchical relationships were generally much more geometrically compact than broad cross-branch or random relationships. However, the distances were not fixed according to relationship type.
This led to an important question: if the hierarchy already establishes that two nodes are siblings, why do individual sibling distances vary?
Concrete path inspection was therefore added rather than relying only on population averages
A shallow path: Animals & Pet Supplies → Pet Supplies → Bird Supplies showed consecutive distances of approximately:
0.503 → 1.375
A deeper path through Arts & Entertainment showed substantially different distances between successive hierarchical levels, including very small distances near the end of the path.
A branching example under Apparel & Accessories showed that children sharing the same parent also occupied substantially different geometric positions.
These observations demonstrated that the learned representation did not translate hierarchy into a rigid distance ruler.

### Interpretation
The experiment provided direct evidence that hierarchical structure can be learned as a geometric configuration rather than as a set of fixed distance values.
The taxonomy determines structural relationships such as:
parent → child and sibling ↔ sibling
but it does not prescribe a numerical distance for each relationship.
The learning process positions all entities simultaneously within the hyperbolic space. The resulting coordinates therefore produce continuous geometric distances that vary according to the learned configuration.
The correct interpretation is:
Structure gives discrete relationship constraints → learning produces continuous geometric positions → geometry produces variable numerical distances.
Therefore, the sibling mean of approximately 0.674 should not be interpreted as:
“the sibling relationship has distance 0.674.”
It is the average of many individual sibling distances.
The standard deviation and percentile distribution describe the variation around that population-level summary.
This distinction was important because structural equivalence does not imply geometric equivalence.
Two pairs can both be siblings while having different geometric distances.
The experiment also established that semantic similarity was not responsible for this variation.
The taxonomy experiment did not calculate these distances from semantic similarity between category names. The learning chain was:
taxonomy structure → hyperbolic training → learned coordinates → hyperbolic distance
rather than:
category text → semantic embedding → cosine similarity
The variation therefore demonstrated that a structural representation can produce continuous geometric differences even without using semantic similarity as the training signal.
The broader structural context matters because the model learns the configuration of the hierarchy as a whole. Different nodes occupy different positions within different branches, depths, parent structures, and branching environments.
The same reasoning explains why depth does not translate into a simple fixed radial distance.
The origin-distance analysis was not monotonic with depth. Therefore the common intuition:
general concepts → centre
specific concepts → progressively outward
should be treated as an intuition rather than a universal rule for every hyperbolic embedding.
The deeper geometric principle is the expanding spatial capacity of hyperbolic space, not a mandatory depth-to-radius mapping.
The experiment also reinforced the distinction between dimensionality and geometry.
Increasing Euclidean dimensionality would provide additional representational capacity, but would not change the underlying Euclidean geometry.
Hyperbolic geometry instead changes the geometric rules themselves.
Therefore: More capacity does not automatically mean better geometric fit.
This was one of the central conceptual conclusions of the experiment.
The experiment also clarified that hyperbolic representation is not a replacement for Euclidean representation.
Euclidean and hyperbolic geometry are alternative geometric frameworks whose suitability depends on the structure and task.
The presence of hierarchy makes hyperbolic geometry relevant; it does not make it automatically superior.
The experiment therefore established hyperbolic geometry as:
one more geometric lens, not one more claim.

### Technical Conclusion

The experiment demonstrated that a real-world hierarchical structure can be represented through a learned Poincaré hyperbolic embedding in which structurally related entities generally occupy more compact geometric relationships than broad cross-branch or random entities.
The representation was learned directly from structural relationships rather than obtained from a frozen semantic embedding model.
The validation also demonstrated that hierarchical relationships should be interpreted as distributions of geometric distances rather than fixed distance values.
The chain is:
hierarchical structure → training constraints → learned hyperbolic coordinates → variable hyperbolic distances
Therefore:
structural relationship type ≠ fixed geometric distance
The distinction between population-level and individual-level analysis was also technically important.
The sequence used was:
population → distribution → individual examples
rather than inferring general behaviour from a small number of paths.
The concrete path inspection confirmed that the variation visible in the distributions was genuinely present at item level.
The experiment did not establish that the learned hyperbolic representation provides better retrieval than Euclidean embeddings.
Representation evaluation and retrieval evaluation remain separate.
The correct boundary is:
representation validation asks whether the intended structure is encoded; retrieval evaluation asks whether that representation produces useful retrieval behaviour.
A good structural representation does not automatically guarantee retrieval accuracy.
The experiment therefore remains a representation/geometry experiment, not a retrieval benchmark.

### Practical Value Demonstrated
#### Representation of hierarchical structure
A real-world hierarchy can be converted into a trainable hyperbolic representation rather than relying exclusively on a conventional frozen Euclidean semantic embedding.
#### Structural geometric validation
Different structural populations can be compared quantitatively:
parent-child, sibling, same-depth cross-branch, cross-level, and random relationships.
This provides a way to investigate whether the learned geometry reflects the structure supplied to the learning process.
#### Distribution-aware geometric analysis
Using mean, median, standard deviation, percentiles, and individual examples prevents a structural relationship from being reduced to a single average distance.
This is particularly useful when the learned geometry contains substantial item-level variation.
#### Structure-driven experimental expansion
The experiment demonstrated a practical research pattern:
initial observation → identify a legitimate question → expand the measurement scope → validate the new observation → inspect individual examples where aggregate statistics are insufficient.
The validation therefore evolved according to evidence rather than according to a fixed list of measurements.
#### Separation of structural and semantic signals
The experiment demonstrated that geometric structure can be learned from explicit hierarchical relationships without requiring semantic similarity as the training signal.
#### Boundary between representation and downstream behaviour
The experiment established a clean point at which representation analysis should stop and retrieval evaluation should begin, preventing claims about downstream performance from being inferred from structural geometry alone.

### Capability Demonstrated
This experiment demonstrated the ability to:
identify an appropriate real-world dataset from the structural nature of the problem;
inspect and validate hierarchical structure before modelling;
convert explicit hierarchical relationships into a learning substrate;
train and work with a Poincaré hyperbolic representation;
reason about geometry independently of dimensionality;
distinguish structural relationships from semantic similarity;
design multiple structural validation populations;
distinguish population-level statistics from item-level behaviour;
expand an experiment when observations expose a legitimate unanswered question;
trace aggregate geometric behaviour back to concrete structural examples;
avoid over-interpreting geometric visualizations or simplified hierarchy metaphors;
distinguish representation validation from retrieval validation;
interpret learned representations as geometric structures rather than merely numerical arrays;
and use a change in geometric framework as a principled research lens rather than assuming that a new geometry is universally superior.

### Final learning from Experiment 3:
The most important learning from this experiment was:
A hierarchy gives us the structural relationships; learning turns those relationships into a geometric configuration; and the resulting geometry expresses those relationships through continuous, variable distances rather than fixed numerical rules.
And the broader S@S insight is:
More representational capacity does not automatically provide better geometric fit. Sometimes the important question is not how much space we have, but what kind of space we are using to express the structure.

## Experiment 4A — DeepWalk on the Amazon Co-Purchase Graph

### Introductory database/experimental context first, then:
Experiment 4A — Dataset & Experimental Context
The experiment uses the Stanford SNAP Amazon co-purchasing network, derived from Amazon's “Customers Who Bought This Item Also Bought” relationship. Each node represents a product, and an undirected edge represents a co-purchasing relationship between two products.
The graph contains 334,863 products and 925,872 edges, with all nodes belonging to the largest connected component. The graph has an average degree of approximately 5.53, median degree 4, and maximum degree 549.
Amazon product-category information and community data were retained as independent reference information rather than being used to train the graph embeddings. The representations were learned from graph structure alone.

### Common training and traversal configuration
The initial DeepWalk experiment used a frozen configuration of 128-dimensional embeddings, 5 walks per node, walk length 20, context window 5, 5 negative samples, 3 epochs, random seed 42, and uniform random walks.
The exact training-run traversal was subsequently verified rather than assumed. Under the DeepWalk configuration, all 334,863 nodes and all 925,872 graph edges were traversed during the actual training walk generation. The 1,674,315 generated walks produced 31,811,985 edge traversals, with a median exposure of approximately 33 traversals per edge. Thus, the embedding was trained with exposure to the complete graph edge set under the frozen configuration.
For the subsequent Node2Vec experiment, the same graph, embedding dimension, number and length of walks, context window, negative samples, epochs, and random seed were retained. Only the walking strategy was changed through three p/q configurations.

The three Node2Vec configurations were:
Configuration	p	q	Walking behaviour
N2V-1	0.25	4.0	Local-biased
N2V-2	1.0	1.0	Unbiased
N2V-3	4.0	0.25	Outward-biased

The exact training-run traversal was also captured independently for each Node2Vec configuration. N2V-2 and N2V-3 traversed all 925,872 edges, giving 100% edge coverage. N2V-1 traversed 925,831 edges, leaving only 41 edges unseen, corresponding to 99.99557% edge coverage. All three configurations generated the same 1,674,315 walks, 33,486,300 node visits, and 31,811,985 edge traversals.
This establishes an important experimental condition: the overall training budget and graph exposure were kept effectively constant while the walking behaviour was varied through p and q. The small difference in edge coverage for N2V-1 is retained as an observed result rather than treated as a problem to be corrected.
The experimental objective was to determine what structural information could be represented in the learned embedding geometry, and whether changing the way graph structure was sampled during walking changed the resulting representation.

### Experiments
4A-1 — Accidental Investigation: Community 2/3 Outlier Analysis
4A-2 — First-Order / Direct Relationship Validation
4A-3 — Higher-Order / 2-Hop Relationship Validation
4A-4 — Node2Vec: Effect of Walking Strategy
N2V-1: p = 0.25, q = 4.0 — local-biased
N2V-2: p = 1.0, q = 1.0 — unbiased reference
N2V-3: p = 4.0, q = 0.25 — outward-biased

## Experiment 4A-1: Accidental Investigation — Community Outlier Analysis

### Introduction
During initial inspection of the Amazon community data, Community 2 and Community 3 appeared structurally similar, with Community 3 differing from Community 2 by one product. This led to an unplanned investigation of whether the differing product represented a meaningful structural or semantic outlier.

### Observed
Community 2 contained 27 nodes.
Community 3 contained 26 nodes.
Community 3 was Community 2 minus node 296858.
Node 296858 belonged to the category Books_Subjects_Mystery&Thrillers_PoliceProcedurals.
Its internal-community edge ratio was 1/3.
Other Community 2 members had equal or lower internal ratios.

### Interpretation
The apparently distinctive node was not uniquely anomalous based on the examined graph-connectivity measure. The investigation also showed that community membership could not be understood from category labels alone; graph structure contributed to the grouping.

### Technical Conclusion
The apparent community difference did not provide evidence that node 296858 was a structural outlier. Community membership reflected graph relationships rather than a simple categorical interpretation.

### Practical Value Demonstrated
The investigation demonstrated the value of tracing unexpected community structure back to the underlying graph rather than assuming that an apparent difference represents a semantic anomaly.

### Capability Demonstrated
Graph-structure inspection and anomaly investigation.

### Final learning from Experiment 4A-1:
An apparent community outlier must be validated against the underlying graph structure; visual or categorical differences alone are insufficient evidence of structural anomaly.

## Experiment 4A-2: DeepWalk — First-Order / Direct Relationship Validation

### Introduction
After training DeepWalk on the Amazon co-purchase graph, the first validation examined whether directly connected products were reflected as similar vectors in the learned embedding space.

### observed
Direct edges: mean cosine similarity 0.9549.
Random non-edges: mean cosine similarity 0.3433.
Mean difference: 0.6116.

### Interpretation
Directly connected products showed substantially higher cosine similarity than randomly selected non-connected products.

### Technical Conclusion
The DeepWalk representation captured first-order graph relationships in the learned Euclidean embedding geometry.

### Practical Value Demonstrated
Direct graph connectivity can be represented as proximity in a learned vector space without using product-category information during training.

### Capability Demonstrated
First-order structural representation learning.

### Final learning from Experiment 4A-2:
DeepWalk successfully encoded direct graph relationships into Euclidean embedding similarity.

## Experiment 4A-3: DeepWalk — Higher-Order / 2-Hop Relationship Validation

### Introduction
Having established that direct relationships were represented, the next experiment examined whether the embedding also captured relationships beyond immediate adjacency. Cosine similarity was measured for 5,000 exactly-2-hop non-edge pairs and 5,000 random non-edge pairs.

### Observed
Exactly-2-hop non-edge pairs: mean cosine 0.8942, median 0.9182.
Random non-edges: mean cosine 0.3440, median 0.3431.
Mean difference: 0.5503.

### Interpretation
Nodes connected through an intermediate node were substantially more similar in the learned embedding space than the sampled random non-edge pairs.

### Technical Conclusion
The DeepWalk representation captured higher-order graph structure, extending beyond direct adjacency.

### Practical Value Demonstrated
Graph relationships that are not explicitly represented by a direct edge can still emerge as proximity in the learned vector space.

### Capability Demonstrated
Higher-order structural representation learning.

### Final learning from Experiment 4A-3:
DeepWalk learns structural relationships through walk-based context, allowing higher-order graph relationships to emerge in the embedding geometry.

## Experiment 4A-4 — Node2Vec: Effect of Walking Strategy

### Introduction
This experiment extends the DeepWalk baseline by changing the graph-walking strategy while retaining the same downstream Skip-gram learning mechanism and essentially the same training budget.
Four walking configurations are considered:

Method	Walking strategy	Parameters
DeepWalk	Uniform random walk	—
N2V-1	Local-biased	p = 0.25, q = 4.0
N2V-2	Unbiased/reference	p = 1.0, q = 1.0
N2V-3	Outward-biased	p = 4.0, q = 0.25

The graph, embedding dimension (128), 5 walks per node, walk length (20), context window (5), negative samples (5), epochs (3), and seed (42) were retained.
The exact training-run traversal was measured rather than assumed. DeepWalk traversed all 925,872 edges. N2V-2 and N2V-3 also achieved 100% edge coverage, while N2V-1 traversed 925,831 edges, leaving only 41 edges unseen (99.99557% coverage).
The central question was:
If the way graph structure is sampled during walking changes, does the resulting learned embedding geometry also change?

### Observed
The same validation populations were used across the Node2Vec configurations: 5,000 direct-edge pairs, 5,000 random non-edge pairs, and 5,000 sampled exactly-2-hop non-edge pairs.
The DeepWalk results are included as the baseline reference.

Method	Direct-edge mean	Random mean	Direct − Random	2-hop mean	2-hop median	2-hop − Random
DeepWalk	0.954920	0.343285	0.611635	0.894213	0.918216	0.550260
N2V-1	0.938251	0.343583	0.594669	0.856988	0.884523	0.513405
N2V-2	0.954810	0.342999	0.611811	0.902325	0.922968	0.559326
N2V-3	0.957619	0.348806	0.608813	0.913743	0.934482	0.564937

For completeness, the Node2Vec direct-edge medians were:

N2V-1: 0.962200
N2V-2: 0.969335
N2V-3: 0.972524

The corresponding sampled 2-hop medians were:

N2V-1: 0.884523
N2V-2: 0.922968
N2V-3: 0.934482
Comparison across all four

DeepWalk vs N2V-2:
The two are very close in direct-edge representation. DeepWalk direct-edge mean = 0.954920, while N2V-2 = 0.954810. Their sampled 2-hop means are also close: 0.894213 vs 0.902325.

This provides a useful reference because p = q = 1 represents the unbiased Node2Vec setting.

DeepWalk vs N2V-1:
N2V-1 produced lower mean similarity for both direct edges and sampled 2-hop relationships than DeepWalk: 0.938251 vs 0.954920 for direct edges, and 0.856988 vs 0.894213 for 2-hop pairs.

DeepWalk vs N2V-3:
N2V-3 produced a slightly higher direct-edge mean than DeepWalk (0.957619 vs 0.954920) and a higher sampled 2-hop mean (0.913743 vs 0.894213).

Across the three Node2Vec configurations:
The largest separation appears in the sampled higher-order relationship signal. The 2-hop mean increases from 0.856988 in N2V-1 → 0.902325 in N2V-2 → 0.913743 in N2V-3.

The direct-edge means vary within a narrower range: 0.938251 → 0.954810 → 0.957619.

### Interpretation
The important observation is not simply which configuration has the largest number.
The important observation is that changing p/q changes the walking behaviour, which changes the node contexts presented to Skip-gram, which changes the resulting embedding geometry.

p/q
 ↓
walking behaviour
 ↓
node contexts experienced
 ↓
Skip-gram learning
 ↓
embedding geometry
 ↓
structural relationships reflected in cosine similarity

The direct structural signal remains strong across all four methods.
The larger variation occurs in the sampled higher-order relationship signal. This suggests that the walking strategy can influence how strongly broader graph structure becomes reflected in the learned geometry.
Under this particular experiment, N2V-3 produced the highest sampled 2-hop cosine similarity, while N2V-1 produced the lowest.
That is an experimental observation, not a universal ranking of graph-walking methods.

### Technical Conclusion
The four-way comparison demonstrates that graph-walking strategy is itself a meaningful part of graph representation learning.
DeepWalk and Node2Vec share the basic idea of converting graph structure into node-context sequences and learning embeddings from those contexts. Node2Vec adds control over the structural sampling behaviour through p and q.
Therefore:
Same graph + same downstream learning mechanism + different structural sampling → different learned embedding geometry.
The experiment also shows that the effect of changing the walking strategy is not necessarily uniform across relationship orders. In this experiment, the sampled higher-order signal changed more noticeably than the direct-edge signal.

### Practical Value Demonstrated
A graph embedding pipeline provides a controllable design choice at the walk-generation stage.
The practitioner can therefore influence what graph contexts the representation learner repeatedly encounters without changing the underlying graph or replacing the downstream learning mechanism.
This is useful when the representation needs to emphasize different aspects of graph structure.

### Capability Demonstrated
Controlled structural representation learning and comparative evaluation.
The experiment demonstrated the ability to:
Keep the graph and learning conditions largely fixed.
Change only the graph-walking strategy.
Measure exact graph exposure during the actual training run.
Train separate representations.
Validate what structural relationships were reflected in each representation.
Compare the resulting geometries against a common DeepWalk baseline.

### Final learning from Experiment 4A-4:
Node2Vec showed that changing how graph structure is sampled during walking can change how that structure is encoded in embedding geometry, even when the underlying graph and downstream Skip-gram learning mechanism remain the same.
And I would preserve this sentence because it captures the methodological lesson beautifully:
Traversal tells us what structural experience was presented to the learner; validation tells us what structural relationships were actually reflected in the learned representation.

## Evidence: 
`/home/surendran/LangChain/Semantics@Suren/S@S_Module 7 Experiments/`

