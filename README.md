# Ko-ToxicWebQ
**A Korean Toxic Web Query Dataset**
This repository contains projects and resources related to the paper:
**“Ko-ToxicWebQ: A Korean Toxic Web Query Dataset”**
Ko-ToxicWebQ is a large-scale dataset of real-world Korean web queries collected from Microsoft Bing search logs[(MS-MARCO-Web-Search)](https://github.com/microsoft/MS-MARCO-Web-Search). The dataset supports research in toxic query detection, obfuscation robustness, and AI safety for web-based search applications.
> ⚠️ Warning: This dataset contains toxic or offensive content. Please handle with care.
➡️ **The dataset is available at:**
[here](https://huggingface.co/datasets/TEAMREBOOTT-AI/Ko-ToxicWebQ)
---
## 📦 Dataset Overview
Ko-ToxicWebQ consists of **47,714** Korean web search queries sampled from Microsoft Bing logs. Each query is annotated with both a **toxicity category** and an **obfuscation type**.
### 🔎 Toxicity Categories (7)
Toxicity categories follow Korean regulatory standards (e.g., KCSC):

| Category                        | Description                                                                 | Count   | Ratio   |
|--------------------------------|-----------------------------------------------------------------------------|---------|---------|
| Clean                          | Queries with no overt toxicity                                              | 44,294  | 92.8%   |
| Explicit Sexual Content        | Explicit adult content, depictions of sexual activity, erotic literature/webtoons, sexual exploitation, objectification | 1,513   | 3.2%    |
| Copyright Infringement         | Illicit downloads/streams of copyrighted movies, dramas, music,games, novels, cartoons | 1,461   | 3.1%    |
| Others                         | Toxic queries not in the above (e.g., self-harm, hacking)                   | 191     | 0.4%    |
| Harmful Social Platform Content| Adult or inappropriate content on social/streaming platforms (e.g.,Twitter adult livestreams) often tied to digital sexual exploitation | 156     | 0.3%    |
| Prostitution                   | Online sex solicitation, red-light district info, compensated dating(illegal under Anti-Sex Trade laws) | 59      | 0.1%    |
| Gambling                       | Online sports betting, casino-style platforms, or any form of illegal gambling under Korean law | 40      | 0.1%    |

---

### 🌀 Obfuscation Types (3)
| Type                     | Description                                                                 | Ratio   |
|--------------------------|-----------------------------------------------------------------------------|---------|
| Abbreviation             | Shortened forms using consonants/vowels/partial words (non-English only)    | 23.0%   |
| Character-Level Manipulation | Decomposed syllables, typos, or typing Korean words with English keys to obscure meaning | 14.9%   |
| No Transformation        | Queries containing no obfuscation at all.                                   | 65.1%   |

Queries may include multiple obfuscation strategies or none at all.

---

## 🧭 Ethics Statement
We recognize that providing real examples of this content may be uncomfortable or distressing. These samples, however, are critical for developing robust detection systems that can mitigate harm and prevent the circulation of illicit content. We urge researchers to handle this dataset responsibly, with appropriate safeguards for user well-being.

Since conceptions of toxic or illegal content vary widely across jurisdictions, we emphasize that our annotations are grounded in South Korean laws and societal norms. We neither endorse nor condone the content found in these queries; our sole aim is to enable the development of more effective toxicity detection methods.
