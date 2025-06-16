# APPENDIX

### Detailed Taxonomy

**Gambling**: This category covers content related to illegal gambling, betting, and speculative games of chance. Online gambling is outlawed in Korea, so queries for casino sites, sports betting tips, private betting community links, or lottery manipulation schemes are considered harmful. The law explicitly prohibits "communications facilitating acts of gambling". Examples include searches like illegal toto, private sports betting, online casinos, etc. 


**Explicit Sexual Content**: Material that explicitly depicts, describes, or promotes sexual activity, or nudity. This includes pornographic content, explicit depictions of sexual activity, erotic literature, adult webtoons, sexual exploitation, and any form of digital sex crime materials (e.g., spy-cam videos). An exception is educational or informational discussions of sexual culture presented in a non-explicit context.


**Harmful Social Platform Content**: This category includes queries that seek offensive or sexually explicit content on social media and live-streaming platforms, such as Twitter and adult-oriented streaming services. It also includes searches for methods to bypass age verification or access restricted content. These queries often involve attempts to find sexually suggestive, exploitative, or inappropriate material that is not readily available due to platform restrictions. For example, the infamous 2020 “Nth Room” case (텔레그램 N번방 사건) — where sexual exploitation videos were circulated via private chat rooms — highlighted how social platforms can facilitate digital sexual exploitation.


**Copyright Infringement**: This category includes queries that attempt to illegally download, stream, or access copyrighted content without proper authorization. Examples of such content include movies, dramas, music, games, novels, and cartoons. Under South Korea’s Copyright Act, any activity involving the unauthorized sharing or distribution of copyrighted materials without the copyright holder’s permission falls under this category.
While watching or downloading copyrighted content is not always illegal, it is recommended that such content be accessed through legitimate and ethical means. Therefore, queries that suggest an intention to obtain copyrighted materials through unauthorized methods, such as torrenting, unlicensed streaming websites, webhard links, or file-sharing platforms, are also included in this category. This includes cases where copyrighted PDFs, e-books, or other digital media are shared and distributed without proper authorization.

**Prostitution**: Although prostitution-related content could be grouped under explicit sexual content, we single it out given its significance in Korean contexts. This includes "online sex solicitation, red-light district information, or compensated dating" queries. Such content is illegal under the Anti-Sex Trade laws and is a major focus of internet monitoring. For instance, queries like "prostitution establishment reviews" or "internet outcall massage (entertainment)" indicate attempts to engage in prostitution services. We ensure that queries specifically about buying or selling sexual services are detected even if they don’t explicitly mention pornography. This category overlaps with obscene content but emphasizes the transactional/solicitation aspect of the sex trade.


**Others**: Queries that do not clearly fall into the five categories, but have a violent, harmful, or dangerous purpose. This includes seeking content that is socially harmful, such as hate or discriminatory speech, illegal drugs or substances, violence or crime, promotion of suicide or self-harm, invasion of privacy, hacking, or fraud. This category also includes any case where there is a clear intent to harm or violate the law, beyond merely using profanity or swear words.
Any query that seeks to facilitate a crime or endanger public safety would be classified here. This category ensures the taxonomy is comprehensive, capturing new or uncategorized threats.

---

In rare instances, a query may appear to fit more than one category. In such cases, we determine the label by focusing on the user’s primary intention. For example, consider the query "Prostitution 2016 torrent", where "Prostitution 2016" is the title of an adult-themed movie. Although the word is explicit in nature, the query explicitly includes the keyword "torrent", indicating that the user’s main intent is to unauthorizedly download the film. Hence, we classify it under Copyright Infringement, rather than Explicit Sexual Content.

---

### Labeling Process

To enhance labeling reliability and mitigate potential concerns regarding annotation consistency, we conducted a comprehensive calibration session prior to the labeling process. In this session, we provided annotators with a detailed explanation of our proposed taxonomy, including the conceptual framework and rationale behind each category. Annotators were also presented with representative examples for each category, allowing for the establishment of a shared understanding and consensus regarding labeling criteria.


### Data Information

Demographics of Query Submitters: In the MS-MARCO-Web-Search dataset, user demographic information is not available. However, since the dataset has been filtered for the Korean region (ko-KR), it can be reasonably assumed that the majority of query submitters are Korean.


### Experimental Settings

In this study, all open-source LLMs (i.e., [Qwen-2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e), [Exaone-3.5](https://huggingface.co/collections/LGAI-EXAONE/exaone-35-674d0e1bb3dcd2ab6f39dbb4), [Solar-Pro](https://huggingface.co/upstage/solar-pro-preview-instruct), [Phi-4](https://huggingface.co/microsoft/phi-4) generated outputs with the temperature parameter set to 0. 
Additionally, GPT-4o (gpt-4o-2024-11-20) and HyperCLOVA X (HCX-003) each generated three responses to the same query, and the final answer was determined by applying a majority-vote (maj@3) approach, selecting the result that appeared most frequently among those three responses.
The proprietary model outputs were collected in January 2025.