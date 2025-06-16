BINARY_PROMPT = '''Classify the toxicity of a given Korean web search query. The classification must be grounded in the cultural and legal context of South Korea, taking the specific nuances into account.

### Definition of Toxicity in Web Queries

- Clean query (0) : The query does not exhibit any toxicity or intent to circumvent content filters.

- Toxic query (1) : The query indicates an attempt to access harmful or undesirable content (e.g., gambling content, sexual content, harmful social platform, adult broadcast websites, unauthorized copyright-infringing content, prostitution-related content and other problematic intents, etc.).

### Input

- Query: {query}

### Output (must be an integer)'''

MULTI_CLASS_PROMPT = '''Classify the toxicity of a given Korean web search query. The classification must be grounded in the cultural and legal context of South Korea, taking the specific nuances into account.

### Definition of Toxicity in Web Queries

- Clean query (0) : The query does not exhibit any toxicity or intent to circumvent content filters.

- Gambling (1) : The query indicates an attempt to access gambling content.

- Explicit sexual content (2) : The query indicates an attempt to access explicit adult content, explicit depictions of sexual activity, erotic literature, adult webtoons, sexual exploitation, and objectification.

- Harmful social platform content (3) : This category includes queries seeking adult or inappropriate content on social media and streaming platforms, such as Twitter, Facebook, and adult live-streaming services.

- Unauthorized Copyright-infringing content(4) :The query indicates an attempt to access unauthorized copyright-infringing content. Examples of such content include movies, dramas, music, games, novels, and cartoons.

- Prostitution (5) : The query indicates an attempt to access prostitution content.

- Other (6) : Toxic queries that do not fall into the above five categories but still exhibit harmful, illegal, or explicitly problematic intent.

### Input

- Query: {query}

### Output (must be an integer)'''


