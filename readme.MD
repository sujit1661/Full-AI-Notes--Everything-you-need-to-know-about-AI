# AI Engineer Interview Preparation Notes
> Sujit Sadalage | B.Tech AI & DS | CGPA 9.43 | Python & AI Developer

---

## Table of Contents
1. [Artificial Intelligence (AI)](#1-artificial-intelligence-ai)
2. [Machine Learning (ML)](#2-machine-learning-ml)
3. [Deep Learning (DL)](#3-deep-learning-dl)
4. [Generative AI & Agentic AI](#4-generative-ai--agentic-ai)
5. [Prompt Engineering - Complete Guide](#5-prompt-engineering--complete-guide)
6. [NLP - Natural Language Processing](#6-nlp--natural-language-processing)
7. [LLMs - Large Language Models](#7-llms--large-language-models)
8. [LLM Parameters](#8-llm-parameters)
9. [RAG - Retrieval-Augmented Generation](#9-rag--retrieval-augmented-generation)
10. [LangChain - Deep Dive](#10-langchain--deep-dive)
11. [LangGraph - Deep Dive](#11-langgraph--deep-dive)
12. [LlamaIndex - Deep Dive](#12-llamaindex--deep-dive)
13. [Vector Databases - Deep Dive](#13-vector-databases--deep-dive)
14. [MCP - Model Context Protocol](#14-mcp--model-context-protocol)
15. [Agentic AI - Deep Dive](#15-agentic-ai--deep-dive)
16. [AI Security & Safety](#16-ai-security--safety)
17. [Scalability in AI Systems](#17-scalability-in-ai-systems)
18. [FastAPI for AI - Deep Dive](#18-fastapi-for-ai--deep-dive)
19. [Python for AI Engineers](#19-python-for-ai-engineers)
20. [Interview Prep - Based on Your Resume](#20-interview-prep--based-on-your-resume)

> Tip: Use Ctrl+F to jump to any section quickly during last-minute revision.

---


---

## 1. Artificial Intelligence (AI)

Artificial Intelligence is the ability of machines to simulate human intelligence such as learning, reasoning, problem-solving, and decision-making.
AI makes machines capable of thinking and acting like humans.

### Types of AI

**1. Narrow AI (Weak AI)**
- Designed to perform one specific task only.
- Works within a limited area and cannot perform beyond its trained function.
- Examples: Voice assistants like Siri, Recommendation systems, Face recognition, Chatbots, Spam filters

**2. General AI (Strong AI)**
- An AI system that can perform any intellectual task like a human.
- Can think, reason, learn, and solve problems across different domains.
- Currently, true General AI does not exist. It is still a research goal.

**3. Super AI**
- A hypothetical future AI that would be smarter than humans in every field.
- Would exceed human intelligence, creativity, and decision-making.
- Currently does not exist. It is only a theoretical concept.

### Difference Between AI, ML, and Deep Learning

- AI is the main field that focuses on making machines intelligent.
- ML is one way to achieve AI by allowing systems to learn from data instead of writing fixed rules.
- Deep Learning is a more advanced form of ML that uses deep neural networks.

```
AI
+-- Machine Learning
    +-- Deep Learning
        +-- LLMs / Transformers
```

---


> ---
> **Section Summary:** AI is the broad field of making machines intelligent. It has 3 types: Narrow AI (exists today), General AI (research goal), Super AI (theoretical). Everything in ML and DL is a subset of AI. The key hierarchy is AI > ML > DL > LLMs.


---

## 2. Machine Learning (ML)

Machine Learning is a subset of AI where machines learn from data and make predictions to improve their performance without being explicitly programmed.
Instead of writing rules manually, we give data to the machine and it learns patterns on its own.

### Types of Machine Learning

**1. Supervised Learning**
- Trained on labeled data (input + correct output both given).
- Examples: Spam Email Detection, House Price Prediction, Student Result Prediction

**2. Unsupervised Learning**
- Works on unlabeled data. Machine finds hidden patterns by itself.
- Examples: Customer Segmentation, Fraud Detection, Market Basket Analysis

**3. Reinforcement Learning**
- Agent learns by interacting with environment. Good action = reward, Bad action = penalty.
- Examples: Self-Driving Cars, Game Playing (Chess, PUBG Bots), Robotics

> Some people also include Semi-Supervised Learning, but these 3 are the main ones for interviews.

### Classification vs Regression

Both are types of Supervised Learning.

- **Classification** - Predicts categories (Yes/No, Spam/Not Spam, Pass/Fail)
- **Regression** - Predicts continuous numbers (price, salary, temperature, marks)

### Key ML Concepts

**Overfitting vs Underfitting**
- Overfitting: Model learns training data too well, performs badly on new data (too complex)
- Underfitting: Model is too simple, can't even learn training data properly
- Fix: Use more data, regularization (L1/L2), dropout, cross-validation

**Bias vs Variance**
- Bias: Error from wrong assumptions (underfitting)
- Variance: Error from sensitivity to small changes in training data (overfitting)
- Goal: Find the sweet spot - low bias + low variance

**Train / Validation / Test Split**
- Train set: Model learns from this (usually 70-80%)
- Validation set: Tune hyperparameters (10-15%)
- Test set: Final evaluation, never touched during training (10-15%)

**Cross-Validation**
- K-Fold: Split data into K parts, train on K-1, test on 1, repeat K times
- Gives a more reliable performance estimate

**Feature Engineering**
- Process of selecting, transforming, or creating features from raw data to improve model performance
- Includes: normalization, encoding categorical variables, handling missing values

**Evaluation Metrics**
```
Classification:
  - Accuracy    = correct predictions / total predictions
  - Precision   = TP / (TP + FP)  -- how many predicted positives are actually positive
  - Recall      = TP / (TP + FN)  -- how many actual positives were caught
  - F1 Score    = 2 * (Precision * Recall) / (Precision + Recall)
  - ROC-AUC     = area under the ROC curve

Regression:
  - MAE  = Mean Absolute Error
  - MSE  = Mean Squared Error
  - RMSE = Root Mean Squared Error
  - R2   = how well model explains variance in data
```

**Common ML Algorithms (know these names)**
```
Supervised:
  - Linear Regression            -- predict numbers
  - Logistic Regression          -- binary classification
  - Decision Tree                -- tree-based decisions
  - Random Forest                -- ensemble of decision trees
  - SVM (Support Vector Machine) -- finds best boundary between classes
  - KNN (K-Nearest Neighbors)    -- classify based on nearest neighbors
  - Naive Bayes                  -- probability-based classifier
  - Gradient Boosting (XGBoost)  -- powerful ensemble method

Unsupervised:
  - K-Means Clustering           -- group data into K clusters
  - DBSCAN                       -- density-based clustering
  - PCA (Principal Component Analysis) -- dimensionality reduction
```

---


> ---
> **Section Summary:** ML lets machines learn from data without explicit rules. 3 main types: Supervised (labeled data), Unsupervised (find patterns), Reinforcement (reward/penalty). Key concepts to know for interviews: overfitting vs underfitting, bias vs variance, train/val/test split, evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC for classification; MAE, MSE, RMSE, R2 for regression), and common algorithms (Linear/Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, KNN).


---

## 3. Deep Learning (DL)

Deep Learning is a subset of Machine Learning that uses artificial neural networks with multiple hidden layers to learn complex patterns from large amounts of data.
It is called "deep" because the neural network has many layers between input and output.

### Neural Network Basics

```
Input Layer --> Hidden Layer(s) --> Output Layer

Each layer has:
  - Neurons (nodes)
  - Weights (how important each connection is)
  - Bias (shifts the activation)
  - Activation Function (adds non-linearity)
```

**Activation Functions**
```
ReLU    -- max(0, x)  -- most common in hidden layers, avoids vanishing gradient
Sigmoid -- 1/(1+e^-x) -- outputs 0-1, used in binary classification output
Tanh    -- outputs -1 to 1, better than sigmoid for hidden layers
Softmax -- converts outputs to probabilities that sum to 1, used in multi-class output
```

**How Training Works**
1. Forward Pass: Input goes through layers, prediction is made
2. Loss Calculation: Compare prediction vs actual (using loss function)
3. Backpropagation: Calculate gradients of loss w.r.t. weights
4. Gradient Descent: Update weights to reduce loss
5. Repeat for many epochs

**Loss Functions**
```
MSE (Mean Squared Error)         -- regression problems
Binary Cross-Entropy             -- binary classification
Categorical Cross-Entropy        -- multi-class classification
```

**Optimizers**
```
SGD (Stochastic Gradient Descent) -- basic, can be slow
Adam                              -- most popular, adaptive learning rate
RMSprop                           -- good for RNNs
```

**Regularization Techniques**
```
Dropout     -- randomly turns off neurons during training to prevent overfitting
L1/L2       -- adds penalty to large weights
Batch Norm  -- normalizes layer inputs, speeds up training
Early Stopping -- stop training when validation loss stops improving
```

### Types of Neural Networks

**CNN (Convolutional Neural Network)**
- Designed for image data
- Uses convolutional layers to detect features (edges, shapes, objects)
- Used in: image classification, object detection, face recognition

**RNN (Recurrent Neural Network)**
- Designed for sequential data (text, time series)
- Has memory - output from previous step feeds into next step
- Problem: Vanishing gradient for long sequences

**LSTM (Long Short-Term Memory)**
- Improved RNN that solves vanishing gradient problem
- Has gates (forget, input, output) to control what to remember
- Used in: text generation, speech recognition, time series

**Transformer**
- Modern architecture that replaced RNNs for NLP
- Uses self-attention instead of recurrence
- Processes all tokens in parallel (much faster)
- Foundation of all modern LLMs (GPT, BERT, Claude, Gemini)

**GAN (Generative Adversarial Network)**
- Two networks: Generator (creates fake data) vs Discriminator (detects fake)
- They compete - generator gets better at fooling, discriminator gets better at detecting
- Used in: image generation, deepfakes, data augmentation

**Autoencoder**
- Encoder compresses input into a smaller representation (latent space)
- Decoder reconstructs the original from that compressed form
- Used in: anomaly detection, dimensionality reduction, denoising

---


> ---
> **Section Summary:** DL uses multi-layer neural networks to learn complex patterns. Training flow: Forward Pass > Loss > Backprop > Optimizer > Repeat. Key activation functions: ReLU (hidden layers), Sigmoid (binary output), Softmax (multi-class). Regularization: Dropout, L1/L2, Batch Norm, Early Stopping. Network types: CNN (images), RNN/LSTM (sequences), Transformer (NLP, foundation of all LLMs), GAN (generation), Autoencoder (compression/anomaly).


---

## 4. Generative AI & Agentic AI

### Generative AI (Gen AI)
Generative AI is a type of Artificial Intelligence that creates new content such as text, images, code, audio, or videos based on the data it has learned from.
It generates something new instead of just analyzing existing data.
Examples: ChatGPT writes emails, DALL-E creates images, GitHub Copilot generates code.

### Agentic AI
Agentic AI is an advanced AI system that can think, plan, make decisions, and take actions automatically to complete a goal with minimal human involvement.
Instead of only answering questions, it can decide what steps to take and perform tasks by itself.
It behaves more like an "AI agent."

**Regular AI vs Agentic AI**
```
Regular AI:  User asks --> AI answers --> Done
Agentic AI:  User gives goal --> AI plans --> AI takes steps --> AI uses tools --> AI completes goal
```

### Prompt Engineering

Prompt Engineering is the practice of designing and optimizing input prompts to get the best possible output from an LLM.

**Zero-Shot Prompting** - Ask directly without examples
```
"Translate this to French: Hello, how are you?"
```

**Few-Shot Prompting** - Give examples before asking
```
"English: Hello --> French: Bonjour
 English: Thank you --> French: Merci
 English: Good morning --> French: ?"
```

**Chain-of-Thought (CoT) Prompting** - Ask model to think step by step
```
"Solve this step by step: If a train travels 60km/h for 2 hours, how far does it go?"
```

**System Prompt** - Set the role/behavior of the AI
```python
{"role": "system", "content": "You are a helpful Python tutor. Always explain with code examples."}
```

**ReAct Prompting** - Reason + Act pattern used in agents
```
Thought: I need to find the current weather
Action: search("current weather Mumbai")
Observation: 32 degrees C, sunny
Thought: I have the answer
Answer: It is 32 degrees C and sunny in Mumbai
```

### Tool Calling / Function Calling

Tool calling allows LLMs to call external functions/APIs when they need real-world data.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the weather in Mumbai?"}],
    tools=tools,
    tool_choice="auto"
)
```

### Memory in AI Systems

```
Short-term memory  -- conversation history (last N messages)
Long-term memory   -- stored in vector DB, retrieved when relevant
Entity memory      -- remembers specific facts about people/things
Summary memory     -- summarizes old conversation to save tokens
```

---


> ---
> **Section Summary:** GenAI creates new content (text, image, code) from learned patterns. Agentic AI goes further -- it plans, decides, and acts autonomously using a Goal > Plan > Act > Observe > Loop cycle. Tool calling is how LLMs interact with the real world (search, DB, APIs). Memory types: buffer (last N messages), summary (compressed history), vector (semantic retrieval), entity (tracks facts). These are the building blocks of every AI agent you will build.


---

## 5. Prompt Engineering - Complete Guide

Prompt Engineering is the practice of designing inputs to get the best output from an LLM.
Bad prompts = bad outputs regardless of model quality. Core skill for every AI engineer.

### Prompt Structure
```
[System Prompt]  -- sets role, tone, constraints, output format
[Context]        -- background info, retrieved docs (RAG), examples
[Instruction]    -- what you want the model to do
[Input]          -- the actual user query or data
[Output Format]  -- how you want the response structured
```

### Prompting Techniques

**Zero-Shot** - Ask directly, no examples
```python
prompt = "Classify this review as Positive or Negative: 'The product broke after one day.'"
# Output: Negative
```

**Few-Shot** - Give 2-5 examples before the actual question
```python
prompt = """
Classify sentiment:
Review: "Amazing product!" -> Positive
Review: "Terrible quality." -> Negative
Review: "Nothing special." -> Neutral
Review: "Best purchase ever!" -> ?
"""
# Output: Positive
# Why: model learns the pattern from examples, much more accurate than zero-shot
```

**Chain-of-Thought (CoT)** - Ask model to reason step by step. Dramatically improves math/logic.
```python
# Without CoT (often wrong)
prompt = "Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many does he have?"

# With CoT (correct) -- just add "Think step by step"
prompt = "Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many? Think step by step."
# Output: Roger starts with 5. He buys 2 x 3 = 6 more. 5 + 6 = 11 tennis balls.
```

**Zero-Shot CoT** - Append "Think step by step." to any prompt
```python
prompt = f"{question}\nThink step by step."
```

**Self-Consistency** - Generate multiple CoT answers, take majority vote. More reliable.
```python
from collections import Counter
answers = [call_llm(cot_prompt) for _ in range(5)]
final_answer = Counter(answers).most_common(1)[0][0]
```

**ReAct (Reason + Act)** - Model alternates Thought, Action, Observation. Used in agents.
```
Thought: I need to find the population of Mumbai.
Action: search("Mumbai population 2024")
Observation: Mumbai population is approximately 20.7 million.
Thought: I now have the answer.
Answer: The population of Mumbai is approximately 20.7 million.
```

**Tree of Thoughts (ToT)** - Explore multiple reasoning paths, backtrack if needed.
```
Problem
  +-- Path A: step1 --> step2 --> dead end, backtrack
  +-- Path B: step1 --> step2 --> step3 --> Solution
```

**Role Prompting** - Assign a specific persona
```python
system = "You are a senior Python engineer with 10 years of experience. Write clean, production-ready code with proper error handling and type hints."
```

**Constrained Generation** - Force specific output format
```python
prompt = """
Extract the following from the text and return as JSON only. No explanation.
Fields: name, email, phone

Text: "Hi I am Sujit, reach me at sujit@email.com or 9876543210"

Return format:
{
  "name": "",
  "email": "",
  "phone": ""
}
"""
```

**RAG Prompt Pattern** - Standard pattern for grounding LLM in retrieved documents
```python
def build_rag_prompt(context: str, question: str) -> str:
    return f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I do not have enough information."
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""
```

**Prompt Chaining** - Output of one prompt feeds into next
```python
key_points = call_llm(f"Extract 5 key points from: {document}")
questions  = call_llm(f"Generate 3 interview questions based on: {key_points}")
answers    = call_llm(f"Answer these questions concisely: {questions}")
```

### System Prompt Best Practices
```python
system_prompt = """
You are an AI assistant specialized in Python and AI engineering.

Rules:
- Answer only questions related to Python, AI, ML, and software engineering
- Always include code examples when explaining technical concepts
- If you do not know something, say "I am not sure" instead of guessing
- Keep answers concise and structured

Output format:
- Start with a one-line summary
- Then detailed explanation
- Then code example if applicable
"""
```

### Prompt Anti-Patterns
```
BAD:  "Tell me about AI"
GOOD: "Explain the difference between supervised and unsupervised learning with one example each"

BAD:  "Write code"
GOOD: "Write a Python function that takes a list of strings and returns only strings longer
       than 5 characters. Include type hints and a docstring."

BAD:  "Summarize this"
GOOD: "Summarize this in 3 sentences for a non-technical audience"

BAD:  "You are an assistant"
GOOD: "You are a senior FastAPI developer reviewing code for production readiness"
```

### Prompt Injection Defense
```python
import re

INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"forget (your|all) instructions",
    r"you are now",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"DAN mode",
    r"system prompt",
    r"reveal your instructions"
]

def sanitize_input(user_input: str) -> str:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            raise ValueError("Potentially malicious input detected")
    if len(user_input) > 2000:
        raise ValueError("Input too long")
    return user_input.strip()
```

### Token Counting
```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def trim_to_token_limit(text: str, max_tokens: int = 3000, model: str = "gpt-4") -> str:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
```

### LangChain Prompt Templates
```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder
)

# Basic template
template = PromptTemplate(
    input_variables=["topic", "level"],
    template="Explain {topic} for a {level} developer. Include a code example."
)
prompt = template.format(topic="RAG", level="beginner")

# Chat template (system + user)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Always respond in {language}."),
    ("human", "{question}")
])
messages = chat_template.format_messages(
    role="Python tutor", language="English", question="What is a decorator?"
)

# With conversation history (for chatbots)
chat_with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),  # injects past messages here
    ("human", "{input}")
])

# Few-shot template
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall",  "output": "short"},
]
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)
few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of each word.",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)
```

### Output Parsers
```python
from langchain_core.output_parsers import (
    StrOutputParser,                    # plain text string
    JsonOutputParser,                   # parse JSON from LLM output
    CommaSeparatedListOutputParser      # "a, b, c" --> ["a", "b", "c"]
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# String parser -- most common, just returns text
str_parser = StrOutputParser()
chain = prompt | llm | str_parser

# Comma separated list
list_parser = CommaSeparatedListOutputParser()
# LLM output "red, green, blue" --> ["red", "green", "blue"]

# Pydantic parser -- structured output with type validation
class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating from 0 to 10")
    summary: str = Field(description="One sentence summary")

pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = PromptTemplate(
    template="Review this movie.\n{format_instructions}\nMovie: {movie}",
    input_variables=["movie"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)
chain = prompt | llm | pydantic_parser
result = chain.invoke({"movie": "Inception"})
print(result.title, result.rating)  # Inception  9.0

# Why Pydantic parser?
# LLM output is text. Parser extracts structured data and validates types.
# If LLM returns wrong format, parser raises a clear error.
```

---


> ---
> **Section Summary:** Prompt engineering is how you get the best output from an LLM. Key techniques: Zero-Shot (direct ask), Few-Shot (give examples), CoT (think step by step), ReAct (reason + act loop), Role Prompting (assign persona). Always be specific about format, length, and constraints. Use LangChain PromptTemplate for dynamic prompts and Output Parsers (Str, JSON, Pydantic) to structure LLM responses. Defend against prompt injection by sanitizing inputs. Count tokens with tiktoken to avoid exceeding context limits.


---

## 6. NLP - Natural Language Processing

**NLP** - Branch of AI that helps computers understand, process, and work with human language.

**NLU (Natural Language Understanding)** - Focuses on understanding the meaning and intent behind human language. Not just words, but the real intention.

**NLG (Natural Language Generation)** - Focuses on generating human-like text or speech from data.

### Text Preprocessing Pipeline
```
Raw Text
  --> Lowercasing
  --> Remove punctuation / special characters
  --> Tokenization (split into words/tokens)
  --> Stop word removal (remove "the", "is", "a")
  --> Stemming or Lemmatization
  --> Vectorization (TF-IDF, Word2Vec, or Embeddings)
```

### Stemming vs Lemmatization

**Stemming** - Removes prefixes/suffixes using simple rules. May not produce a real word.
```
playing --> play
studies --> studi
running --> run
caring  --> care
```

**Lemmatization** - Converts to proper dictionary base form using language rules.
```
playing --> play
studies --> study
better  --> good
running --> run
```

### Text Vectorization Techniques

**TF-IDF (Term Frequency-Inverse Document Frequency)**
- Measures how important a word is in a document relative to a collection
- Common words get lower scores, rare but relevant words get higher scores

**Word2Vec**
- Converts words to dense vectors where similar words are close in vector space
- "king" - "man" + "woman" = "queen"

**Sentence Transformers / BERT Embeddings**
- Converts entire sentences to vectors capturing full context
- Much better than Word2Vec for semantic search and RAG

### Attention Mechanism

Attention lets the model focus on the most relevant parts of the input when generating each output token.

```
Input: "The cat sat on the mat because it was soft"
When predicting what "it" refers to:
  --> Model pays HIGH attention to "mat"
  --> Model pays LOW attention to "cat", "sat"
```

Self-attention = attention within the same sequence (used in transformers)

---


> ---
> **Section Summary:** NLP is the AI branch for understanding and generating human language. NLU = understanding intent, NLG = generating text. Standard preprocessing pipeline: lowercase > remove punctuation > tokenize > remove stopwords > lemmatize. Stemming is fast but may produce non-words (studies->studi). Lemmatization is slower but always produces real words (studies->study). For modern RAG, use Sentence Transformers (all-MiniLM-L6-v2) instead of TF-IDF or Word2Vec -- they capture full sentence meaning in a single vector.


---

## 7. LLMs - Large Language Models

### Model Types

- **LLM (Large Language Model)** - AI model trained on massive text data to understand and generate human-like language. Examples: GPT-4, Claude, Gemini, LLaMA
- **SLM (Small Language Model)** - Smaller, faster, cheaper version. Examples: Phi-3, Gemma, Mistral 7B
- **VLM (Vision Language Model)** - Understands both images and text. Examples: GPT-4V, LLaVA, Gemini Vision
- **Multimodal Model** - Handles text, image, audio, video together. Example: GPT-4o

### How LLMs Work - Full Flow

```
User Input (text)
      |
      v
Tokenization --> split text into tokens
      |
      v
Embedding Layer --> tokens become vectors
      |
      v
Transformer Layers (Self-Attention + Feed-Forward)
      |
      v
[Optional] Fine-Tuning / RLHF
      |
      v
Decoding (greedy / top-k / top-p / temperature)
      |
      v
Detokenization --> vectors back to text
      |
      v
[Optional] Hallucination Check / RAG Grounding
      |
      v
Final Output to User
```

### Fine-Tuning

Fine-tuning = take a pre-trained model + train it further on smaller domain-specific data.

```
Pre-trained model  --> general knowledge (trained on internet-scale data)
Fine-tuned model   --> specialized (e.g., medical, legal, coding assistant)
```

**Techniques:**
- Supervised Fine-Tuning (SFT) - train on labeled instruction-response pairs
- RLHF (Reinforcement Learning from Human Feedback) - humans rate outputs, model improves
- LoRA (Low-Rank Adaptation) - fine-tune only a small subset of weights, very efficient
- QLoRA - LoRA + quantization, runs on consumer GPUs

### Context Window

The context window is the maximum amount of text (tokens) an LLM can process at once.

```
GPT-3.5      --> 4K tokens
GPT-4        --> 8K / 32K tokens
GPT-4 Turbo  --> 128K tokens
Claude 3     --> 200K tokens
Gemini 1.5   --> 1M tokens
```

> 1 token is approximately 0.75 words. "Hello world" = 2 tokens.

### Hallucination Solutions

1. RAG - Fetch real data before generating
2. Fine-Tuning - Train on domain-specific accurate data
3. Prompt Engineering - "Answer only from context. Say I don't know if unsure."
4. Lower Temperature - More deterministic outputs
5. Chain-of-Thought - Guide reasoning step by step
6. Tool Calling - Use external APIs for real data
7. Post-Generation Verification - Second model checks the output
8. Human-in-the-Loop - Manual review for critical outputs
9. Better Training Data - Clean, diverse, up-to-date datasets
10. Confidence Thresholding - Model says "I don't know" when confidence is low

### Popular LLM Providers

```
OpenAI      --> GPT-4, GPT-4o, GPT-4.1
Anthropic   --> Claude 3.5 Sonnet, Claude 3 Opus
Google      --> Gemini 1.5 Pro, Gemini Flash
Meta        --> LLaMA 3.1, LLaMA 3.2 (open source)
Mistral     --> Mistral 7B, Mixtral (open source)
Groq        --> Fast inference API (runs LLaMA, Mixtral on LPU hardware)
HuggingFace --> Hub for open-source models
```

---


> ---
> **Section Summary:** LLMs are trained on massive text data to understand and generate language. Types: LLM (text), SLM (smaller/faster), VLM (vision+text), Multimodal (text+image+audio). Full flow: Tokenize > Embed > Transformer layers > Decode > Output. Fine-tuning adapts a pre-trained model to a specific domain (SFT, RLHF, LoRA, QLoRA). Context window = max tokens the model can process at once (GPT-4: 128K, Claude 3: 200K). Hallucination is the biggest risk -- fix with RAG, lower temperature, CoT, and tool calling.


---

## 8. LLM Parameters

### model
```python
model="gpt-4.1"          # OpenAI GPT-4.1
model="qwen/qwen3-32b"   # Qwen 32B
model="gpt-5-mini"       # faster, cheaper
```

### messages
```python
messages=[
    {"role": "system",    "content": "You are a helpful tutor"},
    {"role": "user",      "content": "Explain RAG"},
    {"role": "assistant", "content": "RAG stands for..."},
    {"role": "user",      "content": "Give me an example"}
]
```

### temperature (0.0 to 1.0+)
Controls randomness. Negative values cause error or default to 0. Values > 2 risk hallucination.
```python
temperature=0.1   # factual (math, code)
temperature=0.7   # balanced
temperature=1.0   # creative writing
```
> Low = accurate and repeatable | High = creative and diverse

### top_p (0.0 to 1.0)
Nucleus sampling. Model picks from smallest set of tokens whose cumulative probability = top_p.
```python
top_p=0.5   # strict word choices
top_p=0.9   # balanced
top_p=1.0   # all words allowed
```

### top_k
Limits model to choose from only top K most probable tokens.

### max_completion_tokens
```python
max_completion_tokens=256    # short answer
max_completion_tokens=1024   # detailed
max_completion_tokens=4096   # long doc
```

### stream
```python
stream=True    # live typing effect (token by token)
stream=False   # full response at once
```

### stop
```python
stop=["END"]
stop=["###"]
stop=["\n\n"]
```

### presence_penalty and frequency_penalty
```python
presence_penalty=0.5    # encourages new topics
frequency_penalty=0.3   # reduces word repetition
```

### seed
```python
seed=42   # same seed = same output (reproducible)
```

### response_format
```python
response_format={"type": "json_object"}   # force JSON output
response_format={"type": "text"}
```

### tool_choice
```python
tool_choice="auto"      # model decides when to use tools
tool_choice="none"      # tools disabled
tool_choice="required"  # must use a tool
```

### Other Parameters
```python
n=3                       # generate 3 different responses
reasoning_effort="high"   # deep step-by-step thinking
logprobs=True             # return token probabilities
timeout=30                # API timeout in seconds
parallel_tool_calls=True  # run multiple tools simultaneously
user="sujit_01"           # identify end user for logging
```

---


> ---
> **Section Summary:** Key parameters to know: temperature (0=deterministic, 1=creative), top_p (nucleus sampling, limits token pool), top_k (limits to top K tokens), max_tokens (output length limit), stream (token-by-token response), stop (stop generation on string), presence_penalty (encourages new topics), frequency_penalty (reduces repetition), seed (reproducible output), response_format (force JSON), tool_choice (auto/none/required). Rule: tune temperature OR top_p, not both at extremes. Use seed for reproducible testing.


---

## 9. RAG - Retrieval-Augmented Generation

RAG is an AI framework that combines information retrieval with text generation. The model first retrieves relevant external knowledge from a database or document store, then uses that retrieved information to generate accurate, context-aware responses.

**Why RAG?**
- LLMs have a knowledge cutoff date
- LLMs can hallucinate facts
- RAG grounds responses in real, up-to-date documents

### Phase 1: Indexing Phase (Data Ingestion)

```
Raw Documents (PDF, DOCX, Excel, Images with OCR)
      |
      v
Step 1: Document Loading
      |  (LangChain loaders: PyPDFLoader, UnstructuredLoader, etc.)
      v
Step 2: Chunking (text_splitters)
      |  chunk_size=500, chunk_overlap=50
      v
Step 3: Embedding Generation
      |  (HuggingFace all-MiniLM-L6-v2 --> 384-dim vectors)
      |  (OpenAI text-embedding-3-small --> 1536-dim vectors)
      v
Step 4: Store in Vector DB
         (ChromaDB / Qdrant / Pinecone / FAISS)
```

**Chunking explained:**
- chunk_size: max tokens per chunk (fits LLM context limit)
- chunk_overlap: shared tokens between consecutive chunks (preserves context at boundaries)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(docs)
```

**Embedding + Store in ChromaDB:**
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
vectorstore.persist()
```

### Phase 2: Retrieving Phase (Query Time)

```
User Query
      |
      v
Step 1: Embed the query (same embedding model)
      |
      v
Step 2: Cosine Similarity Search in Vector DB
      |  --> returns top-k most relevant chunks
      v
Step 3: Rank and assemble context
      |
      v
Step 4: Pass [context + query] to LLM
      |
      v
Final Answer
```

**Retrieval + Generation:**
```python
# Load existing vector store
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Retrieve relevant chunks
docs = retriever.get_relevant_documents("What is the project deadline?")

# Build context
context = "\n\n".join([doc.page_content for doc in docs])

# Send to LLM
prompt = f"Answer based on context only:\n\nContext:\n{context}\n\nQuestion: {query}"
```

### Advanced RAG Techniques

**Hybrid Search** - Combine vector search + keyword search (BM25) for better results

**Re-ranking** - After retrieval, use a cross-encoder model to re-rank chunks by relevance

**HyDE (Hypothetical Document Embeddings)** - Generate a hypothetical answer first, embed it, then search

**Parent-Child Chunking** - Store small chunks for retrieval but pass larger parent chunks to LLM for context

**Self-Query Retriever** - LLM generates structured filters from natural language query

### Vector DB Comparison

| Tool | Type | Best For |
|------|------|----------|
| ChromaDB | Open-source local | Local/small-scale RAG, plug-and-play |
| Pinecone | Managed cloud | Production, high-speed search |
| Qdrant | Open-source | Strong filtering, flexible deployment |
| FAISS (Meta) | Library | High-performance local retrieval |
| Weaviate | Open-source | Hybrid queries, classification |
| Milvus | Open-source | Large-scale embedding storage |
| Astra DB | Managed cloud | Cassandra-based, scalable |
| ElasticSearch | Search engine | Hybrid vector + keyword search |

---



> ---
> **Section Summary:** RAG = retrieval + generation. Solves LLM knowledge cutoff and hallucination by grounding answers in real documents. Two phases: Indexing (Load > Chunk > Embed > Store in VectorDB) and Retrieval (Embed query > Similarity search > Top-K chunks > LLM). Key decisions: chunk_size (500 chars typical), chunk_overlap (50 chars, prevents boundary loss), embedding model (BAAI/bge for free, OpenAI for quality), vector DB (ChromaDB for local, Qdrant for production). Advanced: Hybrid Search (vector+BM25), Re-ranking, MMR for diverse results.


---

## 10. LangChain - Deep Dive

LangChain is a framework for building LLM-powered applications. It provides composable abstractions for every layer of an AI app: models, prompts, chains, memory, agents, tools, callbacks, and output parsers.

### Pros and Cons
```
PROS:
  + Huge ecosystem -- 100+ LLM, vector DB, tool integrations out of the box
  + LCEL pipe syntax -- clean, composable, streaming-ready chains
  + Built-in memory, agents, callbacks, output parsers
  + Works with any LLM (OpenAI, Groq, Anthropic, Ollama, HuggingFace)
  + Active community, frequent updates
  + Async support throughout (ainvoke, astream, abatch)

CONS:
  - Frequent breaking changes between versions (v0.1 -> v0.2 -> v0.3)
  - Abstraction hides what is actually happening (hard to debug)
  - Overkill for simple LLM calls -- just use the SDK directly
  - Documentation can lag behind code changes
  - Can be slow due to abstraction layers
```

### Package Structure
```
langchain-core      -- base abstractions (Runnable, BaseMessage, Document, PromptTemplate)
langchain           -- chains, agents, memory (high-level)
langchain-openai    -- OpenAI LLMs + embeddings
langchain-groq      -- Groq fast inference
langchain-anthropic -- Claude models
langchain-community -- 100+ community integrations (loaders, vectorstores, tools)
```

---

### 1. Models

```python
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

# ChatOpenAI -- chat model, supports system/user/assistant roles
# Use this for GPT-3.5, GPT-4, GPT-4o
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,       # 0=deterministic, 1=creative
    max_tokens=1000,       # max output tokens
    timeout=30,            # API timeout seconds
    max_retries=2,         # auto retry on failure
    api_key="your-key"     # or set OPENAI_API_KEY env var
)

# ChatGroq -- same API as OpenAI but much faster (LPU hardware)
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # or mixtral-8x7b-32768
    temperature=0,
    groq_api_key="your-groq-key"
)

# Ollama -- run models locally, no API key needed
llm = Ollama(model="llama3")  # must have Ollama running locally

# --- Invocation methods ---

# .invoke() -- single call, returns AIMessage
response = llm.invoke("What is RAG?")
print(response.content)

# .batch() -- multiple prompts in parallel, returns list of AIMessage
responses = llm.batch(["What is RAG?", "What is LangChain?", "What is LangGraph?"])

# .stream() -- returns tokens one by one as they generate
for chunk in llm.stream("Explain transformers step by step"):
    print(chunk.content, end='', flush=True)

# .ainvoke() -- async single call
import asyncio
response = asyncio.run(llm.ainvoke("What is RAG?"))

# .abatch() -- async batch
responses = asyncio.run(llm.abatch(["Q1", "Q2", "Q3"]))

# .astream() -- async stream
async for chunk in llm.astream("Explain RAG"):
    print(chunk.content, end='', flush=True)

# .bind() -- pre-fill parameters, returns new runnable
llm_json = llm.bind(response_format={'type': 'json_object'})
llm_stop  = llm.bind(stop=['###'])
llm_tools = llm.bind_tools(tools)  # attach tools for function calling

# .with_retry() -- auto retry on specific exceptions
from openai import RateLimitError
llm_retry = llm.with_retry(
    retry_if_exception_type=(RateLimitError,),
    stop_after_attempt=3,
    wait_exponential_jitter=True
)

# .with_fallbacks() -- fallback to another model if primary fails
llm_with_fallback = llm.with_fallbacks([ChatGroq(model='llama-3.1-8b-instant')])
```

### 2. Document Loaders

```python
from langchain_community.document_loaders import (
    PyPDFLoader,                      # PDF files, splits by page
    PyPDFDirectoryLoader,             # all PDFs in a folder
    TextLoader,                       # plain .txt files
    CSVLoader,                        # CSV, each row = one document
    UnstructuredWordDocumentLoader,   # DOCX files
    WebBaseLoader,                    # scrape web pages
    DirectoryLoader,                  # load all files in folder with any loader
    JSONLoader,                       # JSON files with jq path
    UnstructuredExcelLoader,          # Excel files
    UnstructuredImageLoader,          # images with OCR
)

# PyPDFLoader -- most common for PDFs
# Each page becomes one Document object
loader = PyPDFLoader("report.pdf")
docs = loader.load()
# docs[0].page_content = text of page 1
# docs[0].metadata = {'source': 'report.pdf', 'page': 0}

# PyPDFDirectoryLoader -- load all PDFs in a folder
loader = PyPDFDirectoryLoader("./documents/")
docs = loader.load()

# WebBaseLoader -- scrape web pages
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(["https://docs.langchain.com/", "https://python.langchain.com/"])
docs = loader.load()

# DirectoryLoader -- load all files with a specific loader
loader = DirectoryLoader(
    "./docs/",
    glob="**/*.pdf",          # pattern to match files
    loader_cls=PyPDFLoader,   # which loader to use
    show_progress=True,       # progress bar
    use_multithreading=True   # parallel loading
)
docs = loader.load()

# JSONLoader -- load JSON with jq path
loader = JSONLoader(
    file_path="data.json",
    jq_schema=".messages[].content",  # jq path to text field
    text_content=True
)

# All loaders return List[Document]
# Document has: .page_content (str) and .metadata (dict)
print(f'Loaded {len(docs)} documents')
print(docs[0].page_content[:200])
print(docs[0].metadata)
```

**Pros/Cons of Document Loaders:**
```
PyPDFLoader:    + simple, reliable  - loses formatting, tables become messy
WebBaseLoader:  + easy web scraping - JS-rendered pages not supported
DirectoryLoader:+ batch load        - all files must use same loader class
UnstructuredLoader: + handles many formats - slow, heavy dependency
```

### 3. Text Splitters

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,   # BEST general purpose -- tries multiple separators
    CharacterTextSplitter,            # splits on single character
    TokenTextSplitter,                # splits by token count (precise for LLM limits)
    MarkdownHeaderTextSplitter,       # splits markdown by headers (preserves structure)
    PythonCodeTextSplitter,           # splits Python code by functions/classes
    Language,                         # enum for code language splitters
)

# RecursiveCharacterTextSplitter -- MOST COMMONLY USED
# Tries separators in order: paragraph -> sentence -> word -> character
# Guarantees chunks are <= chunk_size
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # max characters per chunk
    chunk_overlap=50,      # overlap between consecutive chunks
    separators=[
        "\n\n",  # paragraph break (try first)
        "\n",    # line break
        ". ",    # sentence end
        " ",     # word boundary
        ""       # character (last resort)
    ],
    length_function=len,   # how to measure chunk size (default: len)
    is_separator_regex=False
)
chunks = splitter.split_documents(docs)   # List[Document] -> List[Document]
chunks = splitter.split_text('raw text')  # str -> List[str]

# CharacterTextSplitter -- splits on ONE separator only
# Simpler but less reliable than Recursive
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=100
)

# TokenTextSplitter -- splits by actual token count
# More precise for LLM context limits (tokens != characters)
splitter = TokenTextSplitter(
    chunk_size=256,    # tokens per chunk
    chunk_overlap=20
)

# MarkdownHeaderTextSplitter -- preserves document structure
# Each chunk gets metadata with its header hierarchy
from langchain.text_splitter import MarkdownHeaderTextSplitter
headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)
# chunks[0].metadata = {'H1': 'Introduction', 'H2': 'Overview'}

# Code splitter -- splits by function/class boundaries
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100
)
```

**Pros/Cons of Text Splitters:**
```
RecursiveCharacter: + best general purpose, respects natural boundaries
                    - still can cut mid-sentence if paragraph is too long
CharacterSplitter:  + simple, fast
                    - can produce chunks larger than chunk_size if separator not found
TokenTextSplitter:  + precise token control
                    - slower (must tokenize), no semantic awareness
MarkdownHeader:     + preserves document structure in metadata
                    - only works for markdown, not PDF/DOCX
SemanticChunker:    + best quality chunks (groups related sentences)
                    - slow (needs embedding call per sentence), expensive
```

**chunk_size vs chunk_overlap:**
```
chunk_size:    max characters per chunk. Too small = loses context. Too large = exceeds LLM limit.
chunk_overlap: shared text between consecutive chunks.
               Prevents losing info that spans a boundary.
               Rule of thumb: overlap = 10% of chunk_size
               chunk_size=500, chunk_overlap=50 is a good starting point
```



﻿### 4. Embeddings

Embeddings convert text into numerical vectors that capture semantic meaning.

| Model | Dims | Cost | Quality | Use When |
|-------|------|------|---------|----------|
| OpenAI text-embedding-3-small | 1536 | Paid | High | Production, best accuracy |
| OpenAI text-embedding-3-large | 3072 | Paid | Highest | When quality is critical |
| HuggingFace all-MiniLM-L6-v2 | 384 | Free | Good | Local dev, fast prototyping |
| HuggingFace all-mpnet-base-v2 | 768 | Free | Better | Local, better quality |
| BAAI/bge-small-en-v1.5 | 384 | Free | Best OSS | Best free option for RAG |
| Ollama nomic-embed-text | 768 | Free | Good | Fully local, no internet |

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAIEmbeddings -- high quality, paid
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # cheaper, good quality
    # model="text-embedding-3-large" # better, more expensive
)

# HuggingFaceEmbeddings -- FREE, runs locally, no API key
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # best free RAG model
    model_kwargs={"device": "cpu"},        # "cuda" if GPU available
    encode_kwargs={"normalize_embeddings": True}  # needed for cosine similarity
)

# Methods:
# embed_query()     -- embed ONE string (for search queries at retrieval time)
# embed_documents() -- embed LIST of strings (for indexing documents)
# aembed_query()    -- async version of embed_query
# aembed_documents()-- async version of embed_documents

vector = embeddings.embed_query("What is RAG?")
print(f"Dimensions: {len(vector)}")  # 384 for MiniLM, 1536 for OpenAI

vectors = embeddings.embed_documents(["text1", "text2", "text3"])
print(f"Shape: {len(vectors)} x {len(vectors[0])}")

# Why normalize_embeddings=True?
# Normalized vectors: cosine_similarity = dot_product (faster)
# Without normalization: must divide by magnitudes every search
```

**Pros/Cons:**
```
OpenAI:     + best quality, multilingual, 1536-dim
            - costs money, data leaves your server, needs internet
HuggingFace:+ free, private, offline, no API key
            - slower on CPU, lower quality than OpenAI
BAAI/bge:   + best open-source quality for RAG retrieval
            - larger model, more RAM
MiniLM:     + very fast, tiny (80MB), good enough for most RAG
            - 384-dim only, less accurate on complex queries
```

---
### 5. Vector Stores (LangChain Interface)

```python
from langchain_community.vectorstores import Chroma, FAISS
from langchain_qdrant import QdrantVectorStore

# Create from documents (indexes + stores in one call)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_docs"
)

# Load existing store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_docs"
)

# add_documents() -- add more docs to existing store
vectorstore.add_documents(new_chunks)

# add_texts() -- add raw strings with optional metadata
vectorstore.add_texts(texts=["text1","text2"], metadatas=[{"src":"a"},{"src":"b"}])

# similarity_search() -- returns List[Document], no scores
docs = vectorstore.similarity_search("What is RAG?", k=4)

# similarity_search_with_score() -- returns List[Tuple[Document, float]]
for doc, score in vectorstore.similarity_search_with_score("What is RAG?", k=4):
    print(f"Score: {score:.4f} | {doc.page_content[:60]}")

# similarity_search_by_vector() -- search with pre-computed vector
qv = embeddings.embed_query("What is RAG?")
docs = vectorstore.similarity_search_by_vector(qv, k=4)

# max_marginal_relevance_search() -- diverse results (MMR)
# Fetches fetch_k candidates, picks k most diverse ones
# Use when top results are too similar/redundant
docs = vectorstore.max_marginal_relevance_search("What is RAG?",
    k=4,          # final results to return
    fetch_k=20,   # candidates to fetch first
    lambda_mult=0.5  # 0=max diversity, 1=max relevance
)

# delete() -- remove by ID
vectorstore.delete(ids=["id1", "id2"])

# as_retriever() -- convert to retriever for use in chains
r1 = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4})
r2 = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":4,"fetch_k":20})
r3 = vectorstore.as_retriever(search_type="similarity_score_threshold",
     search_kwargs={"score_threshold":0.7,"k":4})

# similarity vs MMR vs threshold:
# similarity:  fastest, most similar docs (may be redundant)
# mmr:         diverse docs, better answer coverage, slightly slower
# threshold:   only confident results, avoids irrelevant chunks
```

---
### 6. Chains - LCEL Deep Dive

LCEL uses the | pipe operator. Every component is a Runnable.

```python
from langchain_core.runnables import (
    RunnablePassthrough,   # passes input unchanged
    RunnableParallel,      # runs multiple runnables simultaneously
    RunnableLambda,        # wraps any Python function as a runnable
    RunnableBranch,        # conditional routing
)

# Basic chain
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "RAG"})

# RunnablePassthrough -- passes original input unchanged
# Use when you need original input alongside transformed data
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
answer = rag_chain.invoke("What is RAG?")

# RunnableParallel -- run multiple chains at the same time
# All branches execute simultaneously, results combined into dict
parallel = RunnableParallel(
    summary=summarize_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain
)
result = parallel.invoke({"text": "Article..."})
# result = {"summary": "...", "keywords": "...", "sentiment": "..."}

# RunnableLambda -- wrap any Python function
def add_context(query: str) -> dict:
    docs = retriever.get_relevant_documents(query)
    return {"context": "\n".join(d.page_content for d in docs), "question": query}

chain = RunnableLambda(add_context) | prompt | llm | StrOutputParser()

# RunnableBranch -- conditional routing
branch = RunnableBranch(
    (lambda x: "code" in x["topic"].lower(), code_chain),
    (lambda x: "math" in x["topic"].lower(), math_chain),
    general_chain  # default fallback
)

# .bind() -- pre-fill parameters, returns new runnable
llm_json  = llm.bind(response_format={"type": "json_object"})
llm_stop  = llm.bind(stop=["###", "END"])
llm_tools = llm.bind_tools(tools)

# .with_retry() -- auto retry on failure
chain_retry = chain.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

# .with_fallbacks() -- use backup if primary fails
safe_chain = primary_chain.with_fallbacks([backup_chain])

# Streaming
for chunk in chain.stream({"topic": "RAG"}):
    print(chunk, end='', flush=True)

# Async
result = await chain.ainvoke({"topic": "RAG"})
async for chunk in chain.astream({"topic": "RAG"}):
    print(chunk, end='', flush=True)

# Batch -- runs all inputs in parallel
results = chain.batch([{"topic": "RAG"}, {"topic": "LangChain"}, {"topic": "LangGraph"}])
```

**Pros/Cons of LCEL:**
```
PROS:
  + Streaming works automatically through the whole chain
  + Async and batch support built-in
  + Parallel execution with RunnableParallel
  + Easy retry, fallback, config at any step
  + Composable -- mix any runnables freely
CONS:
  - Pipe syntax confusing for beginners
  - Errors deep in chain are hard to trace
  - No compile-time type checking
```

---
### 7. Memory

| Type | Stores | Pros | Cons | Use When |
|------|--------|------|------|----------|
| BufferMemory | All messages | Simple, no loss | Grows forever, expensive | Short chats |
| WindowMemory | Last K messages | Controls tokens | Loses old context | Medium chats |
| SummaryMemory | LLM summary | Handles long chats | Extra LLM call | Long chats |
| SummaryBufferMemory | Summary + recent | Best of both | Extra LLM call | Production (recommended) |
| EntityMemory | Named entities | Tracks facts | Complex, extra LLM | Entity tracking |
| VectorStoreMemory | Vector DB | Scales to 1000s | Complex setup | Very long history |

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
)

# BufferMemory -- stores ALL messages
memory = ConversationBufferMemory(memory_key='history', return_messages=True)

# WindowMemory -- only last K exchanges
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# SummaryMemory -- LLM summarizes old conversation
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

# SummaryBufferMemory -- RECOMMENDED for production
# Keeps recent messages exact + summarizes older ones
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=500,  # summarize when buffer exceeds this
    return_messages=True
)

# Manual operations
memory.save_context({'input': 'My name is Sujit'}, {'output': 'Hi Sujit!'})
print(memory.load_memory_variables({}))
memory.clear()  # wipe all memory

# Use with ConversationChain
from langchain.chains import ConversationChain
chain = ConversationChain(llm=llm, memory=memory, verbose=True)
chain.predict(input='Hi, my name is Sujit')
chain.predict(input='What is my name?')  # remembers Sujit
```

---
### 8. Agents

| Agent Type | How it works | Pros | Cons | Use When |
|-----------|-------------|------|------|----------|
| ReAct | Text Thought/Action/Observation loop | Works with any LLM | Fragile parsing, slow | Any LLM |
| OpenAI Tools | Native JSON tool calling | Reliable, fast, structured | OpenAI/Anthropic only | OpenAI models |
| Structured Chat | Structured output for tool args | Good for complex args | More setup | Multi-arg tools |

```python
from langchain.agents import (
    create_react_agent,            # ReAct: Thought->Action->Observation
    create_openai_tools_agent,     # OpenAI native tool calling (recommended)
    create_structured_chat_agent,  # structured output for tool args
    AgentExecutor,                 # runs the agent loop
)
from langchain.tools import tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define tools with @tool decorator
# Docstring = description shown to LLM -- write it clearly!
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Input: valid Python math like 2+2*3."""
    try: return str(eval(expression))
    except Exception as e: return f'Error: {e}'

@tool
def get_word_length(word: str) -> int:
    """Returns the number of characters in a word."""
    return len(word)

tools = [calculator, get_word_length]

# --- ReAct Agent ---
# Works with ANY LLM. Uses text-based reasoning.
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(
    agent=agent, tools=tools,
    verbose=True,                  # print all Thought/Action/Observation
    max_iterations=10,             # prevent infinite loops
    handle_parsing_errors=True,    # recover from bad LLM output format
    return_intermediate_steps=True # return all reasoning steps
)
result = executor.invoke({"input": "What is 25 * 4 + 10?"})
print(result['output'])
print(result['intermediate_steps'])  # list of (action, observation) tuples

# --- OpenAI Tools Agent (RECOMMENDED for OpenAI models) ---
# Uses native JSON tool calling -- more reliable than ReAct text parsing
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "How many letters in artificial intelligence?"})

# AgentExecutor key params:
# max_iterations:          prevent infinite loops (default 15)
# max_execution_time:      timeout in seconds
# handle_parsing_errors:   recover from LLM output format errors
# return_intermediate_steps: return all reasoning steps in output
```

---
### 9. Callbacks

```python
from langchain.callbacks.base import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    # on_llm_start      -- before every LLM call
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM starting, {len(prompts)} prompts")

    # on_llm_end        -- after every LLM call (token usage here)
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            print(f"Tokens used: {usage.get(\"total_tokens\", 0)}")

    # on_llm_error      -- when LLM raises an error
    def on_llm_error(self, error, **kwargs):
        print(f"LLM error: {error}")

    # on_llm_new_token  -- each streaming token
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end='', flush=True)

    # on_chain_start    -- before a chain runs
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain: {serialized.get(\"name\", \"unknown\")}")

    # on_chain_end      -- after a chain finishes
    def on_chain_end(self, outputs, **kwargs):
        print("Chain done")

    # on_tool_start     -- before a tool is called
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool: {serialized[\"name\"]} | Input: {input_str}")

    # on_tool_end       -- after tool returns result
    def on_tool_end(self, output, **kwargs):
        print(f"Tool result: {output[:100]}")

    # on_agent_action   -- when agent decides to use a tool
    def on_agent_action(self, action, **kwargs):
        print(f"Agent action: {action.tool} | {action.tool_input}")

    # on_agent_finish   -- when agent gives final answer
    def on_agent_finish(self, finish, **kwargs):
        print(f"Agent finished: {finish.return_values}")

    # on_retriever_start -- before retriever searches
    def on_retriever_start(self, serialized, query, **kwargs):
        print(f"Retrieving: {query}")

    # on_retriever_end   -- after retriever returns docs
    def on_retriever_end(self, documents, **kwargs):
        print(f"Retrieved {len(documents)} docs")

# Attach to LLM, chain, or executor
cb = MyCallback()
llm = ChatOpenAI(callbacks=[cb])
executor = AgentExecutor(agent=agent, tools=tools, callbacks=[cb])
```

---
### 10. Retrieval Chains

| Chain | Style | Pros | Cons |
|-------|-------|------|------|
| RetrievalQA | Old style | Simple, quick | Less flexible |
| ConversationalRetrievalChain | Old style | Has memory built-in | Verbose setup |
| LCEL RAG chain | Modern | Flexible, streaming, async | More code |

```python
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# RetrievalQA -- simple one-shot RAG
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",          # how to combine retrieved docs
    retriever=retriever,
    return_source_documents=True
)
result = qa.invoke({"query": "What is RAG?"})
print(result['result'])
print(result['source_documents'])

# chain_type options:
# "stuff"       -- all docs in one prompt. Fast. Fails if too many docs.
# "map_reduce"  -- summarize each doc separately, then combine. Good for many docs.
# "refine"      -- iteratively refine answer with each doc. Most thorough, slowest.
# "map_rerank"  -- score each doc answer, return highest. Good for factual Q&A.

# ConversationalRetrievalChain -- RAG with memory
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory, return_source_documents=True
)
conv_chain.invoke({"question": "What is RAG?"})
conv_chain.invoke({"question": "Give an example?"})  # uses memory

# Modern LCEL RAG chain (RECOMMENDED)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template('''
Answer using ONLY the context. Say I dont know if not in context.
Context: {context}
Question: {question}
''')

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
answer = rag_chain.invoke("What is RAG?")

# With source documents
rag_with_sources = RunnableParallel(
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
).assign(answer=prompt | llm | StrOutputParser())
result = rag_with_sources.invoke("What is RAG?")
print(result['answer'])
print(result['context'])
```

---

> ---
> **Section Summary:** LangChain is the go-to framework for LLM apps. Core packages: langchain-core (base), langchain-openai/groq/anthropic (model integrations), langchain-community (100+ loaders/vectorstores). Key components: Models (invoke/batch/stream/async), Document Loaders (PyPDF, Web, Directory), Text Splitters (RecursiveCharacter is best general-purpose), Embeddings (HuggingFace=free, OpenAI=quality), Vector Stores (similarity/MMR/threshold search), LCEL chains (pipe syntax with RunnablePassthrough/Parallel/Lambda/Branch), Memory (SummaryBufferMemory recommended for production), Agents (OpenAI Tools agent recommended over ReAct for OpenAI models), Callbacks (track tokens, log events), Retrieval Chains (LCEL RAG chain recommended over old RetrievalQA).


---

## 11. LangGraph - Deep Dive

LangGraph builds multi-step AI workflows as directed graphs. Built on LangChain.

**Pros/Cons:**
```
PROS:
  + Handles loops, branches, retries -- impossible in plain LangChain chains
  + Shared state across all nodes -- no manual data passing
  + Built-in checkpointing -- resume from failure, save tokens
  + Human-in-the-Loop -- pause and wait for human approval
  + Multi-agent coordination -- multiple specialized agents in one graph
  + Streaming support -- stream tokens from any node
  + Works with any LangChain-compatible LLM
CONS:
  - More complex than simple chains -- overkill for linear pipelines
  - State schema must be defined upfront (TypedDict)
  - Debugging graph execution can be tricky
  - Checkpointing adds overhead for simple workflows
```

### Core Concepts
```
StateGraph   -- the graph builder, takes state schema as input
State        -- TypedDict shared across ALL nodes
Node         -- function: reads state, returns dict of updates
Edge         -- connection between nodes (fixed or conditional)
START        -- entry point constant
END          -- exit point constant
Checkpointer -- saves state after each node
```

### State Design
```python
from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Annotated[List, operator.add] = APPEND new values (not replace)
    # Without Annotated, returning {'messages': [...]} REPLACES the list
    messages: Annotated[List[BaseMessage], operator.add]
    context:  str          # replaced each time
    iteration: int         # replaced each time
    error: Optional[str]   # replaced each time

# Reducer functions:
# operator.add  -- appends: [1,2] + [3] = [1,2,3]  (use for messages)
# operator.or_  -- merges dicts
# default       -- replaces with new value (no Annotated needed)
```

### Building a Graph
```python
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Nodes -- functions that read state and return updates
def call_llm(state: AgentState) -> dict:
    response = llm.invoke(state['messages'])
    return {'messages': [response], 'iteration': state['iteration'] + 1}

def call_tool(state: AgentState) -> dict:
    last_msg = state['messages'][-1]
    tool_call = last_msg.tool_calls[0]
    result = execute_tool(tool_call['name'], tool_call['args'])
    tool_msg = ToolMessage(content=str(result), tool_call_id=tool_call['id'])
    return {'messages': [tool_msg]}

# Conditional edge -- returns name of next node
def route(state: AgentState) -> str:
    last_msg = state['messages'][-1]
    if state['iteration'] >= 10: return 'error'
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls: return 'tool'
    return END

# Build graph
graph = StateGraph(AgentState)

# add_node(name, function)
graph.add_node("llm",  call_llm)
graph.add_node("tool", call_tool)

# add_edge(from, to) -- fixed connection
graph.add_edge(START, "llm")   # always start at llm
graph.add_edge("tool", "llm")  # after tool, always back to llm

# add_conditional_edges(from, routing_fn, mapping)
graph.add_conditional_edges("llm", route, {"tool": "tool", END: END})

# compile() -- validates graph, returns runnable app
app = graph.compile()

# Run
result = app.invoke({
    "messages": [HumanMessage(content="What is 25 * 4?")],
    "context": "", "iteration": 0, "error": None
})
print(result['messages'][-1].content)
```

---
### Checkpointing

| Checkpointer | Storage | Pros | Cons | Use When |
|-------------|---------|------|------|----------|
| MemorySaver | RAM | Zero setup, fast | Lost on restart | Dev/testing |
| SqliteSaver | SQLite file | Persistent, no server | Single instance only | Single-server apps |
| PostgresSaver | PostgreSQL | Production-grade, multi-instance | Needs DB server | Production |

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# MemorySaver -- in-memory, lost on restart
checkpointer = MemorySaver()

# SqliteSaver -- persists to SQLite file
checkpointer = SqliteSaver.from_conn_string("./checkpoints.db")

app = graph.compile(checkpointer=checkpointer)

# thread_id = unique ID per conversation/session
config = {"configurable": {"thread_id": "user_123_session_1"}}

# First message
result = app.invoke({"messages": [HumanMessage("Hello, I am Sujit")]}, config=config)

# Second message -- agent remembers previous context from checkpoint
result = app.invoke({"messages": [HumanMessage("What is my name?")]}, config=config)
# Agent knows the name is Sujit

# Inspect checkpoint state
state = app.get_state(config)
print(state.values)   # current state dict
print(state.next)     # which node runs next (empty if finished)

# Full history of all checkpoints
for checkpoint in app.get_state_history(config):
    print(checkpoint.metadata['step'], checkpoint.values['messages'][-1].content)

# Update state manually
app.update_state(config, {"context": "Updated context from human"})

# Why checkpointing?
# Without: agent fails at step 5 of 10 -> restart from step 1 (waste tokens)
# With:    agent fails at step 5 -> resume from step 5 (save tokens + time)
```

### Human-in-the-Loop (HITL)
```python
# interrupt_before -- pause BEFORE executing specified nodes
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tool"]  # pause before every tool call
)

config = {"configurable": {"thread_id": "hitl_1"}}

# Run until interrupt
result = app.invoke({"messages": [HumanMessage("Send email to team")]}, config=config)
# Execution pauses before tool node

# Check what agent wants to do
state = app.get_state(config)
print('Agent wants to:', state.next)  # ('tool',)
print('Tool call:', state.values['messages'][-1].tool_calls)

# Option 1: Approve -- pass None to resume from checkpoint
result = app.invoke(None, config=config)

# Option 2: Modify -- update state then resume
app.update_state(config, {'messages': [HumanMessage('Send to manager only')]})
result = app.invoke(None, config=config)

# interrupt_after -- pause AFTER executing specified nodes
app = graph.compile(checkpointer=MemorySaver(), interrupt_after=['llm'])
```

### Streaming
```python
# stream() -- yields events as each node completes
for event in app.stream(
    {"messages": [HumanMessage("What is 2+2?")]}, config=config
):
    for node_name, node_output in event.items():
        if node_name != "__end__":
            print(f"[{node_name}]:", node_output)

# stream_mode options:
# "values"  -- full state after each node (default)
# "updates" -- only the changes each node made
# "debug"   -- detailed debug info

# astream_events() -- async, stream individual tokens from LLM nodes
async for event in app.astream_events(
    {"messages": [HumanMessage("Explain RAG")]}, config=config, version="v1"
):
    if event['event'] == 'on_chat_model_stream':
        print(event['data']['chunk'].content, end='', flush=True)
    elif event['event'] == 'on_tool_start':
        print(f"\nCalling tool: {event[\"name\"]}")
```

### Multi-Agent Supervisor Pattern
```python
class TeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    task: str
    results: dict

def supervisor(state: TeamState) -> dict:
    decision = supervisor_llm.invoke(
        f'Which agent next? Options: researcher, coder, reviewer, FINISH'
    )
    return {'next_agent': decision.content.strip()}

def researcher(state): return {'results': {**state['results'], 'research': research_llm.invoke(state['task']).content}}
def coder(state):      return {'results': {**state['results'], 'code': code_llm.invoke(state['task']).content}}
def reviewer(state):   return {'results': {**state['results'], 'review': review_llm.invoke(state['results'].get('code','')).content}}

graph = StateGraph(TeamState)
for name, fn in [('supervisor',supervisor),('researcher',researcher),('coder',coder),('reviewer',reviewer)]:
    graph.add_node(name, fn)

graph.set_entry_point('supervisor')
graph.add_conditional_edges('supervisor', lambda s: s['next_agent'].lower(),
    {'researcher':'researcher','coder':'coder','reviewer':'reviewer','finish':END})
for agent in ['researcher','coder','reviewer']:
    graph.add_edge(agent, 'supervisor')  # all agents report back to supervisor

app = graph.compile(checkpointer=MemorySaver())
```

---

> ---
> **Section Summary:** LangGraph is for multi-step agentic workflows that need loops, branches, and shared state -- impossible in plain LangChain chains. Core concepts: State (TypedDict shared across all nodes), Nodes (functions that update state), Edges (fixed or conditional connections), START/END constants. Use Annotated[List, operator.add] for message lists so new messages append instead of replace. Checkpointing enables conversation memory and failure recovery (MemorySaver for dev, SqliteSaver for single-server, PostgresSaver for production). HITL uses interrupt_before to pause before critical nodes. Multi-agent supervisor pattern: one LLM routes tasks to specialized agents.


---

## 12. LlamaIndex - Deep Dive

LlamaIndex is a data framework for LLM apps. Best for advanced indexing and document Q&A.

**LlamaIndex vs LangChain:**
```
LlamaIndex:  focused on data ingestion, indexing, querying
             more index types, better for complex document Q&A
             SimpleDirectoryReader handles 50+ file types automatically
LangChain:   broader -- agents, chains, tools, integrations
             better for multi-step workflows and agents
Both can do RAG. Use LlamaIndex when indexing strategy matters.
```

**Pros/Cons:**
```
PROS:
  + Best-in-class data ingestion (50+ file types)
  + Multiple index types for different query patterns
  + Advanced retrieval -- re-ranking, hybrid, recursive
  + Source tracing -- know exactly which nodes answered
  + Built-in chat engine with memory
CONS:
  - Steeper learning curve than LangChain
  - Smaller community, fewer tool integrations
  - Documentation can be inconsistent
```

### Setup
```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Global settings -- applied to all indexes and queries
Settings.llm = OpenAI(model='gpt-4', temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')
Settings.chunk_size = 512
Settings.chunk_overlap = 50
```

### Document Loading
```python
from llama_index.core import SimpleDirectoryReader, Document

# SimpleDirectoryReader -- auto-detects PDF, DOCX, TXT, CSV, JSON, HTML
documents = SimpleDirectoryReader(
    input_dir='./data/',
    recursive=True,                  # include subdirectories
    required_exts=['.pdf', '.txt'],  # only these extensions
    filename_as_id=True              # use filename as document ID
).load_data()

# Specific files
documents = SimpleDirectoryReader(input_files=['report.pdf','notes.txt']).load_data()

# Manual Document
doc = Document(
    text='RAG stands for Retrieval-Augmented Generation...',
    metadata={'source': 'rag_guide.pdf', 'page': 3, 'workspace_id': 'ws_123'},
    doc_id='doc_001',
    excluded_llm_metadata_keys=['workspace_id'],  # hide from LLM prompt
    excluded_embed_metadata_keys=['author']        # hide from embedding
)

print(documents[0].text[:200])
print(documents[0].metadata)
print(documents[0].doc_id)
```

---
### Node Parsers (Chunking)

| Parser | How it splits | Pros | Cons | Use When |
|--------|--------------|------|------|----------|
| SentenceSplitter | Sentence boundaries | Fast, good quality | Fixed size | Most cases (default) |
| TokenTextSplitter | Token count | Precise LLM limit | Slower, may cut sentences | Token-sensitive |
| SemanticSplitter | Semantic similarity | Best quality chunks | Slow, expensive | Quality critical |
| MarkdownNodeParser | Markdown headers | Preserves structure | Markdown only | Markdown docs |
| HierarchicalNodeParser | Parent-child tree | Small retrieve, large context | Complex setup | Advanced RAG |
| SentenceWindowNodeParser | Sentence + window | Small retrieve, window to LLM | Larger prompts | Advanced RAG |

```python
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    SemanticSplitterNodeParser,
    MarkdownNodeParser,
    HierarchicalNodeParser,
    SentenceWindowNodeParser,
)

# SentenceSplitter -- MOST COMMONLY USED
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

# SemanticSplitterNodeParser -- groups semantically similar sentences
# PROS: best quality chunks  CONS: slow, expensive (embedding per sentence)
from llama_index.embeddings.openai import OpenAIEmbedding
parser = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,  # higher = fewer, larger chunks
    embed_model=OpenAIEmbedding()
)

# SentenceWindowNodeParser -- stores surrounding sentences as metadata
# Retrieve small chunks, pass larger window to LLM for context
parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,                    # sentences on each side
    window_metadata_key='window',
    original_text_metadata_key='original_text'
)
# Use with MetadataReplacementPostProcessor at query time

# HierarchicalNodeParser -- parent-child hierarchy
# Retrieve small child chunks, auto-merge into parent for LLM
parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
# Use with AutoMergingRetriever

# Node properties
print(nodes[0].text)
print(nodes[0].metadata)
print(nodes[0].node_id)
print(nodes[0].relationships)  # parent/child/prev/next
```

---
### Index Types

| Index | How it retrieves | Pros | Cons | Use When |
|-------|-----------------|------|------|----------|
| VectorStoreIndex | Cosine similarity | Fast, semantic, scalable | Needs embeddings | Most RAG use cases |
| SummaryIndex | Reads ALL nodes | Great for summarization | Slow, expensive | Summarize all docs |
| KeywordTableIndex | Keyword match | No embeddings needed | Misses semantic | Exact term lookup |
| DocumentSummaryIndex | Per-doc summary + nodes | Both summary and Q&A | Expensive to build | Mixed query types |

```python
from llama_index.core import VectorStoreIndex, SummaryIndex, KeywordTableIndex

# VectorStoreIndex -- MOST USED, semantic search
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# With external vector store (Chroma)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
chroma_client = chromadb.PersistentClient(path='./chroma_db')
chroma_col = chroma_client.get_or_create_collection('docs')
vector_store = ChromaVectorStore(chroma_collection=chroma_col)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# SummaryIndex -- reads ALL nodes to answer
# Good for: 'Summarize all documents', 'What are the main themes?'
# Bad for:  specific fact lookup (reads everything = slow + expensive)
summary_index = SummaryIndex.from_documents(documents)

# KeywordTableIndex -- keyword-based, no embeddings
# Good for: 'Find all mentions of GDPR', exact term lookup
keyword_index = KeywordTableIndex.from_documents(documents)

# Persist and Load (avoid re-embedding every time)
index.storage_context.persist(persist_dir='./storage')

from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir='./storage')
index = load_index_from_storage(storage_context)

# Add / delete / refresh documents
index.insert(new_document)
index.insert_nodes(new_nodes)
index.delete_ref_doc('doc_001', delete_from_docstore=True)
refreshed = index.refresh_ref_docs(documents)  # update changed docs
```

---
### Query Engines

| response_mode | How it works | Pros | Cons | Use When |
|--------------|-------------|------|------|----------|
| compact | Combine nodes, 1-2 LLM calls | Fast, good balance | May truncate | Default, most cases |
| refine | Iteratively refine with each node | Most thorough | Slowest, N LLM calls | Complex questions |
| tree_summarize | Build summary tree bottom-up | Scalable for many nodes | Slower than compact | Many retrieved nodes |
| simple_summarize | Truncate all nodes, 1 LLM call | Fastest | May lose info | Speed critical |
| no_text | Return source nodes only | Retrieval-only | No answer generated | Custom generation |
| accumulate | Query each node separately | Good for aggregation | N LLM calls | Count/list questions |

```python
# Basic query engine
query_engine = index.as_query_engine(
    similarity_top_k=4,
    response_mode='compact'  # see table above
)
response = query_engine.query('What is RAG?')
print(response.response)
print(response.source_nodes)   # which nodes were used
print(response.metadata)       # metadata from source nodes

# Streaming
query_engine = index.as_query_engine(streaming=True)
response = query_engine.query('Explain the main concepts')
response.print_response_stream()

# Async
query_engine = index.as_query_engine(use_async=True)
response = await query_engine.aquery('What is RAG?')

# Custom prompt
from llama_index.core import PromptTemplate
qa_prompt = PromptTemplate(
    'Context: {context_str}\n\nQuestion: {query_str}\n\n'
    'Answer using ONLY the context. Say I dont know if not in context.\nAnswer: '
)
query_engine = index.as_query_engine(text_qa_template=qa_prompt, similarity_top_k=4)
```

### Chat Engines

| chat_mode | How it works | Pros | Cons | Use When |
|-----------|-------------|------|------|----------|
| simple | No retrieval, just LLM | Fast | No document knowledge | Chitchat |
| condense_question | Condense history, then retrieve | Handles follow-ups | Extra LLM call | Follow-up questions |
| context | Retrieve for every message | Always grounded | May retrieve irrelevant | Always need docs |
| condense_plus_context | Condense + retrieve | Best for RAG chatbots | Two LLM calls | Production RAG chat |
| react | ReAct agent with index as tool | Decides when to search | Slower | Complex reasoning |

```python
chat_engine = index.as_chat_engine(chat_mode='condense_plus_context', verbose=True)

response = chat_engine.chat('What is RAG?')
print(response.response)
print(response.source_nodes)

response2 = chat_engine.chat('Give an example?')  # remembers context

chat_engine.reset()  # clear conversation history

# Streaming chat
response = chat_engine.stream_chat('Explain RAG in detail')
for token in response.response_gen:
    print(token, end='', flush=True)
```

---
### Retrievers

| Retriever | Method | Pros | Cons | Use When |
|-----------|--------|------|------|----------|
| VectorIndexRetriever | Cosine similarity | Fast, semantic | Misses exact keywords | Default |
| BM25Retriever | Keyword ranking | No embeddings, exact match | Misses semantic | Keyword search |
| QueryFusionRetriever | Vector + BM25 combined | Best coverage | More complex | Production RAG |
| RouterRetriever | LLM picks best retriever | Smart routing | Extra LLM call | Multiple data types |
| AutoMergingRetriever | Merges child into parent | Small retrieve, large context | Complex setup | Hierarchical nodes |

```python
from llama_index.core.retrievers import VectorIndexRetriever

# VectorIndexRetriever -- most common
retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
nodes = retriever.retrieve('What is RAG?')
for node in nodes:
    print(f'Score: {node.score:.4f} | {node.text[:80]}')

# BM25Retriever -- keyword-based, no embeddings
from llama_index.retrievers.bm25 import BM25Retriever
bm25 = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=4)

# QueryFusionRetriever -- hybrid vector + BM25
from llama_index.core.retrievers import QueryFusionRetriever
hybrid = QueryFusionRetriever(
    retrievers=[VectorIndexRetriever(index=index), bm25],
    similarity_top_k=4,
    num_queries=4,              # generate N query variations
    mode='reciprocal_rerank',   # RRF algorithm to combine results
    use_async=True
)
# mode options:
# 'reciprocal_rerank' -- RRF algorithm, good default
# 'dist_based_score'  -- normalize and combine scores
# 'relative_score'    -- relative score combination

# RouterRetriever -- LLM decides which retriever to use
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import LLMSingleSelector

vector_tool = RetrieverTool.from_defaults(
    retriever=VectorIndexRetriever(index=index),
    description='Useful for semantic questions about AI concepts'
)
keyword_tool = RetrieverTool.from_defaults(
    retriever=bm25,
    description='Useful for finding specific terms or exact phrases'
)
router = RouterRetriever(
    selector=LLMSingleSelector.from_defaults(),
    retriever_tools=[vector_tool, keyword_tool]
)
nodes = router.retrieve('Find all mentions of GDPR')  # routes to keyword
```

### Post-Processors (Re-ranking)

| Post-Processor | Method | Pros | Cons |
|---------------|--------|------|------|
| SimilarityPostprocessor | Filter by score threshold | Simple, fast | May filter too many |
| LLMRerank | LLM re-ranks nodes | Best quality | Expensive, slow |
| SentenceTransformerRerank | Cross-encoder model | Good quality, free | Model download needed |
| MetadataReplacementPostProcessor | Replace text with window | Better context | Larger prompts |

```python
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    LLMRerank,
    MetadataReplacementPostProcessor,
)

# Filter by similarity score
sim_filter = SimilarityPostprocessor(similarity_cutoff=0.7)

# LLM re-ranking -- best quality, expensive
llm_rerank = LLMRerank(choice_batch_size=5, top_n=3)

# Cross-encoder re-ranking -- good quality, free
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
reranker = FlagEmbeddingReranker(model='BAAI/bge-reranker-base', top_n=3)

# Use in query engine
query_engine = index.as_query_engine(
    similarity_top_k=10,            # retrieve 10 candidates
    node_postprocessors=[reranker]  # re-rank to top 3
)

# MetadataReplacementPostProcessor -- for SentenceWindowNodeParser
# Replaces small chunk text with the surrounding window stored in metadata
window_pp = MetadataReplacementPostProcessor(target_metadata_key='window')
query_engine = index.as_query_engine(
    similarity_top_k=4,
    node_postprocessors=[window_pp]
)
```

---



> ---
> **Section Summary:** LlamaIndex specializes in data indexing and document Q&A. SimpleDirectoryReader auto-loads 50+ file types. Node parsers: SentenceSplitter (default), SemanticSplitter (best quality, slow), SentenceWindowNodeParser (small retrieve + large context window). Index types: VectorStoreIndex (most used, semantic), SummaryIndex (summarization), KeywordTableIndex (exact match, no embeddings). Query engine response_mode: compact (default), refine (thorough), tree_summarize (many nodes), accumulate (aggregation). Chat engine chat_mode: condense_plus_context (recommended for RAG chatbots). Retrievers: QueryFusionRetriever for hybrid search, RouterRetriever for smart routing. Post-processors: FlagEmbeddingReranker for quality re-ranking.


---

## 13. Vector Databases - Deep Dive

Vector databases store embeddings and enable fast similarity search. Core to RAG, semantic search, recommendations.

### How Vector Search Works
```
1. Store:  text --> embedding model --> vector --> stored in VectorDB with metadata
2. Query:  query text --> same embedding model --> query vector
3. Search: compare query vector vs all stored vectors using cosine similarity
4. Return: top-k most similar vectors + their original text + metadata
```

### Cosine Similarity
```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Range: -1 to 1
# 1.0  = identical direction (same meaning)
# 0.0  = perpendicular (unrelated)
# -1.0 = opposite direction

v1 = np.array([0.2, 0.8, 0.1])
v2 = np.array([0.3, 0.7, 0.2])
print(cosine_similarity(v1, v2))  # high similarity
```

### ChromaDB - Complete Guide
```python
import chromadb

# In-memory (for testing)
client = chromadb.Client()

# Persistent (saves to disk)
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity
)

# Get existing / get-or-create (safe)
collection = client.get_collection("my_documents")
collection = client.get_or_create_collection("my_documents")

# Add documents
collection.add(
    documents=["RAG is a technique...", "LangChain is a framework..."],
    metadatas=[{"source": "doc1.pdf", "page": 1, "workspace_id": "ws_1"},
               {"source": "doc2.pdf", "page": 3, "workspace_id": "ws_1"}],
    ids=["id1", "id2"]   # must be unique strings
)

# Query
results = collection.query(
    query_texts=["What is RAG?"],   # ChromaDB embeds this automatically
    n_results=4,
    where={"workspace_id": "ws_1"},          # metadata filter
    where_document={"$contains": "RAG"},     # content filter
    include=["documents", "metadatas", "distances"]
)
print(results["documents"])
print(results["distances"])  # lower = more similar (1 - cosine_similarity)

# Update
collection.update(ids=["id1"], documents=["Updated text..."])

# Delete by ID
collection.delete(ids=["id1", "id2"])

# Delete by filter
collection.delete(where={"source": "old_doc.pdf"})

# Count
print(collection.count())

# With LangChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and persist
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_docs"
)
vectorstore.persist()

# Load existing
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_docs"
)

# Add more documents later
vectorstore.add_documents(new_chunks)

# Search methods
docs = vectorstore.similarity_search("What is RAG?", k=4)
docs_with_scores = vectorstore.similarity_search_with_score("What is RAG?", k=4)
for doc, score in docs_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:80]}")

# As retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",                          # default
    search_kwargs={"k": 4}
)
retriever = vectorstore.as_retriever(
    search_type="mmr",                                 # diverse results
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
)
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",          # only above threshold
    search_kwargs={"score_threshold": 0.7, "k": 4}
)
```

### FAISS - Complete Guide
```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create from documents
vectorstore = FAISS.from_documents(chunks, embeddings)

# Create from texts
vectorstore = FAISS.from_texts(
    texts=["text1", "text2", "text3"],
    embedding=embeddings,
    metadatas=[{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
)

# Save to disk
vectorstore.save_local("./faiss_index")

# Load from disk
vectorstore = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # required flag
)

# Add more documents
vectorstore.add_documents(new_chunks)

# Merge two FAISS stores
vectorstore.merge_from(another_faiss_store)

# Search
docs = vectorstore.similarity_search("What is RAG?", k=4)
docs_with_scores = vectorstore.similarity_search_with_score("What is RAG?", k=4)

# Why FAISS?
# - Extremely fast for local use
# - No server needed, just files
# - Good for prototyping and medium-scale production
# - Meta-developed, battle-tested
# - No filtering support (unlike Qdrant/Chroma)
```

### Qdrant - Complete Guide
```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)

# Connect
client = QdrantClient(":memory:")                     # in-memory
client = QdrantClient(url="http://localhost:6333")    # local Docker
client = QdrantClient(url="https://xxx.qdrant.io", api_key="key")  # cloud

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=384,                  # must match embedding dimensions
        distance=Distance.COSINE   # COSINE, EUCLID, or DOT
    )
)

# Upsert points (add or update)
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],   # 384-dim vector
            payload={                  # metadata (any JSON)
                "text": "RAG is...",
                "source": "doc.pdf",
                "workspace_id": "ws_123",
                "page": 1
            }
        )
    ]
)

# Search with filter (workspace isolation)
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, ...],
    limit=4,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="workspace_id",
                match=MatchValue(value="ws_123")  # only this workspace
            )
        ]
    ),
    with_payload=True
)
for r in results:
    print(f"Score: {r.score:.4f} | {r.payload['text'][:80]}")

# Filter with range
results = client.search(
    collection_name="documents",
    query_vector=query_vector,
    query_filter=Filter(
        must=[FieldCondition(key="page", range=Range(gte=1, lte=10))]
    ),
    limit=5
)

# Delete by filter
client.delete(
    collection_name="documents",
    points_selector=Filter(
        must=[FieldCondition(key="workspace_id", match=MatchValue(value="ws_old"))]
    )
)

# With LangChain
from langchain_qdrant import QdrantVectorStore
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="documents",
    embedding=embeddings
)
vectorstore.add_documents(chunks)
docs = vectorstore.similarity_search("What is RAG?", k=4)

# Why Qdrant over ChromaDB?
# Qdrant: production-grade, advanced filtering, cloud/local, REST + gRPC
# ChromaDB: simpler, plug-and-play, better for local dev
```

### Vector DB Comparison

| Feature | ChromaDB | FAISS | Qdrant | Pinecone |
|---------|----------|-------|--------|----------|
| Type | Open-source local | Library | Open-source | Managed cloud |
| Setup | Plug-and-play | Plug-and-play | Docker/Cloud | API key |
| Filtering | Basic | No | Advanced | Advanced |
| Scale | Small-medium | Medium | Large | Large |
| Persistence | SQLite | File | Server | Cloud |
| Best for | Local dev, RAG | Fast local search | Production | Production SaaS |
| Cost | Free | Free | Free/Paid | Paid |

---


> ---
> **Section Summary:** Vector DBs store embeddings and enable fast semantic search via cosine similarity. ChromaDB: plug-and-play local DB, best for dev (get_or_create_collection, query with where filter). FAISS: Meta library, fastest local search, no filtering support, good for medium scale. Qdrant: production-grade, advanced metadata filtering, workspace isolation via FieldCondition filter. Key methods: add/upsert, search/query with k and filters, delete by ID or filter. Always store workspace_id in metadata for multi-tenant isolation. Choose: ChromaDB for local dev, Qdrant for production with filtering needs, FAISS for pure speed.


---

## 14. MCP - Model Context Protocol

MCP was developed by Anthropic. Open standard for LLMs to connect with external tools and data using one common protocol.

### Problem MCP Solves
```
Before MCP:
  Every agent had custom API integrations -- messy, not reusable
  Agent A uses custom search API, Agent B uses different custom search API

With MCP:
  One standard protocol -- any MCP tool works with any MCP agent
  Like USB for AI tools
```

### MCP Architecture
```
LLM / AI Agent
      |
      v
   MCP Client  (inside your agent/app)
      |
      v  (MCP protocol over stdio or HTTP/SSE)
   MCP Server  (exposes tools/resources)
      |
      v
   External System (DB, File, API, Search, Browser)
```

### MCP vs Tool Calling
```
Tool Calling:
  - You define specific functions in your code
  - LLM calls them via JSON schema
  - Tightly coupled to your codebase

MCP:
  - Standardized protocol
  - Tools are discovered dynamically from MCP server
  - Loosely coupled -- swap tools without changing agent code
  - MCP server can be written in any language
```

### MCP Server Example (Python)
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio

app = Server("my-tools-server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_documents",
            description="Search through company documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_documents":
        query = arguments["query"]
        limit = arguments.get("limit", 5)
        results = vector_db.search(query, k=limit)
        return [types.TextContent(type="text", text=str(results))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

asyncio.run(main())
```

---


> ---
> **Section Summary:** MCP is Anthropic's open standard for LLMs to connect with external tools and data. Before MCP every agent had custom API integrations -- messy and not reusable. MCP provides one standard protocol (like USB for AI tools). Architecture: LLM > MCP Client > MCP Server > External System. Difference from tool calling: tool calling is tightly coupled to your code, MCP is a standardized protocol where tools are discovered dynamically. MCP servers expose list_tools() and call_tool() endpoints over stdio or HTTP.


---

## 15. Agentic AI - Deep Dive

### Agent Loop
```
Goal --> [LLM thinks] --> [Tool Call] --> [Observation] --> [LLM decides: done or loop?]
                                                                  |
                                              done --> Final Answer
                                              loop --> back to Tool Call
```

### ReAct Agent with LangChain
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

@tool
def search_web(query: str) -> str:
    """Search the web for current information. Input: search query string."""
    # integrate with SerpAPI, Tavily, DuckDuckGo, etc.
    return f"Search results for: {query}"

@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output. Input: valid Python code string."""
    import io, contextlib
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(code)
    return output.getvalue()

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Input: Python math expression like '2 + 2 * 3'."""
    return str(eval(expression))

tools = [search_web, run_python, calculator]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

result = executor.invoke({"input": "What is the square root of 144 plus 50?"})
print(result["output"])
print(result["intermediate_steps"])  # all Thought/Action/Observation steps
```

### Agent Memory
```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

memory = ConversationBufferWindowMemory(
    k=10,
    memory_key="chat_history",
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant with access to tools."),
    MessagesPlaceholder(variable_name="chat_history"),  # memory injected here
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

executor.invoke({"input": "My name is Sujit"})
executor.invoke({"input": "What is my name?"})  # remembers "Sujit"
```

### n8n Key Nodes for AI Workflows
```
HTTP Request     -- call any REST API (OpenAI, Groq, vector DB, etc.)
AI Agent         -- built-in LLM agent node
Code             -- run JavaScript or Python
If               -- conditional branching (route based on condition)
Loop Over Items  -- iterate over arrays (process each document)
Set              -- set/transform variables
Email            -- send emails (SMTP, Gmail)
Webhook          -- receive HTTP triggers from external systems
Schedule Trigger -- run on cron schedule
```

---


> ---
> **Section Summary:** An AI agent = LLM + Tools + Memory + Planning loop. The ReAct pattern (Thought > Action > Observation > repeat) is the foundation of most agents. Use create_openai_tools_agent for OpenAI models (reliable JSON tool calling) and create_react_agent for any LLM (text-based, more fragile). AgentExecutor key params: max_iterations (prevent loops), handle_parsing_errors (recover from bad output), return_intermediate_steps (see all reasoning). Add ConversationBufferWindowMemory with MessagesPlaceholder for agents that remember conversation. n8n is used for no-code agent orchestration with HTTP Request, AI Agent, If, and Loop nodes.


---

## 16. AI Security & Safety

### Prompt Injection Defense
```python
import re

INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"forget (your|all) instructions",
    r"you are now",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"DAN mode",
    r"system prompt",
    r"reveal your instructions"
]

def sanitize_input(user_input: str) -> str:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            raise ValueError("Potentially malicious input detected")
    if len(user_input) > 2000:
        raise ValueError("Input too long")
    return user_input.strip()
```

### PII Redaction
```python
import re

def redact_pii(text: str) -> str:
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', text)
    return text
```

### Output Moderation
```python
from openai import OpenAI
client = OpenAI()

def moderate_content(text: str) -> bool:
    response = client.moderations.create(input=text)
    result = response.results[0]
    if result.flagged:
        categories = {k: v for k, v in result.categories.__dict__.items() if v}
        print(f"Flagged: {categories}")
        return False
    return True

if not moderate_content(user_input):
    return {"error": "Input contains inappropriate content"}
```

### Workspace Isolation (Multi-tenant RAG)
```python
# ChromaDB -- filter by workspace_id
results = collection.query(
    query_texts=[query],
    n_results=4,
    where={"workspace_id": current_user_workspace_id}
)

# Qdrant -- filter by workspace_id
from qdrant_client.models import Filter, FieldCondition, MatchValue
results = qdrant_client.search(
    collection_name="documents",
    query_vector=query_vector,
    query_filter=Filter(
        must=[FieldCondition(
            key="workspace_id",
            match=MatchValue(value=current_user_workspace_id)
        )]
    ),
    limit=4
)
```

### API Key Security
```python
import os
from dotenv import load_dotenv

load_dotenv()

# NEVER hardcode keys in code
# ALWAYS load from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# .env file (never commit to git -- add to .gitignore)
# OPENAI_API_KEY=sk-abc123...
# GROQ_API_KEY=gsk_abc123...
```

---


> ---
> **Section Summary:** Key AI security threats: Prompt Injection (user hijacks LLM via malicious input -- defend with regex pattern matching), Jailbreaking (bypass safety filters -- use moderation API), Data Poisoning (malicious docs in RAG -- validate sources), PII Leakage (LLM reveals sensitive data -- redact before indexing). Always: sanitize inputs, moderate outputs with OpenAI moderation API, use HTTPBearer auth on all endpoints, store workspace_id in vector DB metadata for tenant isolation, load API keys from .env never hardcode them.


---

## 17. Scalability in AI Systems

### Async LLM Calls
```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def call_llm(query: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

async def process_batch(queries: list) -> list:
    # All queries run in parallel -- not one by one
    tasks = [call_llm(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results

# Sequential: 10 queries x 2s each = 20s total
# Parallel:   10 queries x 2s each = ~2s total
```

### Streaming with FastAPI SSE
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI()

async def stream_llm(query: str):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}],
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield f"data: {content}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/stream")
async def stream_endpoint(query: str):
    return StreamingResponse(
        stream_llm(query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

### Caching with Redis
```python
import redis
import hashlib
import json

r = redis.Redis(host='localhost', port=6379, db=0)

def cached_llm_call(query: str, context: str, ttl: int = 3600) -> str:
    key = hashlib.md5(f"{query}:{context}".encode()).hexdigest()
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    response = call_llm(query, context)
    r.setex(key, ttl, json.dumps(response))
    return response
```

### Rate Limiting
```python
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, query: str):
    return {"answer": await call_llm(query)}
```

### Token Optimization
```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def trim_context_to_limit(chunks: list, max_tokens: int = 3000) -> list:
    enc = tiktoken.encoding_for_model("gpt-4")
    total, trimmed = 0, []
    for chunk in chunks:
        t = len(enc.encode(chunk.page_content))
        if total + t > max_tokens:
            break
        trimmed.append(chunk)
        total += t
    return trimmed
```

---


> ---
> **Section Summary:** AI APIs are I/O bound (slow LLM calls) -- async is critical. Use AsyncOpenAI + asyncio.gather to run multiple LLM calls in parallel (10 calls x 2s = 2s total instead of 20s). Streaming (stream=True) improves perceived latency -- return tokens as they generate via SSE. Cache repeated queries with Redis (hash query+context as key, set TTL). Rate limit with slowapi (@limiter.limit). Count tokens with tiktoken and trim context to stay within limits. Production architecture: FastAPI > Auth > Rate Limit > RAG Pipeline > LLM > Cache > Stream response.


---

## 18. FastAPI for AI - Deep Dive

FastAPI is the go-to framework for AI APIs. Async-first, Pydantic validation, auto Swagger docs.

### Complete AI API
```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from typing import Optional
import os
from openai import OpenAI

# ── Lifespan (startup/shutdown) ──────────────────────────────────────────────
vectorstore = None
embeddings  = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, embeddings
    # Startup: load heavy resources once
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    print("Models loaded, ready!")
    yield
    # Shutdown: cleanup
    print("Shutting down...")

app = FastAPI(title="AI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Pydantic Models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=1, le=4096)

    @field_validator('query')
    @classmethod
    def clean_query(cls, v: str) -> str:
        return v.strip()

class ChatResponse(BaseModel):
    answer: str
    tokens_used: int
    model: str

class RAGRequest(BaseModel):
    query: str
    workspace_id: str
    top_k: int = Field(default=4, ge=1, le=10)

# ── Auth Dependency ───────────────────────────────────────────────────────────
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    valid_tokens = os.getenv("VALID_API_KEYS", "").split(",")
    if credentials.credentials not in valid_tokens:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "AI API"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, token: str = Depends(verify_token)):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": request.query}],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return ChatResponse(
            answer=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model=response.model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": request.query}],
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield f"data: {content}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/rag")
async def rag_query(request: RAGRequest):
    docs = vectorstore.similarity_search(
        request.query,
        k=request.top_k,
        filter={"workspace_id": request.workspace_id}
    )
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {request.query}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": [d.metadata.get("source", "unknown") for d in docs]
    }
```

### File Upload for RAG
```python
import tempfile, os

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: str = "default",
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    allowed = ["application/pdf", "text/plain"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="File type not supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    def process_in_background(path: str, ws_id: str):
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["workspace_id"] = ws_id
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)
        os.unlink(path)

    background_tasks.add_task(process_in_background, tmp_path, workspace_id)
    return {"message": "Processing started", "filename": file.filename}
```

### Exception Handlers
```python
from fastapi import Request
from fastapi.responses import JSONResponse
from openai import RateLimitError, APIConnectionError

@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    return JSONResponse(status_code=429, content={"error": "LLM rate limit exceeded. Try again later."})

@app.exception_handler(APIConnectionError)
async def connection_handler(request: Request, exc: APIConnectionError):
    return JSONResponse(status_code=503, content={"error": "LLM service unavailable"})

@app.exception_handler(Exception)
async def general_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": f"Internal error: {str(exc)}"})
```

### Path Params, Query Params, Request Body
```python
# Path parameter -- part of the URL
@app.get("/workspaces/{workspace_id}/documents/{doc_id}")
def get_document(workspace_id: str, doc_id: int):
    return {"workspace": workspace_id, "doc": doc_id}
# GET /workspaces/ws_123/documents/5

# Query parameter -- after ?
@app.get("/search")
def search(query: str, top_k: int = 4, threshold: float = 0.7):
    return {"query": query, "top_k": top_k}
# GET /search?query=RAG&top_k=5

# Request body -- JSON in POST body (use Pydantic model)
@app.post("/chat")
def chat(request: ChatRequest):   # ChatRequest is a Pydantic BaseModel
    return {"answer": "..."}
```

### Swagger UI
```
/docs   -- Swagger UI (interactive, test endpoints in browser)
/redoc  -- ReDoc (cleaner read-only docs)
/openapi.json -- raw OpenAPI schema

Access at: http://localhost:8000/docs
Run with:  uvicorn main:app --reload
```

---


> ---
> **Section Summary:** FastAPI is the standard for AI APIs -- async-first, Pydantic validation, auto Swagger docs. Use lifespan context manager to load embedding models and vector stores once at startup (not per request). Define Pydantic models for all request/response types with Field validators. Use Depends() for auth, settings, and shared resources. Background tasks for async document processing. StreamingResponse with text/event-stream for LLM streaming. Exception handlers for RateLimitError and APIConnectionError. File upload with UploadFile, save to tempfile, process in background. Access Swagger at /docs, ReDoc at /redoc.


---

## 19. Python for AI Engineers

### Async / Await
```python
import asyncio, httpx

# async def = coroutine, does not block the event loop
async def fetch(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)  # pause here, let others run
        return response.json()

# Run multiple coroutines in parallel
async def main():
    results = await asyncio.gather(
        fetch("https://api1.com/data"),
        fetch("https://api2.com/data"),
        fetch("https://api3.com/data"),
    )
    return results

asyncio.run(main())

# asyncio.gather vs asyncio.wait:
# gather: returns results in order, raises on first exception
# wait:   more control, can handle partial failures
```

### Generators
```python
from pathlib import Path

def load_documents_lazy(folder: str):
    """Load documents one at a time -- memory efficient for large datasets"""
    for path in Path(folder).glob("**/*.txt"):
        with open(path, encoding="utf-8") as f:
            yield {"path": str(path), "content": f.read()}

def chunk_generator(documents, chunk_size: int = 100):
    """Yield batches of documents"""
    batch = []
    for doc in documents:
        batch.append(doc)
        if len(batch) == chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch

# Process 10,000 docs without loading all into memory
for batch in chunk_generator(load_documents_lazy("./docs"), chunk_size=50):
    process_batch(batch)
```

### Decorators
```python
import time, functools

def timer(func):
    @functools.wraps(func)  # preserves function name and docstring
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter()-start:.3f}s")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (attempt + 1))  # exponential backoff
        return wrapper
    return decorator

@timer
@retry(max_attempts=3, delay=2.0)
def call_llm_api(query: str) -> str:
    response = client.chat.completions.create(...)
    return response.choices[0].message.content
```

### Pydantic v2
```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
from enum import Enum

class ModelName(str, Enum):
    GPT4  = "gpt-4"
    GPT4O = "gpt-4o"
    LLAMA = "llama-3.1-8b-instant"

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    model: ModelName = ModelName.GPT4
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    workspace_id: Optional[str] = None

    @field_validator('query')
    @classmethod
    def clean_query(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode='after')
    def validate_params(self) -> 'ChatRequest':
        if "o1" in self.model and self.temperature != 1.0:
            self.temperature = 1.0  # o1 models require temperature=1
        return self

req = ChatRequest(query="What is RAG?")
print(req.model_dump())          # dict
print(req.model_dump_json())     # JSON string
print(req.model_json_schema())   # JSON schema (used in Swagger)
```

### Settings from Environment
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str
    groq_api_key: str
    database_url: str = "sqlite:///./app.db"
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    debug: bool = False

    class Config:
        env_file = ".env"

@lru_cache()  # only reads .env once, cached for app lifetime
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)
```

### Useful Libraries
```
openai                --> OpenAI API (GPT-4, embeddings, DALL-E)
anthropic             --> Claude API
groq                  --> Groq fast inference (LLaMA, Mixtral)
langchain-core        --> core abstractions (LCEL, prompts, parsers)
langchain-openai      --> LangChain + OpenAI
langchain-groq        --> LangChain + Groq
langchain-community   --> community loaders, vectorstores
langgraph             --> multi-step agent workflows
llama-index-core      --> LlamaIndex core
chromadb              --> local vector database
faiss-cpu             --> fast similarity search
qdrant-client         --> Qdrant vector DB client
sentence-transformers --> local embedding models
tiktoken              --> OpenAI tokenizer (count tokens)
fastapi               --> async Python API framework
uvicorn               --> ASGI server
pydantic              --> data validation
pydantic-settings     --> settings from env vars
python-dotenv         --> load .env files
httpx                 --> async HTTP client
redis                 --> cache + message broker
supabase              --> Postgres + Auth + Storage
slowapi               --> rate limiting for FastAPI
```

---



> ---
> **Section Summary:** Essential Python for AI: async/await for non-blocking LLM calls (asyncio.gather for parallel), generators for memory-efficient large dataset processing, decorators for @timer and @retry logic on API calls, Pydantic v2 for request validation (field_validator, model_validator, Field with ge/le constraints), pydantic-settings for loading config from .env with @lru_cache. Key libraries: openai, groq, anthropic, langchain-core, langgraph, chromadb, sentence-transformers, tiktoken, fastapi, uvicorn, httpx, redis, supabase, slowapi.


---

## 20. Interview Prep - Based on Your Resume

> This section is built specifically for you, Sujit. Based on your resume, projects, and skills.

---

### Your Self Introduction

> Practice this until it feels natural. Keep it under 90 seconds.

"Hi, I am Sujit Sadalage, a final-year B.Tech student in Artificial Intelligence and Data Science from SITCOE, Yadrav, with a CGPA of 9.43.

I am a Python and AI developer focused on building real-world Generative AI and Agentic AI applications. My core work involves LLMs, RAG pipelines, LangChain, LangGraph, and FastAPI.

I have built three major projects - a full-stack Generative AI platform integrating 17+ AI models, a Multi-Agent AI Team Automation workflow using Groq and open-source LLMs with human-in-the-loop supervision, and an AI RAG Console - a multi-workspace document intelligence platform with semantic search, OCR ingestion, and Groq-powered inference.

I have also solved 450+ LeetCode problems, secured 1st place in a national-level Tech Quiz, and published a research paper on my GenAI platform.

I am looking for a role where I can apply my AI engineering skills to build production-grade intelligent systems."

---

### Questions They Will Ask About Your Projects

**Q: Tell me about your RAG Console project.**

A: I built a full-stack RAG platform with multi-workspace support. Users can upload PDFs, DOCX, Excel, and even images with OCR. The system chunks the documents, generates embeddings using HuggingFace all-MiniLM-L6-v2, and stores them in ChromaDB. When a user asks a question, I embed the query, do cosine similarity search, retrieve the top-k chunks, and pass them as context to Groq LLM for low-latency inference. I also added Supabase for authentication and workspace isolation so each user only sees their own documents. The frontend is built with Tailwind CSS and FastAPI serves the backend.

**Q: What was the hardest challenge in your RAG project?**

A: The trickiest part was workspace isolation in the vector DB. I had to make sure that when User A queries, they only get results from their own documents, not User B's. I solved this by storing workspace_id as metadata with every chunk and filtering by it during similarity search.

**Q: Explain your Multi-Agent workflow.**

A: I built a multi-agent system that acts like a full coding team. There is a Planner agent that breaks down the task, a Coder agent that writes the code, an Analyst agent that reviews it, and a Reporter agent that generates a PDF report and sends it via email. I used n8n for workflow orchestration, Groq for fast inference, and open-source LLMs like LLaMA 3.1 8B. I also added Human-in-the-Loop checkpoints where a human can approve or reject before the next step runs.

**Q: What is Human-in-the-Loop and why did you use it?**

A: HITL is a pattern where the AI pauses at critical decision points and waits for human approval before continuing. I used it in my multi-agent workflow because some actions like sending emails or generating final reports are irreversible. You do not want an AI to do those autonomously without a human sanity check.

**Q: Tell me about your GenAI platform.**

A: I built a full-stack platform that integrates 17+ AI models for text, image, and code generation. The backend is Node.js with Express, and I used GenKit to manage model integrations. I integrated OpenAI and Gemini APIs. The key design decision was a modular architecture - each AI tool is a separate module, so adding a new model requires minimal code changes. I also published a research paper on it.

---

### Core Technical Questions

**Q: What is RAG and why is it better than just using an LLM?**
A: RAG combines a retrieval system with an LLM. Instead of relying on the model's training data (which has a cutoff and can hallucinate), RAG fetches relevant documents from a knowledge base first, then passes them as context to the LLM. This gives accurate, up-to-date, and source-traceable answers.

**Q: What is the difference between LangChain and LangGraph?**
A: LangChain is for building single-task pipelines like a RAG chain or a simple chatbot. LangGraph is for multi-step agentic workflows where you need loops, branches, state management, and retry logic. LangGraph is built on top of LangChain. I used LangChain for the RAG pipeline in my RAG Console and LangGraph concepts for the multi-agent workflow.

**Q: How does ChromaDB store and retrieve data?**
A: ChromaDB stores text chunks along with their vector embeddings. The text and metadata go into a SQLite file, and the embeddings go into binary files for fast retrieval. When you query, it converts your query to a vector using the same embedding model, then does cosine similarity search to find the most similar stored chunks and returns the top-k results.

**Q: What is cosine similarity?**
A: Cosine similarity measures the angle between two vectors. If two vectors point in the same direction, similarity = 1 (identical meaning). If perpendicular, similarity = 0 (unrelated). It is used in vector search to find the most semantically similar document chunks to a query.

**Q: What is Groq and why did you use it?**
A: Groq is an inference API that runs open-source LLMs like LLaMA and Mixtral on custom LPU (Language Processing Unit) hardware. It is extremely fast - much lower latency than OpenAI for the same model size. I used it in my RAG Console for low-latency inference, which is important for a good user experience.

**Q: What is Supabase?**
A: Supabase is an open-source Firebase alternative built on PostgreSQL. It provides authentication, a real-time database, storage, and edge functions. I used it in my RAG Console for user authentication and storing workspace metadata.

**Q: What is the difference between synchronous and asynchronous in FastAPI?**
A: Synchronous (def) blocks the thread while waiting for a response. Asynchronous (async def) releases the thread while waiting, allowing other requests to be handled. For LLM API calls which are slow I/O operations, async is critical - it lets FastAPI handle many concurrent users without blocking.

**Q: What is Pydantic and why is it important?**
A: Pydantic is a Python library for data validation using type hints. FastAPI uses it to automatically validate request bodies, query parameters, and response models. If a user sends wrong data types, FastAPI returns a clear error automatically - no manual validation code needed.

**Q: What is the difference between top_p and temperature?**
A: Temperature scales the probability distribution of all tokens - high temperature = more uniform = more random. top_p (nucleus sampling) cuts off the token pool to only the top tokens whose cumulative probability reaches top_p. Both control creativity but work differently. In practice, you usually tune one or the other, not both.

**Q: What is fine-tuning vs RAG? When to use which?**
A: Fine-tuning trains the model on new data, changing its weights - good for teaching the model a new style, tone, or domain-specific knowledge that is static. RAG retrieves external documents at query time - good for dynamic, frequently updated knowledge bases. RAG is cheaper and more flexible. Fine-tuning is better when you need consistent behavior or style.

**Q: What is RLHF?**
A: Reinforcement Learning from Human Feedback. Humans rate model outputs (which response is better), and those ratings are used as reward signals to fine-tune the model using RL. This is how ChatGPT was trained to be helpful and safe.

**Q: What is LoRA?**
A: Low-Rank Adaptation. Instead of fine-tuning all model weights (expensive), LoRA adds small trainable matrices to specific layers. This reduces trainable parameters by 10,000x while achieving similar results. QLoRA adds quantization on top, making it possible to fine-tune large models on consumer GPUs.

**Q: What is chunking and why is chunk overlap important?**
A: Chunking splits large documents into smaller pieces to fit LLM input limits. Chunk overlap keeps some shared content between consecutive chunks to preserve context and prevent information loss at boundaries.

**Q: What is the difference between Stemming and Lemmatization?**
A: Stemming cuts words to their root using simple rules and may not produce a real word (studies --> studi). Lemmatization converts words to their proper dictionary base form using language rules (studies --> study). Lemmatization is more accurate but slower.

---

### Python and DSA Questions

**Q: What is the difference between a list and a generator in Python?**
A: A list stores all elements in memory at once. A generator yields elements one at a time - memory efficient for large datasets. In AI, generators are useful for processing large document batches without loading everything into RAM.

**Q: What is a decorator in Python?**
A: A decorator is a function that wraps another function to add behavior without modifying it. Example: @timer decorator to measure execution time, @app.get("/") in FastAPI to register a route.

**Q: What is async/await in Python?**
A: async def defines a coroutine. await pauses execution until an async operation completes, releasing the event loop to handle other tasks. Used in FastAPI for non-blocking LLM API calls.

**Q: What are *args and **kwargs?**
A: *args collects extra positional arguments as a tuple. **kwargs collects extra keyword arguments as a dict. Used to write flexible functions that accept variable numbers of arguments.

**Q: What is the difference between == and is in Python?**
A: == checks value equality. is checks identity (same object in memory). [1,2] == [1,2] is True, but [1,2] is [1,2] is False.

---

### Behavioral / HR Questions

**Q: Why AI engineering?**
A: I have always been drawn to building things that feel intelligent. When I started working with LLMs and RAG, I realized this is where software engineering and real-world impact intersect. Building systems that can understand documents, reason over data, and take actions autonomously - that is genuinely exciting to me.

**Q: What is your biggest achievement?**
A: Winning 1st place in the national-level Tech Quiz at Impetus 2k25 was great, but honestly the RAG Console project is what I am most proud of. It is a real production-grade system - multi-workspace, multimodal ingestion, semantic search, and low-latency inference. It solved a real problem and I built it end-to-end.

**Q: Where do you see yourself in 2 years?**
A: I want to be building production AI systems at scale - working on things like multi-agent architectures, fine-tuning pipelines, or AI infrastructure. I am also interested in contributing to open-source AI tooling.

**Q: How do you stay updated with AI?**
A: I follow papers on arXiv, watch releases from Anthropic, OpenAI, and Meta, and I build things. The best way to understand a new concept is to implement it. That is how I learned RAG, LangGraph, and MCP - by actually building with them.

---

### Quick-Fire Concepts - Know These Cold

```
Transformer         --> architecture behind all modern LLMs (attention-based)
Token               --> smallest unit of text the model processes
Embedding           --> numerical vector representation of text
Context Window      --> max tokens an LLM can process at once
Hallucination       --> LLM generating false/fabricated information
RAG                 --> retrieval + generation to ground LLM in real data
Fine-tuning         --> training pre-trained model on specific data
LoRA                --> efficient fine-tuning with low-rank matrices
RLHF                --> training with human feedback as reward signal
Agent               --> LLM + tools + memory + planning loop
ReAct               --> Reason + Act pattern for agents
HITL                --> Human-in-the-Loop, human approval in agent workflow
MCP                 --> standard protocol for LLMs to access external tools
LangChain           --> framework for LLM pipelines
LangGraph           --> framework for multi-step agent workflows
ChromaDB            --> local vector database
Groq                --> fast LLM inference API (LPU hardware)
Supabase            --> open-source Firebase alternative (PostgreSQL)
FastAPI             --> async Python API framework
Pydantic            --> Python data validation with type hints
Uvicorn             --> ASGI server for FastAPI
n8n                 --> no-code workflow automation
Flowise             --> no-code LangChain flow builder
Cosine Similarity   --> measure of similarity between two vectors
Chunking            --> splitting documents into smaller pieces for RAG
top_k / top_p       --> token sampling strategies for LLM output
Temperature         --> controls randomness of LLM output
Streaming           --> returning LLM response token by token in real time
Astra DB            --> cloud vector DB (Cassandra-based)
HuggingFace         --> open-source model hub and embeddings
GenKit              --> Google's framework for AI app development
```

---

*Good luck Sujit. You have built real things - talk about them confidently.*


> ---
> **Section Summary:** Your strongest talking points: RAG Console (multi-workspace, multimodal ingestion, ChromaDB + HuggingFace embeddings, Groq inference, Supabase auth, workspace isolation via metadata filter), Multi-Agent workflow (n8n orchestration, Groq + LLaMA, HITL checkpoints, PDF report + email automation), GenAI Platform (17+ models, modular architecture, research paper). Key differentiators: 9.43 CGPA, 450+ LeetCode, national Tech Quiz 1st place, published research paper. Practice your self-intro until it flows naturally in under 90 seconds.
