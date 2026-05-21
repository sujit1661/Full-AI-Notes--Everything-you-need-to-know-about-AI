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

## 1. Artificial Intelligence (AI).

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

Hi, I am Sujit Sadalage, a final-year student at Sharad Institute of Technology College of Engineering pursuing B.Tech in Artificial Intelligence and Data Science, with a CGPA of 9.43.

I am a Python, AI, and full-stack developer focused on building real-world Generative AI and Agentic AI applications. My core expertise includes LLMs, RAG pipelines, LangChain, LangGraph, and FastAPI. Alongside AI engineering, I also have strong full-stack web development experience using HTML, CSS, JavaScript, Bootstrap, Node.js, Express.js, and PostgreSQL Global Development Group PostgreSQL, which helps me build complete production-ready AI platforms end-to-end.

I have built three major projects: a full-stack Generative AI platform integrating 17+ AI models, a Multi-Agent AI Team Automation workflow using Groq and open-source LLMs with human-in-the-loop supervision, and an AI RAG Console—a multi-workspace document intelligence platform with semantic search, OCR ingestion, and Groq-powered inference.

I have also solved 450+ problems on LeetCode, secured 1st place in a national-level tech quiz, and published a research paper on my GenAI platform.

I am looking for an opportunity where I can apply both my AI engineering and full-stack development skills to build production-grade intelligent systems.

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
# AI / Backend Interview Q&A

---

## Q: How would you scale an AI application to handle 10,000+ users?

**A:**  
To scale an AI application for 10,000+ concurrent users, I would scale it in multiple layers.

### 1. Identify bottlenecks
First, I would profile the entire pipeline to identify where latency is coming from:
- API layer
- Database
- Vector search
- LLM inference
- Network calls

I believe scaling without measuring only increases infrastructure cost.

### 2. Use asynchronous processing
AI applications mostly perform I/O operations like:
- Calling LLM APIs
- Querying vector databases
- Fetching metadata

So I would use async APIs to process multiple requests concurrently instead of sequentially.

### 3. Implement caching
I would cache:
- Frequently asked queries
- Embeddings
- Retrieval results
- Final model responses

This avoids repeated expensive inference calls.

### 4. Horizontal scaling
Instead of upgrading one server, I would deploy multiple API instances behind a load balancer to distribute traffic.

### 5. Move heavy tasks to background workers
Tasks like:
- OCR
- PDF parsing
- Embedding generation
- Indexing

would run asynchronously in worker queues instead of blocking user requests.

### 6. Model optimization
I would:
- Use smaller models for simple queries
- Use larger models only for complex reasoning
- Batch requests when possible

### 7. Monitoring and auto scaling
I would continuously monitor:
- Latency
- CPU
- Memory
- Queue size
- Error rates

and automatically scale infrastructure during traffic spikes.

---

## Q: What if your RAG pipeline takes 10 seconds per query?

**A:**  
I would not directly scale a slow system. First, I would identify where the 10 seconds are being spent.

### Step 1: Profile each stage
I would measure latency for:
- Query preprocessing
- Embedding generation
- Vector retrieval
- Reranking
- LLM generation

### Step 2: Optimize retrieval
I would:
- Use metadata filtering
- Improve vector indexing
- Reduce search space
- Use hybrid search

This improves retrieval speed.

### Step 3: Optimize chunking
Chunk size directly affects:
- Retrieval quality
- Token usage
- Inference latency

I would tune chunk size for optimal context.

### Step 4: Optimize prompts
I would remove:
- Duplicate context
- Unnecessary instructions
- Irrelevant chunks

This reduces token usage.

### Step 5: Model routing
Simple queries would use smaller models, while complex reasoning uses larger models.

### Step 6: Implement caching
Repeated queries should return cached responses instead of re-running the full pipeline.

My principle is:
**Optimize first, then scale.**

---

## Q: How would you reduce AI inference cost?

**A:**  
AI inference cost mainly comes from:
- Token usage
- Model size
- Repeated requests

To reduce cost:

### 1. Token optimization
I would send only relevant context to the model.

### 2. Model routing
Simple tasks use smaller models.
Complex tasks use larger models.

### 3. Caching
Repeated queries should use cached outputs.

### 4. Batch processing
Multiple requests can be processed in batches.

### 5. Incremental embeddings
Only new or updated documents should be embedded.

My goal is:
**Maximum quality with minimum token cost.**

---

## Q: What if your LLM provider fails?

**A:**  
External AI providers can fail, so I always design for fault tolerance.

### 1. Health checks
Continuously monitor provider availability.

### 2. Retry mechanism
Use exponential backoff:
- 1 second
- 2 seconds
- 4 seconds

### 3. Circuit breaker
If the provider keeps failing, temporarily stop sending requests.

### 4. Model fallback
If the primary provider fails, automatically switch to a backup provider.

### 5. Graceful degradation
If large models are unavailable, temporarily use smaller models.

My principle:
**Never depend on a single provider in production.**

---

## Q: How would you handle 1000 PDF uploads at once?

**A:**  
Heavy document processing should never block real-time APIs.

### Step 1: Immediate upload response
When a user uploads a PDF, I immediately return a job ID.

### Step 2: Queue processing
The document goes into a task queue.

### Step 3: Worker processing
Background workers perform:
- OCR
- Chunking
- Embedding generation
- Vector indexing

### Step 4: Parallel workers
Multiple workers process multiple PDFs simultaneously.

### Step 5: Status tracking
Users can check:
- Processing
- Completed
- Failed

This keeps the API responsive.

---

## Q: How would you prevent hallucinations in RAG?

**A:**  
Hallucination happens when the model guesses instead of using verified context.

To reduce hallucinations:

### 1. Retrieval grounding
The model should answer only using retrieved documents.

### 2. Prompt constraints
I use prompts like:

*"Answer only from the provided context. If information is not available, say I don't know."*

### 3. Confidence thresholds
Low-confidence responses trigger fallback responses.

### 4. Reranking
Only highly relevant chunks are sent to the model.

### 5. Citations
Show users which documents were used.

My belief:
**Better retrieval reduces hallucination more than bigger models.**

---

## Q: What if traffic suddenly spikes 100×?

**A:**  
I would handle traffic spikes in multiple layers.

### 1. Auto scaling
Automatically add more compute resources.

### 2. Load balancing
Distribute traffic across multiple servers.

### 3. Queue buffering
Overflow traffic gets queued.

### 4. Rate limiting
Prevent abuse.

### 5. Graceful degradation
Temporarily disable expensive features like:
- Reranking
- Large models

and switch to lighter models.

### 6. Cache hot queries
Frequently requested queries should come from cache.

My priority during spikes:
**Keep the system available first, optimize later.**




## Q: How would you reduce inference latency in an AI application?

**A:**  
Inference latency directly impacts user experience.

I would reduce it using multiple strategies.

### 1. Model routing
Not every request needs the largest model.

Simple tasks use smaller models.
Complex reasoning uses larger models.

### 2. Token optimization
I would remove unnecessary context from prompts.

Fewer tokens means faster inference.

### 3. Caching
Repeated prompts should return cached responses.

### 4. Batch inference
Multiple requests can be processed together.

### 5. Warm model instances
Keep models preloaded in memory to avoid cold starts.

### 6. Deploy closer to users
Use multi-region deployment to reduce network latency.

---

## Q: How would you handle sudden traffic spikes?

**A:**  
During sudden traffic spikes, my first goal is availability.

### 1. Auto scaling
Automatically add more servers.

### 2. Load balancing
Distribute traffic evenly.

### 3. Queue overflow
Excess requests should go into queues.

### 4. Rate limiting
Prevent abuse.

### 5. Graceful degradation
Temporarily disable expensive features like:
- Reranking
- Large models

and use lighter alternatives.

### 6. Cache hot queries
Frequently requested responses should come from cache.

My principle:
**Keep the system available first.**

---

## Q: How would you reduce inference cost in production?

**A:**  
Inference cost mainly comes from:
- Token usage
- Large models
- Repeated requests

I would optimize all three.

### 1. Token optimization
Send only relevant context.

### 2. Model routing
Use smaller models when possible.

### 3. Semantic caching
Avoid repeated inference.

### 4. Batch processing
Process multiple requests together.

### 5. Incremental embeddings
Only process new documents.

My goal:
**Maximum quality with minimum inference cost.**

---

## Q: How would you handle provider failures?

**A:**  
AI providers can fail, so I always design for fault tolerance.

### 1. Health checks
Continuously monitor provider availability.

### 2. Retry mechanism
Use exponential backoff:
- 1 sec
- 2 sec
- 4 sec

### 3. Circuit breaker
Stop sending traffic to unhealthy providers.

### 4. Model fallback
Switch to secondary providers.

### 5. Graceful degradation
If premium models fail, use smaller models.

My principle:
**Never depend on one provider.**

---

## Q: How would you optimize a vector database as data grows?

**A:**  
As vector data grows, retrieval can become slow.

To optimize:

### 1. Sharding
Split vectors across multiple partitions.

### 2. Metadata filtering
Search only relevant subsets.

Example:
- workspace
- department
- document type

### 3. Index optimization
Use optimized ANN indexes.

### 4. Namespace isolation
Separate tenants.

### 5. Archive cold data
Move rarely used vectors to cheaper storage.

This keeps retrieval fast at scale.



## Q: How would you prevent one user from consuming all resources?

**A:**  
In multi-user AI systems, resource isolation is critical.

I would implement:

### 1. Rate limiting
Limit requests per minute.

### 2. Quotas
Limit:
- Tokens
- Uploads
- Storage

### 3. Queue prioritization
Premium users may get priority.

### 4. Tenant isolation
Separate resources per tenant.

### 5. Abuse detection
Block suspicious activity.

This prevents noisy-neighbor problems.

---

## Q: How would you scale document ingestion?

**A:**  
Document ingestion includes:
- OCR
- Parsing
- Chunking
- Embedding

To scale:

### 1. Parallel workers
Multiple workers process files simultaneously.

### 2. Batch embeddings
Embed multiple chunks together.

### 3. Incremental processing
Only process new files.

### 4. Deduplication
Avoid embedding duplicate documents.

### 5. GPU workers
Use GPUs for heavy OCR or embeddings.

This allows high-throughput ingestion.

---

## Q: What if one component becomes slow?

**A:**  
I would first isolate the bottleneck.

### Step 1: Monitoring
Measure:
- API latency
- DB latency
- Queue latency
- Model latency

### Step 2: Scale only the bottleneck
Instead of scaling the whole system.

Example:
If vector search is slow:
Scale vector nodes only.

If OCR is slow:
Scale worker nodes only.

My principle:
**Scale the bottleneck, not everything.**

---

## Q: How would you monitor AI applications in production?

**A:**  
I monitor both system metrics and AI metrics.

### System metrics
- CPU
- Memory
- Disk
- Network

### Application metrics
- API latency
- Error rates
- Queue size

### AI metrics
- Token usage
- Retrieval latency
- Cache hit rate
- Model failures
- Hallucination rate

This helps detect issues before users notice.

---

## Q: How would you handle cold starts?

**A:**  
Cold starts happen when models are loaded on demand.

To reduce them:

### 1. Warm model instances
Keep models loaded in memory.

### 2. Predictive scaling
Scale before peak traffic.

### 3. Background warmup
Periodically ping instances.

This reduces first-response latency.

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


---

## 21. Python Basics - Tricky Interview Questions

> These are the questions that trip up even experienced developers. Interviewers love these because they reveal whether you truly understand Python or just use it.

---

### Q1. What is the difference between `is` and `==`?

`==` checks value equality — it compares what two objects contain. `is` checks identity — it checks whether two variables point to the exact same object in memory (same address via `id()`).

Most of the time you want `==`. Use `is` only when checking against singletons like `None`, `True`, or `False` — because there is only one `None` object in Python, so `is None` is always safe and correct.

```python
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)   # True  -- same values
print(a is b)   # False -- two different list objects in memory

c = a           # c points to the SAME object as a
print(a is c)   # True
```

---

### Q2. What is mutable vs immutable in Python?

Mutable objects can be changed after creation. Immutable objects cannot — any "change" creates a new object instead.

Immutable types: `int`, `float`, `str`, `tuple`, `bool`, `frozenset`. Mutable types: `list`, `dict`, `set`, `bytearray`.

This matters because when you pass a mutable object to a function, the function can modify the original. When you pass an immutable object, the function cannot change the original — it can only create a new one.

Tricky edge case: a tuple is immutable, but if it contains a list inside, that list can still be changed because the list itself is mutable.

```python
t = (1, [2, 3])
t[1].append(4)   # works! t is now (1, [2, 3, 4])
# The tuple didn't change (still holds same list reference), but the list inside did
```

---

### Q3. What is the difference between shallow copy and deep copy?

A shallow copy creates a new outer object but does not copy nested objects — it just copies references to them. So the outer container is new, but the inner objects are still shared with the original.

A deep copy creates a completely independent copy at every level — new outer object, new inner objects, all the way down. Changes to the deep copy never affect the original.

Use `copy.copy()` for shallow, `copy.deepcopy()` for deep. In practice, deep copy is what most people actually want when they say "I want a copy that won't affect the original."

```python
import copy
original = [[1, 2], [3, 4]]

shallow = copy.copy(original)
deep    = copy.deepcopy(original)

original[0][0] = 99
print(shallow[0][0])  # 99 -- shallow copy shares the inner list
print(deep[0][0])     # 1  -- deep copy is fully independent
```

---

### Q4. What are `*args` and `**kwargs`?

`*args` lets a function accept any number of positional arguments. Python collects them into a tuple. `**kwargs` lets a function accept any number of keyword arguments. Python collects them into a dictionary.

They are useful when you want to write flexible functions that can handle a variable number of inputs — like a wrapper function that passes arguments through to another function without knowing what those arguments are.

```python
def func(*args, **kwargs):
    print(args)    # (1, 2, 3)  -- tuple
    print(kwargs)  # {'name': 'Sujit', 'age': 21}  -- dict

func(1, 2, 3, name="Sujit", age=21)
```

---

### Q5. What is a Python decorator and how does it work?

A decorator is a function that takes another function as input, wraps it with extra behavior, and returns the wrapped version. The `@decorator` syntax is just shorthand for `func = decorator(func)`.

Decorators are used for cross-cutting concerns — things you want to add to many functions without repeating code, like logging, timing, authentication checks, retry logic, or caching. FastAPI uses `@app.get("/")` as a decorator to register route handlers.

```python
def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.3f}s")
        return result
    return wrapper

@timer
def slow_function():
    import time; time.sleep(1)

slow_function()  # prints: slow_function took 1.001s
```

---

### Q6. What is a generator? How is it different from a list?

A generator is a function that uses `yield` instead of `return`. It produces values one at a time on demand (lazy evaluation) instead of computing and storing all values upfront.

The key difference is memory. A list with 1 million items stores all 1 million items in RAM at once. A generator stores only the current item — it computes the next one only when you ask for it with `next()`. This makes generators essential for processing large datasets, reading big files line by line, or streaming data.

```python
# List -- all 1 million values in memory immediately
squares_list = [x**2 for x in range(1_000_000)]  # ~8MB RAM

# Generator -- only one value in memory at a time
def squares_gen(n):
    for x in range(n):
        yield x**2          # pauses here, resumes on next()

gen = squares_gen(1_000_000)  # ~200 bytes RAM
print(next(gen))  # 0
print(next(gen))  # 1
```

---

### Q7. What is the difference between `list`, `tuple`, `set`, and `dict`?

These are Python's four main built-in collection types, each with a different purpose.

`list` is ordered and mutable — use it when you need a sequence you can modify. `tuple` is ordered and immutable — use it for fixed data like coordinates or function return values (also faster than list). `set` is unordered with no duplicates — use it for membership testing or removing duplicates. `dict` is a key-value mapping — use it for lookups by name.

```
list   -- ordered, mutable, allows duplicates     [1, 2, 2, 3]
tuple  -- ordered, immutable, allows duplicates   (1, 2, 2, 3)
set    -- unordered, NO duplicates, mutable       {1, 2, 3}
dict   -- key-value pairs, ordered (Python 3.7+)  {"a": 1, "b": 2}
```

---

### Q8. What is a lambda function? When should you use it?

A lambda is an anonymous, single-expression function defined inline. It is syntactic sugar for a simple function you only need once — typically as an argument to `sorted()`, `map()`, or `filter()`.

You should use lambda only for simple, short operations. If the logic is more than one expression, write a proper named function — it is more readable and easier to test and debug.

```python
# Instead of defining a named function just for sorting
students = [("Sujit", 9.43), ("Raj", 8.5), ("Priya", 9.1)]
students.sort(key=lambda s: s[1], reverse=True)
# lambda s: s[1] means "for each student tuple, use index 1 (the score) as the sort key"
```

---

### Q9. What is list comprehension? When should you NOT use it?

List comprehension is a concise way to create a list by applying an expression to each item in an iterable, optionally filtering with a condition. It is faster than an equivalent `for` loop because it is optimized at the C level in CPython.

You should NOT use it when the logic is complex enough that it becomes hard to read on one line, when you have deeply nested loops, or when you have side effects (like printing or writing to a file) — a regular `for` loop is clearer in those cases. Readability always wins.

```python
# Good use -- simple and readable
squares = [x**2 for x in range(10) if x % 2 == 0]

# Bad use -- too complex, use a regular loop instead
result = [process(transform(x)) for x in data if condition1(x) and condition2(x)]
```

---

### Q10. What is the difference between `append`, `extend`, and `insert`?

All three add elements to a list but in different ways. `append` adds its argument as a single element — even if you pass a list, it adds the whole list as one nested element. `extend` iterates over its argument and adds each item individually. `insert` adds one element at a specific index position.

```python
lst = [1, 2, 3]
lst.append([4, 5])   # [1, 2, 3, [4, 5]]  -- list added as ONE element
lst = [1, 2, 3]
lst.extend([4, 5])   # [1, 2, 3, 4, 5]    -- each item added separately
lst = [1, 2, 3]
lst.insert(1, 99)    # [1, 99, 2, 3]       -- 99 inserted at index 1
```

---

### Q11. What is the GIL (Global Interpreter Lock)?

The GIL is a mutex (lock) in CPython — the standard Python interpreter — that allows only one thread to execute Python bytecode at a time, even on a multi-core machine.

It exists because CPython's memory management (reference counting for garbage collection) is not thread-safe. The GIL protects it from race conditions.

The practical impact: Python threads cannot achieve true parallelism for CPU-bound tasks. If you have 4 CPU cores and 4 threads doing heavy computation, only one runs at a time — you get no speedup. For I/O-bound tasks (waiting for network, file reads), the GIL is released during the wait, so threading works fine. For CPU-bound parallelism, use `multiprocessing` — each process has its own GIL.

---

### Q12. What is the difference between `__str__` and `__repr__`?

Both return a string representation of an object, but they serve different audiences. `__str__` is for end users — it should be readable and human-friendly. `__repr__` is for developers — it should be unambiguous and ideally show enough information to recreate the object.

When you call `print(obj)`, Python uses `__str__`. When you type an object in the Python shell or use `repr(obj)`, Python uses `__repr__`. If `__str__` is not defined, Python falls back to `__repr__`.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):   # for print() -- user-friendly
        return f"{self.name}, age {self.age}"

    def __repr__(self):  # for debugging -- developer-friendly
        return f"Person(name='{self.name}', age={self.age})"

p = Person("Sujit", 21)
print(p)    # Sujit, age 21
repr(p)     # Person(name='Sujit', age=21)
```

---

### Q13. What are dunder (magic) methods in Python?

Dunder methods (double underscore methods) are special methods that Python calls automatically in response to certain operations. They let you define how your objects behave with built-in operations like `+`, `len()`, `[]`, `in`, `with`, and `print()`.

This is how Python achieves operator overloading — you define `__add__` and suddenly `obj1 + obj2` works for your custom class. They are the foundation of Python's data model.

```
__init__     called when object is created
__str__      called by print() and str()
__repr__     called by repr() and in the shell
__len__      called by len(obj)
__getitem__  called by obj[key]
__eq__       called by obj1 == obj2
__add__      called by obj1 + obj2
__call__     called by obj() -- makes object callable like a function
__enter__    called at start of with block
__exit__     called at end of with block (even if exception occurred)
```

---

### Q14. What is the difference between `@staticmethod`, `@classmethod`, and instance method?

An instance method receives `self` as the first argument — it has access to the specific instance and can read/modify its attributes. This is the most common type.

A class method receives `cls` (the class itself) as the first argument instead of an instance. It can access and modify class-level state. Most commonly used as alternative constructors — for example, `datetime.fromisoformat("2024-01-01")` is a class method that creates a `datetime` object from a string.

A static method receives neither `self` nor `cls`. It is just a regular function that lives inside the class namespace for organizational purposes. It has no access to instance or class state.

```python
class Temperature:
    unit = "Celsius"

    def __init__(self, value):
        self.value = value

    def describe(self):                    # instance method -- needs self
        return f"{self.value}° {Temperature.unit}"

    @classmethod
    def from_fahrenheit(cls, f):           # class method -- alternative constructor
        return cls((f - 32) * 5 / 9)

    @staticmethod
    def is_valid(value):                   # static method -- utility, no self/cls needed
        return -273.15 <= value <= 1e6

t = Temperature.from_fahrenheit(212)      # creates Temperature(100)
print(Temperature.is_valid(-300))         # False
```

---

### Q15. What is the difference between `pass`, `continue`, and `break`?

`pass` is a no-op — it does absolutely nothing. It is used as a placeholder where Python syntax requires a statement but you have nothing to put there yet, like an empty function body or an empty class.

`continue` skips the rest of the current loop iteration and jumps to the next one. The loop itself keeps running.

`break` exits the loop entirely. No more iterations happen.

```python
for i in range(6):
    if i == 1:
        pass      # does nothing, i=1 still prints
    if i == 3:
        continue  # skip printing 3, jump to i=4
    if i == 5:
        break     # stop the loop, never reach i=5
    print(i)
# Output: 0 1 2 4
```

---

### Q16. What is exception handling? What is the role of `else` and `finally`?

Exception handling lets you catch and respond to runtime errors gracefully instead of crashing. The `try` block contains code that might fail. `except` catches specific errors. `else` runs only if no exception occurred in the `try` block. `finally` always runs regardless — used for cleanup like closing files or database connections.

The `else` block is often overlooked but important — it separates "code that might fail" from "code that runs on success," making the intent clearer.

```python
try:
    result = 10 / int(input("Enter a number: "))
except ZeroDivisionError:
    print("Cannot divide by zero")
except ValueError:
    print("Please enter a valid number")
else:
    print(f"Result: {result}")   # only runs if no exception
finally:
    print("Done")                # always runs -- good for cleanup
```

---

### Q17. What is the difference between `range` and `xrange`?

In Python 2, `range()` returned a full list in memory, while `xrange()` returned a lazy iterator that generated numbers one at a time. For large ranges, `xrange` was much more memory-efficient.

In Python 3, `xrange` was removed and `range()` itself became lazy — it behaves like Python 2's `xrange`. So `range(1_000_000)` in Python 3 uses almost no memory. This is why you can write `for i in range(1_000_000)` without worrying about memory.

---

### Q18. What is `zip`, `map`, and `filter`?

`zip` combines multiple iterables element by element into tuples. It stops at the shortest iterable. Useful for iterating over two lists in parallel.

`map` applies a function to every element of an iterable and returns an iterator of results. It is a functional alternative to a list comprehension.

`filter` keeps only the elements for which a function returns `True`. Both `map` and `filter` return lazy iterators in Python 3 — wrap with `list()` to get a list.

```python
names  = ["Sujit", "Raj", "Priya"]
scores = [9.43, 8.5, 9.1]

# zip -- pair them up
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# map -- apply function to each element
upper_names = list(map(str.upper, names))   # ['SUJIT', 'RAJ', 'PRIYA']

# filter -- keep only elements that pass the condition
toppers = list(filter(lambda s: s > 9.0, scores))  # [9.43, 9.1]
```

---

### Q19. What is a context manager? How does `with` work?

A context manager is an object that defines setup and teardown behavior using `__enter__` and `__exit__` methods. The `with` statement calls `__enter__` at the start and guarantees `__exit__` is called at the end — even if an exception occurs.

The most common use is resource management: opening files, database connections, or locks. Without `with`, you risk forgetting to close a file or release a lock if an exception happens midway.

```python
# Without with -- risky, file may not close if exception occurs
f = open("file.txt")
data = f.read()
f.close()   # what if read() raised an exception? close() never called!

# With context manager -- __exit__ always called, file always closed
with open("file.txt") as f:
    data = f.read()
# file is closed here automatically, even if an exception occurred
```

---

### Q20. What is the difference between `global` and `nonlocal`?

By default, a variable assigned inside a function is local to that function. `global` tells Python that a variable name refers to the module-level (global) variable, not a new local one. `nonlocal` tells Python that a variable refers to the nearest enclosing function's scope (not global, not local).

`nonlocal` is used in nested functions (closures) when the inner function needs to modify a variable from the outer function.

```python
x = 10

def outer():
    y = 20
    def inner():
        global x     # x refers to the module-level x = 10
        nonlocal y   # y refers to outer()'s y = 20
        x = 99
        y = 99
    inner()
    print(y)  # 99 -- nonlocal let inner() modify outer's y

outer()
print(x)  # 99 -- global let inner() modify module-level x
```

---

### Q21. What is the difference between `/`, `//`, and `%`?

`/` is true division — always returns a float, even if both operands are integers. `//` is floor division — returns the largest integer less than or equal to the result (rounds toward negative infinity, not toward zero). `%` is modulo — returns the remainder after floor division.

The tricky part is negative numbers. `-7 // 2` gives `-4` (not `-3`) because floor division rounds toward negative infinity. Python's modulo always returns a non-negative result when the divisor is positive.

```python
10 / 3    # 3.3333...  -- always float
10 // 3   # 3          -- floor division
10 % 3    # 1          -- remainder
-7 // 2   # -4         -- rounds DOWN toward -infinity (not -3!)
-7 % 2    # 1          -- always non-negative when divisor is positive
```

---

### Q22. What is the difference between `None`, `False`, `0`, `""`, and `[]`?

All of these are "falsy" — they evaluate to `False` in a boolean context like an `if` statement. But they are not equal to each other (except `0 == False`, which is `True` because `bool` is a subclass of `int` in Python).

`None` means "no value" or "absence of a value." `False` is a boolean. `0` is an integer. `""` is an empty string. `[]` is an empty list. They represent different things and should not be confused even though they all behave like `False` in conditions.

```python
# All falsy -- these if-blocks won't execute
if None: ...
if False: ...
if 0: ...
if "": ...
if []: ...

# But they are NOT equal
None == False   # False
None == 0       # False
0 == False      # True  -- because bool is a subclass of int
```

---

### Q23. What is the difference between `sort()` and `sorted()`?

`sort()` is a list method that sorts the list in place and returns `None`. It modifies the original list. `sorted()` is a built-in function that returns a new sorted list and leaves the original unchanged. `sorted()` works on any iterable (tuples, sets, strings), not just lists.

Use `sort()` when you want to sort the list itself and don't need the original order. Use `sorted()` when you need to keep the original or when working with non-list iterables.

```python
lst = [3, 1, 4, 1, 5]
lst.sort()                        # modifies lst in place, returns None
new = sorted([3, 1, 4, 1, 5])    # returns new list, original unchanged

# Both support key and reverse
sorted(lst, key=lambda x: -x)    # sort descending
sorted(students, key=lambda s: s[1])  # sort by second element
```

---

### Q24. What is a closure in Python?

A closure is a function that remembers and has access to variables from its enclosing scope even after that outer function has finished executing. The inner function "closes over" the variables from the outer scope.

Closures are used to create function factories (functions that create other functions), to implement decorators, and to maintain state without using a class.

```python
def make_multiplier(n):
    def multiplier(x):
        return x * n   # n is from the outer scope -- "closed over"
    return multiplier  # returns the inner function

double = make_multiplier(2)  # outer() finished, but n=2 is remembered
triple = make_multiplier(3)

print(double(5))  # 10 -- n=2 is still accessible
print(triple(5))  # 15 -- n=3 is still accessible
```

---

### Q25. What is the difference between `__init__` and `__new__`?

`__new__` is called first — it creates and returns the new instance of the class. It is a static method that receives the class as its first argument. `__init__` is called second — it receives the already-created instance (`self`) and initializes its attributes.

In normal usage you only define `__init__`. You override `__new__` only in advanced cases like implementing the Singleton pattern (to control how many instances are created) or when subclassing immutable types like `int` or `str`.

```python
class MyClass:
    def __new__(cls, value):
        print(f"__new__: creating instance")
        instance = super().__new__(cls)  # actually creates the object
        return instance

    def __init__(self, value):
        print(f"__init__: initializing with {value}")
        self.value = value

obj = MyClass(42)
# __new__: creating instance
# __init__: initializing with 42
```

---

### Q26. What is `@property` and why use it instead of a regular attribute?

`@property` lets you define a method that is accessed like an attribute — no parentheses needed. It is used to add computed attributes (values calculated from other attributes) and to add validation when setting a value, without changing the public interface of the class.

The benefit is encapsulation. You can start with a plain attribute, and later add validation or computation by converting it to a property — all existing code that accesses `obj.radius` continues to work unchanged.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):           # getter -- accessed as c.radius (no parentheses)
        return self._radius

    @radius.setter
    def radius(self, value):    # setter -- called when c.radius = 5
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property
    def area(self):             # computed attribute -- no setter needed
        return 3.14 * self._radius ** 2

c = Circle(5)
c.radius = 10   # calls setter with validation
print(c.area)   # 314.0 -- computed on access, looks like an attribute
```

---

### Q27. What is the difference between `isinstance()` and `type()`?

`type(obj)` returns the exact class of an object. It does not consider inheritance — `type(dog) == Animal` is `False` even if `Dog` inherits from `Animal`.

`isinstance(obj, cls)` checks if an object is an instance of a class or any of its subclasses. It respects the inheritance chain. This is almost always what you want when doing type checking.

Always prefer `isinstance` because it works correctly with inheritance and is more Pythonic (duck typing friendly).

```python
class Animal: pass
class Dog(Animal): pass

d = Dog()
type(d) == Dog      # True
type(d) == Animal   # False -- type() is exact match only

isinstance(d, Dog)    # True
isinstance(d, Animal) # True -- isinstance checks the full inheritance chain
```

---

### Q28. What is `enumerate` and why use it instead of `range(len(...))`?

`enumerate` wraps an iterable and yields `(index, value)` pairs. It is the Pythonic way to loop over a sequence when you need both the index and the value.

`range(len(lst))` is considered un-Pythonic because it is indirect — you get the index and then have to use it to access the value. `enumerate` is cleaner, more readable, and works on any iterable (not just lists).

```python
fruits = ["apple", "banana", "cherry"]

# Un-Pythonic
for i in range(len(fruits)):
    print(i, fruits[i])

# Pythonic -- cleaner and works on any iterable
for i, fruit in enumerate(fruits):
    print(i, fruit)

# Start counting from 1 instead of 0
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}. {fruit}")   # 1. apple, 2. banana, 3. cherry
```

---

### Q1. What is the difference between `is` and `==`?
```python
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)   # True  -- same value
print(a is b)   # False -- different objects in memory

c = a
print(a is c)   # True  -- same object (same reference)
```
- `==` checks value equality
- `is` checks identity (same memory address via `id()`)

---

### Q2. What is mutable vs immutable in Python?
```python
# Immutable -- cannot be changed after creation
# int, float, str, tuple, bool, frozenset
x = "hello"
x[0] = "H"  # TypeError! strings are immutable

# Mutable -- can be changed in place
# list, dict, set, bytearray
lst = [1, 2, 3]
lst[0] = 99  # works fine
```
> Trick: Tuples are immutable but if they contain a mutable object (like a list), that inner object CAN be changed.
```python
t = (1, [2, 3])
t[1].append(4)  # works! t is now (1, [2, 3, 4])
```

---

### Q3. What is the difference between `deepcopy` and `copy`?
```python
import copy

original = [[1, 2], [3, 4]]

shallow = copy.copy(original)       # new outer list, same inner lists
deep    = copy.deepcopy(original)   # completely new object at every level

original[0][0] = 99
print(shallow[0][0])  # 99  -- shallow copy shares inner objects
print(deep[0][0])     # 1   -- deep copy is fully independent
```

---

### Q4. What are *args and **kwargs?
```python
def func(*args, **kwargs):
    print(args)    # tuple of positional args
    print(kwargs)  # dict of keyword args

func(1, 2, 3, name="Sujit", age=21)
# args   = (1, 2, 3)
# kwargs = {'name': 'Sujit', 'age': 21}
```

---

### Q5. What is a Python decorator?
```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function")
        result = func(*args, **kwargs)
        print("After function")
        return result
    return wrapper

@my_decorator
def greet(name):
    print(f"Hello {name}")

greet("Sujit")
# Before function
# Hello Sujit
# After function
```
> A decorator is a function that takes another function and extends its behavior without modifying it.

---

### Q6. What is a generator and how is it different from a list?
```python
# List -- stores all values in memory at once
squares_list = [x**2 for x in range(1000000)]  # uses lots of RAM

# Generator -- yields one value at a time, lazy evaluation
def squares_gen(n):
    for x in range(n):
        yield x**2

gen = squares_gen(1000000)  # uses almost no RAM
print(next(gen))  # 0
print(next(gen))  # 1

# Generator expression (like list comprehension but with ())
gen = (x**2 for x in range(10))
```
> Use generators when dealing with large data you don't need all at once.

---

### Q7. What is the difference between `list`, `tuple`, `set`, and `dict`?
```
list   -- ordered, mutable, allows duplicates          [1, 2, 2, 3]
tuple  -- ordered, immutable, allows duplicates        (1, 2, 2, 3)
set    -- unordered, mutable, NO duplicates            {1, 2, 3}
dict   -- key-value pairs, ordered (Python 3.7+)       {"a": 1, "b": 2}
```

---

### Q8. What is a lambda function?
```python
# Regular function
def add(x, y):
    return x + y

# Lambda -- anonymous one-liner function
add = lambda x, y: x + y
print(add(3, 4))  # 7

# Common use: sorting
students = [("Sujit", 9.43), ("Raj", 8.5), ("Priya", 9.1)]
students.sort(key=lambda s: s[1], reverse=True)
```

---

### Q9. What is list comprehension and when should you NOT use it?
```python
# List comprehension -- concise, fast
squares = [x**2 for x in range(10) if x % 2 == 0]

# Dict comprehension
word_lengths = {word: len(word) for word in ["hello", "world"]}

# Set comprehension
unique_squares = {x**2 for x in [-2, -1, 0, 1, 2]}

# When NOT to use: complex logic, multiple nested loops, side effects
# If it's hard to read in one line, use a regular for loop
```

---

### Q10. What is the difference between `append`, `extend`, and `insert`?
```python
lst = [1, 2, 3]

lst.append([4, 5])    # adds as single element: [1, 2, 3, [4, 5]]
lst.extend([4, 5])    # adds each element:      [1, 2, 3, 4, 5]
lst.insert(1, 99)     # inserts at index 1:     [1, 99, 2, 3]
```

---

### Q11. What is the GIL (Global Interpreter Lock)?
The GIL is a mutex in CPython that allows only ONE thread to execute Python bytecode at a time.

```
Threading in Python:
  - I/O-bound tasks (file read, API calls): threading works fine
  - CPU-bound tasks (heavy computation): use multiprocessing instead

# For CPU-bound: use multiprocessing (bypasses GIL)
from multiprocessing import Pool
```
> The GIL exists to protect Python's memory management (reference counting) from race conditions.

---

### Q12. What is the difference between `__str__` and `__repr__`?
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name}, age {self.age}"   # for end users (print)

    def __repr__(self):
        return f"Person('{self.name}', {self.age})"  # for developers (debug)

p = Person("Sujit", 21)
print(str(p))   # Sujit, age 21
print(repr(p))  # Person('Sujit', 21)
```

---

### Q13. What are Python's dunder (magic) methods?
```python
__init__    # constructor
__str__     # string representation for users
__repr__    # string representation for developers
__len__     # len(obj)
__getitem__ # obj[key]
__setitem__ # obj[key] = value
__contains__# item in obj
__eq__      # obj1 == obj2
__lt__      # obj1 < obj2
__add__     # obj1 + obj2
__call__    # obj()  -- makes object callable
__enter__   # with statement start
__exit__    # with statement end
```

---

### Q14. What is `@staticmethod` vs `@classmethod` vs instance method?
```python
class MyClass:
    class_var = "I am class variable"

    def instance_method(self):
        # has access to self (instance) and class
        return self.class_var

    @classmethod
    def class_method(cls):
        # has access to cls (class), not instance
        # used for alternative constructors
        return cls.class_var

    @staticmethod
    def static_method():
        # no access to self or cls
        # just a regular function inside the class namespace
        return "I don't need self or cls"
```

---

### Q15. What is the difference between `pass`, `continue`, and `break`?
```python
for i in range(5):
    if i == 2:
        pass      # does nothing, placeholder
    if i == 3:
        continue  # skip this iteration, go to next
    if i == 4:
        break     # exit the loop entirely
    print(i)
# Output: 0 1 2
```

---

### Q16. What is exception handling in Python?
```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
except (TypeError, ValueError) as e:
    print(f"Type or Value error: {e}")
except Exception as e:
    print(f"Any other error: {e}")
else:
    print("No error occurred")   # runs only if no exception
finally:
    print("Always runs")         # cleanup code here
```

---

### Q17. What is the difference between `range` and `xrange`?
- In Python 2: `range()` returns a list, `xrange()` returns an iterator (lazy)
- In Python 3: `range()` IS lazy (like Python 2's xrange). `xrange` no longer exists.

---

### Q18. What is `zip`, `map`, and `filter`?
```python
# zip -- combine multiple iterables element by element
names = ["Sujit", "Raj"]
scores = [9.43, 8.5]
for name, score in zip(names, scores):
    print(name, score)

# map -- apply function to every element
squares = list(map(lambda x: x**2, [1, 2, 3, 4]))  # [1, 4, 9, 16]

# filter -- keep elements where function returns True
evens = list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4, 5]))  # [2, 4]
```

---

### Q19. What is a context manager and how does `with` work?
```python
# with statement calls __enter__ and __exit__ automatically
with open("file.txt", "r") as f:
    content = f.read()
# file is automatically closed even if an exception occurs

# Custom context manager
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Time: {time.time() - self.start:.2f}s")
        return False  # don't suppress exceptions

with Timer():
    # some code
    pass
```

---

### Q20. What is the difference between `global` and `nonlocal`?
```python
x = 10

def outer():
    y = 20

    def inner():
        global x    # refers to module-level x
        nonlocal y  # refers to enclosing function's y
        x = 99
        y = 99

    inner()
    print(y)  # 99 -- changed by nonlocal

outer()
print(x)  # 99 -- changed by global
```

---

### Q21. What is the difference between `//`, `/`, and `%`?
```python
10 / 3    # 3.333...  -- true division (always float)
10 // 3   # 3         -- floor division (integer result)
10 % 3    # 1         -- modulo (remainder)
-7 // 2   # -4        -- floor division rounds DOWN (not toward zero!)
-7 % 2    # 1         -- Python modulo always non-negative
```

---

### Q22. What is the difference between `None`, `False`, `0`, `""`, `[]`?
```python
# All are "falsy" -- evaluate to False in boolean context
bool(None)  # False
bool(False) # False
bool(0)     # False
bool("")    # False
bool([])    # False
bool({})    # False

# But they are NOT equal to each other
None == False  # False
None == 0      # False
0 == False     # True  (this one is True!)
```

---

### Q23. What is the difference between `sort()` and `sorted()`?
```python
lst = [3, 1, 4, 1, 5]

lst.sort()          # modifies list IN PLACE, returns None
new = sorted(lst)   # returns NEW sorted list, original unchanged

# Both support key and reverse
sorted(lst, key=lambda x: -x)  # descending
```

---

### Q24. What is a Python closure?
```python
def make_multiplier(n):
    def multiplier(x):
        return x * n   # n is "closed over" from outer scope
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(5))  # 10
print(triple(5))  # 15
```
> A closure is a function that remembers variables from its enclosing scope even after that scope has finished.

---

### Q25. What is the difference between `__init__` and `__new__`?
```python
class MyClass:
    def __new__(cls, *args, **kwargs):
        # called FIRST -- creates and returns the new instance
        print("__new__ called")
        instance = super().__new__(cls)
        return instance

    def __init__(self, value):
        # called SECOND -- initializes the instance
        print("__init__ called")
        self.value = value
```
> `__new__` creates the object. `__init__` initializes it. You rarely override `__new__` unless implementing singletons or metaclasses.

---

### Q26. What is the difference between `@property` and a regular attribute?
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def area(self):
        return 3.14 * self._radius ** 2  # computed on access

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

c = Circle(5)
print(c.area)    # 78.5 -- no () needed, looks like attribute
c.radius = 10    # calls setter with validation
```

---

### Q27. What is the difference between `isinstance` and `type`?
```python
class Animal: pass
class Dog(Animal): pass

d = Dog()
print(type(d) == Dog)        # True
print(type(d) == Animal)     # False -- type() is exact match only

print(isinstance(d, Dog))    # True
print(isinstance(d, Animal)) # True  -- isinstance checks inheritance chain
```
> Always prefer `isinstance` over `type()` for type checking.

---

### Q28. What is `enumerate` and when do you use it?
```python
fruits = ["apple", "banana", "cherry"]

# Without enumerate (bad)
for i in range(len(fruits)):
    print(i, fruits[i])

# With enumerate (good)
for i, fruit in enumerate(fruits):
    print(i, fruit)

# Start from custom index
for i, fruit in enumerate(fruits, start=1):
    print(i, fruit)  # 1 apple, 2 banana, 3 cherry
```

---

### Q29. What is the difference between `del`, `.remove()`, and `.pop()`?
```python
lst = [10, 20, 30, 40]

del lst[1]       # delete by index -- lst is now [10, 30, 40]
lst.remove(30)   # delete by value (first occurrence) -- [10, 40]
val = lst.pop()  # remove and return last element -- val=40, lst=[10]
val = lst.pop(0) # remove and return element at index 0
```

---

### Q30. What is the difference between `input()` and `sys.stdin.readline()`?
```python
# input() -- strips trailing newline, returns str
name = input("Enter name: ")

# sys.stdin.readline() -- keeps trailing newline, faster for large input
import sys
name = sys.stdin.readline().strip()

# For competitive programming / large input, sys.stdin is much faster
```

---

### Q31. What are Python's built-in data types?
```
Numeric:   int, float, complex
Sequence:  str, list, tuple, range
Mapping:   dict
Set:       set, frozenset
Boolean:   bool
Binary:    bytes, bytearray, memoryview
None:      NoneType
```

---

### Q32. What is the difference between `//` floor division and `int()` truncation?
```python
int(-7 / 2)   # -3  -- truncates toward zero
-7 // 2       # -4  -- floor division rounds toward negative infinity
```

---

### Q33. What is `*` unpacking?
```python
a, *b, c = [1, 2, 3, 4, 5]
# a = 1, b = [2, 3, 4], c = 5

# Merge lists
list1 = [1, 2]
list2 = [3, 4]
merged = [*list1, *list2]  # [1, 2, 3, 4]

# Merge dicts
d1 = {"a": 1}
d2 = {"b": 2}
merged = {**d1, **d2}  # {"a": 1, "b": 2}
```

---

### Q34. What is the difference between `any()` and `all()`?
```python
nums = [1, 2, 0, 4]

any(nums)                    # True  -- at least one truthy value
all(nums)                    # False -- not all are truthy (0 is falsy)

any(x > 3 for x in nums)    # True  -- 4 > 3
all(x > 0 for x in nums)    # False -- 0 is not > 0
```

---

### Q35. What is the difference between `__slots__` and regular attributes?
```python
# Regular class -- each instance has a __dict__ (flexible but uses more memory)
class Normal:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# With __slots__ -- no __dict__, fixed attributes, less memory
class Optimized:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Use __slots__ when creating millions of instances (e.g., data processing)
```

---

### Q36. What is the difference between `try/except` and `try/except/else/finally`?
```python
try:
    result = risky_operation()
except ValueError:
    handle_error()
else:
    # runs ONLY if no exception was raised in try block
    use_result(result)
finally:
    # ALWAYS runs -- use for cleanup (close files, DB connections)
    cleanup()
```

---

### Q37. What is `collections` module and its key classes?
```python
from collections import Counter, defaultdict, OrderedDict, deque, namedtuple

# Counter -- count occurrences
c = Counter("banana")  # Counter({'a': 3, 'n': 2, 'b': 1})
c.most_common(2)       # [('a', 3), ('n', 2)]

# defaultdict -- dict with default value for missing keys
dd = defaultdict(list)
dd["key"].append(1)  # no KeyError even if "key" doesn't exist

# deque -- double-ended queue, O(1) append/pop from both ends
dq = deque([1, 2, 3])
dq.appendleft(0)  # [0, 1, 2, 3]
dq.popleft()      # 0

# namedtuple -- tuple with named fields
Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)
print(p.x, p.y)  # 3 4
```

---

### Q38. What is the difference between `==` and `is` for strings (string interning)?
```python
a = "hello"
b = "hello"
print(a == b)   # True
print(a is b)   # True  -- Python interns short strings (same object!)

a = "hello world"
b = "hello world"
print(a is b)   # May be False -- longer strings may not be interned
```
> Python automatically interns small integers (-5 to 256) and short strings. Don't rely on `is` for value comparison.

---

### Q39. What is the difference between `import module` and `from module import name`?
```python
import os
os.path.join("a", "b")   # must use full path

from os import path
path.join("a", "b")       # shorter, but pollutes namespace

from os.path import join
join("a", "b")            # shortest, but can conflict with other names

# Avoid: from module import *  -- imports everything, causes name conflicts
```

---

### Q40. What is the difference between `@dataclass` and a regular class?
```python
from dataclasses import dataclass, field

# Regular class -- lots of boilerplate
class PersonOld:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __repr__(self):
        return f"Person(name={self.name}, age={self.age})"
    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

# Dataclass -- auto-generates __init__, __repr__, __eq__
@dataclass
class Person:
    name: str
    age: int
    scores: list = field(default_factory=list)  # mutable default

p1 = Person("Sujit", 21)
p2 = Person("Sujit", 21)
print(p1 == p2)   # True -- __eq__ auto-generated
print(p1)         # Person(name='Sujit', age=21)
```

---

---

## 22. Software Engineering Fundamentals - Interview Questions

> These are the questions every interviewer asks regardless of role — frontend, backend, full-stack, AI engineer, or DevOps. Know these cold.

---

### Q1. What is HTTP and HTTPS? What is the difference?

HTTP (HyperText Transfer Protocol) is the foundation of data communication on the web. It is a request-response protocol — the client (browser or app) sends a request, and the server sends back a response. Every time you open a webpage, your browser is making HTTP requests.

HTTPS is HTTP with an added security layer called SSL/TLS. The data transferred between client and server is encrypted, so even if someone intercepts the traffic, they cannot read it. This is why login pages, payment forms, and any page handling sensitive data must use HTTPS.

HTTP uses port 80 by default. HTTPS uses port 443. In modern web development, you should always use HTTPS — browsers now show a "Not Secure" warning for HTTP sites.

---

**Q5. What is the difference between REST and GraphQL?**

REST uses multiple endpoints — one URL per resource type (`/users`, `/posts`, `/comments`). The server decides what data each endpoint returns. This can lead to over-fetching (getting more data than you need) or under-fetching (needing to make multiple requests to get all the data you need).

GraphQL uses a single endpoint (`/graphql`). The client sends a query specifying exactly which fields it needs, and the server returns only those fields. No over-fetching, no under-fetching. This is especially powerful for mobile apps where bandwidth matters.

REST is simpler, more widely understood, and easier to cache. GraphQL is better for complex, nested data with many related entities, or when different clients (mobile, web, desktop) need different shapes of the same data.

In practice, most companies use REST for simple CRUD APIs and consider GraphQL when they have complex data requirements or many different client types.

**Q6. What is an API?**

API (Application Programming Interface) is a contract that defines how two software systems communicate. It specifies what requests you can make, what format to send them in, and what responses to expect.

A REST API is one specific type of API that uses HTTP and follows REST principles. Other types include GraphQL (query language for APIs), gRPC (high-performance binary protocol, used between microservices), WebSocket (real-time bidirectional communication), and SOAP (older XML-based protocol, still used in enterprise/banking).

**Q7. What is JSON and why is it used in APIs?**

JSON (JavaScript Object Notation) is a lightweight, text-based data format for representing structured data as key-value pairs, arrays, and nested objects. It is the standard format for REST API request and response bodies.

It became the standard because it is human-readable, easy to parse in every programming language, and maps naturally to objects in most languages. It replaced XML (which was verbose and harder to parse) as the dominant API data format around 2010.

```json
{
  "name": "Sujit",
  "age": 21,
  "skills": ["Python", "AI", "FastAPI"],
  "address": { "city": "Pune", "country": "India" }
}
```

**Q8. What is the difference between synchronous and asynchronous programming?**

Synchronous programming means tasks execute one at a time, in order. Each task must complete before the next one starts. If a task involves waiting (like an API call that takes 2 seconds), the entire program is blocked and does nothing during that wait.

Asynchronous programming means a task can start, then yield control while waiting for something (like a network response), allowing other tasks to run in the meantime. When the wait is over, the original task resumes. This makes much better use of time — instead of sitting idle during I/O waits, the program does other useful work.

For AI applications, async is critical. LLM API calls can take 2–10 seconds. With sync code, your server can only handle one request at a time. With async, it can handle hundreds of concurrent requests while they all wait for their LLM responses.

```python
# Sync -- server blocked for 2s, can't handle other requests
def get_answer(query):
    response = requests.post(llm_url, json={"query": query})  # blocks here
    return response.json()

# Async -- server free to handle other requests while waiting
async def get_answer(query):
    async with aiohttp.ClientSession() as session:
        response = await session.post(llm_url, json={"query": query})  # yields control
        return await response.json()
```

**Q9. What is a database? What is the difference between SQL and NoSQL?**

A database is an organized system for storing, retrieving, and managing data. Without a database, data would be lost when the application restarts.

SQL databases (PostgreSQL, MySQL, SQLite) store data in structured tables with rows and columns. They enforce a fixed schema — every row must have the same columns with the same data types. They support ACID transactions, which guarantee data integrity. They are excellent when your data has clear relationships (users have orders, orders have products).

NoSQL databases (MongoDB, Cassandra, Redis, DynamoDB) store data in flexible formats — documents, key-value pairs, or wide columns. They do not enforce a fixed schema, so different records can have different fields. They scale horizontally more easily and handle very high write volumes better.

Choose SQL when data is structured, relationships matter, and consistency is critical (banking, e-commerce). Choose NoSQL when data is unstructured, schema changes frequently, or you need massive scale (social media activity logs, IoT sensor data).

**Q10. What is CRUD?**

CRUD stands for Create, Read, Update, Delete — the four basic operations you can perform on any data. Almost every application is fundamentally a CRUD application at its core.

Create maps to SQL INSERT and HTTP POST. Read maps to SQL SELECT and HTTP GET. Update maps to SQL UPDATE and HTTP PUT or PATCH. Delete maps to SQL DELETE and HTTP DELETE.

When an interviewer asks you to build a simple API, they are almost always asking for a CRUD API.

**Q11. What is an ORM and why use it?**

ORM (Object-Relational Mapper) is a library that lets you interact with a relational database using your programming language's objects instead of writing raw SQL. You define Python classes that map to database tables, and the ORM translates your Python operations into SQL queries automatically.

The benefits are: you write Python instead of SQL (less context switching), the ORM handles SQL injection prevention automatically (parameterized queries by default), and you get database portability (switch from SQLite to PostgreSQL by changing one config line).

The downside is that ORMs can generate inefficient SQL for complex queries. For performance-critical queries, you can always drop down to raw SQL even when using an ORM.

```python
# Raw SQL -- verbose, manual, injection risk if not careful
cursor.execute("SELECT * FROM users WHERE id = %s AND active = %s", (user_id, True))

# ORM (SQLAlchemy) -- cleaner, safe, Pythonic
user = session.query(User).filter(User.id == user_id, User.active == True).first()
```

**Q12. What is database indexing and why does it matter?**

An index is a separate data structure (usually a B-tree) that the database maintains alongside a table to speed up lookups on specific columns. Without an index, a query like `WHERE email = 'sujit@email.com'` requires scanning every single row in the table — O(n). With an index on the email column, the database can jump directly to the matching rows — O(log n).

The trade-off is that indexes take up extra disk space and slow down INSERT, UPDATE, and DELETE operations slightly (because the index must be updated too). So you add indexes on columns you frequently search, filter, sort, or join on — not on every column.

Primary keys are automatically indexed. Foreign keys should almost always be indexed. Columns used in WHERE clauses frequently are good candidates.

**Q13. What is a primary key vs a foreign key?**

A primary key is a column (or combination of columns) that uniquely identifies each row in a table. It cannot be NULL and must be unique. Every table should have a primary key. It is automatically indexed.

A foreign key is a column in one table that references the primary key of another table. It creates a relationship between the two tables and enforces referential integrity — you cannot insert a foreign key value that does not exist in the referenced table, and you cannot delete a row that is referenced by a foreign key (unless you cascade the delete).

```
users table:  id (PK), name, email
posts table:  id (PK), user_id (FK → users.id), title, content

user_id in posts is a foreign key -- it must match an existing id in users
```

**Q14. What is database normalization?**

Normalization is the process of organizing a database to eliminate data redundancy (storing the same data in multiple places) and ensure data integrity. The idea is that each piece of information should be stored in exactly one place.

1NF (First Normal Form): each column holds atomic (indivisible) values. No repeating groups or arrays in a column.

2NF (Second Normal Form): no partial dependencies — every non-key column must depend on the entire primary key, not just part of it. Relevant when you have composite primary keys.

3NF (Third Normal Form): no transitive dependencies — non-key columns should not depend on other non-key columns. If city depends on zip_code, and zip_code is not the primary key, move city and zip_code to a separate table.

In practice, most databases are designed to 3NF. Sometimes you intentionally denormalize (store redundant data) for performance — this is a deliberate trade-off.

**Q15. What is a database transaction? What is ACID?**

A transaction is a sequence of database operations that are treated as a single unit of work. Either all operations succeed and are committed, or if any operation fails, all changes are rolled back and the database is left unchanged. This prevents partial updates that leave data in an inconsistent state.

The classic example is a bank transfer: debit $100 from Account A, credit $100 to Account B. If the debit succeeds but the credit fails, you have lost $100. A transaction ensures both happen or neither happens.

ACID is the set of properties that guarantee reliable transactions:
- Atomicity: all operations in the transaction succeed, or none of them do. No partial updates.
- Consistency: the database moves from one valid state to another. All constraints and rules are maintained.
- Isolation: concurrent transactions do not interfere with each other. Each transaction sees a consistent snapshot of the data.
- Durability: once a transaction is committed, it is permanent — even if the server crashes immediately after.

**Q16. What is caching? What is Redis?**

Caching is storing the result of an expensive operation in fast memory so future requests can be served without repeating the work. The goal is to reduce latency and reduce load on the database or external services.

Redis is an in-memory key-value store that is extremely fast — it can handle millions of operations per second with sub-millisecond latency. It is the most popular caching solution. Beyond caching, Redis is also used for session storage, rate limiting, pub/sub messaging, and leaderboards.

Cache-aside (lazy loading) is the most common pattern: the application checks the cache first. On a cache miss, it reads from the database and stores the result in the cache for future requests. The cache entry expires after a TTL (time-to-live) to prevent stale data.

The hardest problem in caching is cache invalidation — knowing when to remove or update a cached value when the underlying data changes.

**Q17. What is a message queue? When do you use Kafka vs RabbitMQ?**

A message queue is a buffer that sits between services. A producer sends a message to the queue, and a consumer reads and processes it. The producer and consumer are decoupled — they do not need to be running at the same time or at the same speed.

You use a message queue when you want to process tasks asynchronously (send a welcome email after signup without blocking the signup response), handle traffic spikes (queue absorbs the burst, consumers process at their own pace), or communicate between microservices without tight coupling.

Kafka is designed for high-throughput event streaming. It retains messages for days and allows multiple independent consumers to read the same messages. Use Kafka for event sourcing, analytics pipelines, and audit logs.

RabbitMQ is a traditional message broker with flexible routing. Each message is typically consumed by exactly one consumer and then deleted. Use RabbitMQ for task queues where you want to distribute work across multiple workers.

**Q18. What is Docker and why is it used?**

Docker is a platform for packaging an application and all its dependencies (runtime, libraries, config) into a standardized unit called a container. The container runs the same way on any machine — your laptop, a colleague's machine, or a production server.

The problem Docker solves is "it works on my machine." Before Docker, deploying an application meant manually installing the right Python version, the right libraries, and the right system dependencies on every server. With Docker, you define everything in a Dockerfile, build an image once, and run it anywhere.

A container is like a lightweight virtual machine, but instead of virtualizing the entire OS, it shares the host OS kernel and only isolates the application's filesystem and processes. This makes containers much faster to start and much smaller than VMs.

**Q19. What is CI/CD?**

CI (Continuous Integration) is the practice of developers merging their code changes into a shared repository frequently — multiple times a day. Each merge triggers an automated pipeline that builds the code and runs all tests. The goal is to catch integration bugs early, before they accumulate.

CD (Continuous Delivery) means the code is always in a deployable state. After CI passes, the code is automatically packaged and ready to deploy to production with a single click or approval.

CD (Continuous Deployment) goes one step further — every change that passes CI is automatically deployed to production without any manual step.

The benefit is speed and confidence. Instead of a stressful monthly release with hundreds of changes, you deploy small changes frequently. If something breaks, it is easy to identify which change caused it and roll it back.

**Q20. What is authentication vs authorization?**

Authentication is the process of verifying who you are. It answers the question "are you who you claim to be?" — login with username/password, Google OAuth, fingerprint, etc.

Authorization is the process of verifying what you are allowed to do. It answers "do you have permission to do this?" — can this user read this document, delete this record, access this admin panel?

Authentication always comes first. You cannot authorize someone whose identity you have not verified. A common mistake is confusing 401 (Unauthorized — actually means unauthenticated, you need to log in) with 403 (Forbidden — you are authenticated but do not have permission).

**Q21. What is JWT (JSON Web Token)?**

JWT is a compact, self-contained token format used for authentication. After a user logs in, the server creates a JWT containing the user's ID and role, signs it with a secret key, and sends it to the client. The client stores it and sends it with every subsequent request in the Authorization header.

The server can verify the token by checking the signature — it does not need to look up the user in the database on every request. This makes JWT stateless and scalable.

The JWT has three parts separated by dots: Header (algorithm used), Payload (user data like id, role, expiry), and Signature (HMAC of header + payload using the secret key). If anyone tampers with the payload, the signature will not match and the token is rejected.

The main weakness of JWT is that you cannot invalidate a token before it expires. If a user logs out or their account is compromised, the token is still valid until expiry. Solutions include short expiry times with refresh tokens, or maintaining a token blacklist in Redis.

**Q22. What is OAuth 2.0?**

OAuth 2.0 is an authorization framework that allows a user to grant a third-party application access to their data on another service, without sharing their password.

When you click "Login with Google," you are using OAuth. Your app never sees your Google password. Instead, Google authenticates you and gives your app an access token that it can use to call Google APIs on your behalf.

The key insight is delegation — you are delegating limited access to your data. The scope defines exactly what the app can access (read your email, see your profile, etc.). You can revoke this access at any time from your Google account settings.

**Q23. What is CORS and why does it exist?**

CORS (Cross-Origin Resource Sharing) is a browser security mechanism that blocks JavaScript from making requests to a different origin (different domain, port, or protocol) than the page it is running on.

It exists to prevent malicious websites from making API calls on behalf of a logged-in user. If you are logged into your bank and visit a malicious site, without CORS the malicious site's JavaScript could make requests to your bank's API using your cookies.

When your frontend (running on `localhost:3000`) calls your backend API (running on `localhost:8000`), the browser blocks it because the origins are different. You fix this by configuring your backend to send CORS headers that explicitly allow your frontend's origin.

**Q24. What is SQL injection and how do you prevent it?**

SQL injection is an attack where a malicious user inserts SQL code into an input field, and that code gets executed as part of a database query. It is one of the most common and dangerous web vulnerabilities.

If you build a query by concatenating user input directly into a SQL string, an attacker can input something like `'; DROP TABLE users; --` and destroy your database.

The prevention is always using parameterized queries (also called prepared statements). The database driver separates the SQL code from the data — the user input is always treated as data, never as SQL code. ORMs like SQLAlchemy use parameterized queries by default, which is one reason to use them.

```python
# VULNERABLE -- never do this
query = f"SELECT * FROM users WHERE name = '{user_input}'"

# SAFE -- parameterized query
cursor.execute("SELECT * FROM users WHERE name = %s", (user_input,))
```

**Q25. What is the difference between TCP and UDP?**

TCP (Transmission Control Protocol) is a connection-oriented protocol. Before sending data, it establishes a connection with a 3-way handshake. It guarantees that all data arrives, in the correct order, with no duplicates. If a packet is lost, TCP automatically retransmits it. This reliability comes at the cost of overhead and latency.

UDP (User Datagram Protocol) is connectionless. It just sends packets without establishing a connection or guaranteeing delivery. Packets can arrive out of order, be duplicated, or be lost entirely. But it is much faster because there is no overhead.

Use TCP when reliability matters: HTTP, HTTPS, email, file transfers, database connections. Use UDP when speed matters more than reliability: video streaming (a dropped frame is better than pausing to retransmit), online gaming (old position data is useless anyway), DNS lookups (fast, small queries).

**Q26. What is DNS?**

DNS (Domain Name System) is the internet's phone book. It translates human-readable domain names like `www.google.com` into IP addresses like `142.250.195.46` that computers use to communicate.

When you type a URL, your browser first checks its local cache, then your OS cache, then queries a DNS resolver (usually provided by your ISP or a public resolver like 8.8.8.8). The resolver works through a hierarchy — root nameservers, TLD nameservers (.com, .in), and finally the authoritative nameserver for that specific domain — to find the IP address.

DNS is why you can change your server's IP address without users noticing — you just update the DNS record and within minutes (based on TTL) everyone is directed to the new IP.

**Q27. What is a CDN?**

A CDN (Content Delivery Network) is a globally distributed network of servers that caches and serves content from the location geographically closest to the user.

Without a CDN, a user in Mumbai requesting an image from a server in the US experiences 200ms+ of network latency just for the round trip. With a CDN, that image is cached on a server in Mumbai and served in under 10ms.

CDNs are used for static assets (images, CSS, JavaScript, fonts), video streaming, and as a DDoS protection layer. When you deploy a new version of a file, you either use cache-busting (change the filename) or invalidate the CDN cache.

**Q28. What is WebSocket?**

WebSocket is a communication protocol that provides a persistent, full-duplex (two-way) connection between client and server over a single TCP connection.

Regular HTTP is request-response — the client asks, the server answers, the connection closes. For real-time features like chat, live notifications, or collaborative editing, this is inefficient because the client would have to constantly poll the server asking "anything new?"

WebSocket keeps the connection open. Either side can send a message at any time without waiting for the other to ask. This is how WhatsApp Web, Google Docs live collaboration, and live stock price feeds work.

**Q29. What is the difference between horizontal and vertical scaling?**

Vertical scaling (scaling up) means adding more resources to your existing server — more CPU, more RAM, faster storage. It is simple because you do not change your application. But it has a hard limit (you can only add so much to one machine) and is a single point of failure.

Horizontal scaling (scaling out) means adding more servers and distributing the load across them with a load balancer. There is no theoretical limit — you can keep adding servers. But it requires your application to be stateless (no session data stored on the server, because the next request might go to a different server).

Most modern cloud architectures use horizontal scaling because it is more resilient and cost-effective — you can add servers during peak traffic and remove them when traffic drops.

**Q30. What is Big O notation?**

Big O notation describes how the time or space requirements of an algorithm grow as the input size grows. It gives you a way to compare algorithms independent of hardware or implementation details.

O(1) means constant time — the operation takes the same time regardless of input size. Dictionary lookup is O(1). O(n) means linear time — time grows proportionally with input size. Looping through a list is O(n). O(n²) means quadratic — nested loops. O(log n) means logarithmic — binary search halves the search space each step.

In interviews, you are expected to state the time and space complexity of your solution. Always analyze the worst case unless told otherwise.

```
O(1)       -- dict lookup, array access by index
O(log n)   -- binary search, balanced BST operations
O(n)       -- single loop, linear search
O(n log n) -- merge sort, Python's built-in sort (Timsort)
O(n²)      -- nested loops, bubble sort
O(2^n)     -- recursive fibonacci without memoization
```

**Q31. What is the difference between a stack and a queue?**

A stack is a LIFO (Last In, First Out) data structure. The last element you add is the first one you remove — like a stack of plates. Operations: push (add to top), pop (remove from top). Used for: function call stack, undo/redo, DFS traversal, matching brackets.

A queue is a FIFO (First In, First Out) data structure. The first element you add is the first one you remove — like a line at a ticket counter. Operations: enqueue (add to back), dequeue (remove from front). Used for: BFS traversal, task scheduling, print queues, message queues.

In Python, use a list for a stack (`.append()` and `.pop()`). Use `collections.deque` for a queue (`.append()` and `.popleft()`) — deque has O(1) operations on both ends, while a list's `.pop(0)` is O(n).

**Q32. What is recursion? What must every recursive function have?**

Recursion is when a function calls itself to solve a smaller version of the same problem. It is useful for problems that have a naturally recursive structure — trees, graphs, divide-and-conquer algorithms.

Every recursive function must have a base case — a condition where the function stops calling itself and returns a value directly. Without a base case, the function calls itself forever until Python raises a `RecursionError` (default limit is 1000 calls deep).

The mental model: trust that the recursive call correctly solves the smaller problem, and focus on how to combine that result with the current step.

```python
def factorial(n):
    if n == 0:              # base case -- stop here
        return 1
    return n * factorial(n - 1)  # recursive case -- smaller problem

# factorial(4) = 4 * factorial(3) = 4 * 3 * factorial(2) = 4 * 3 * 2 * 1 = 24
```

**Q33. What is the difference between compiled and interpreted languages?**

A compiled language (C, C++, Go, Rust) translates source code into machine code before execution. The compiler produces a binary executable that runs directly on the CPU. This makes compiled programs very fast at runtime, but you must compile for each target platform.

An interpreted language (Python, JavaScript, Ruby) is executed line by line at runtime by an interpreter. There is no separate compilation step — you just run the source file. This makes development faster and more flexible, but execution is slower because the interpreter adds overhead.

Python is actually a hybrid — it compiles source code to bytecode (`.pyc` files) first, then the CPython virtual machine interprets the bytecode. This is why Python is faster than a pure interpreter but slower than a fully compiled language.

**Q34. What is an environment variable and why should you use it?**

An environment variable is a key-value pair stored in the operating system's environment, outside of your application code. Your application reads these values at runtime.

You use environment variables to store configuration that changes between environments (development, staging, production) and to store secrets (API keys, database passwords) that should never be committed to version control.

If you hardcode `OPENAI_API_KEY = "sk-abc123"` in your code and push it to GitHub, that key is now public and will be found by bots within minutes. Environment variables keep secrets out of your codebase entirely.

```python
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file into environment

api_key = os.getenv("OPENAI_API_KEY")  # reads from environment
db_url  = os.getenv("DATABASE_URL", "sqlite:///./default.db")  # with fallback
```

**Q35. What is rate limiting and why is it important?**

Rate limiting restricts how many requests a client can make to an API within a given time window. For example, 100 requests per minute per user.

Without rate limiting, a single client could flood your server with thousands of requests per second, consuming all your resources and making the service unavailable for everyone else (a denial-of-service attack). Rate limiting also prevents abuse of expensive operations like LLM API calls.

When a client exceeds the limit, the server returns 429 Too Many Requests. The response typically includes headers telling the client when the limit resets so it can retry.

**Q36. What is a webhook?**

A webhook is a way for one system to notify another system in real time when an event occurs, by sending an HTTP POST request to a pre-configured URL.

The alternative is polling — your application repeatedly asks "did anything happen?" every few seconds. This is wasteful. A webhook is push-based — the other system calls you when something happens.

For example, when a payment succeeds on Stripe, Stripe sends a POST request to your webhook URL with the payment details. Your server receives it and updates the order status. You do not need to poll Stripe every second asking "did the payment go through?"

**Q37. What is Docker and what is the difference between an image and a container?**

A Docker image is a read-only blueprint — it contains the application code, runtime, libraries, and all dependencies, packaged together. Think of it like a class definition.

A Docker container is a running instance of an image — it is the actual running process with its own isolated filesystem and network. Think of it like an object created from a class. You can run multiple containers from the same image simultaneously.

You build an image from a Dockerfile (a set of instructions). You push images to a registry (Docker Hub, AWS ECR). You pull images and run them as containers on any machine.

**Q38. What is CI/CD and what are its benefits?**

CI (Continuous Integration) means developers merge code to the main branch frequently — multiple times a day. Each merge automatically triggers a pipeline that builds the code and runs all tests. This catches bugs early, before they compound with other changes.

CD (Continuous Delivery/Deployment) means every change that passes CI is automatically prepared for deployment (Delivery) or automatically deployed to production (Deployment).

The benefits: faster release cycles, smaller and safer deployments (easier to identify what broke), higher confidence (automated tests catch regressions), and less manual work. Instead of a stressful quarterly release, you deploy small changes daily.

**Q39. What is SOLID and why does it matter?**

SOLID is a set of five design principles for writing maintainable, extensible object-oriented code.

Single Responsibility: a class should do one thing and have one reason to change. A class that handles both database queries and email sending is doing too much.

Open/Closed: classes should be open for extension but closed for modification. Add new behavior by adding new code, not by changing existing code.

Liskov Substitution: a subclass should be usable anywhere its parent class is used, without breaking the program.

Interface Segregation: do not force classes to implement methods they do not need. Split large interfaces into smaller, focused ones.

Dependency Inversion: depend on abstractions (interfaces/ABCs), not concrete implementations. This makes code easier to test and swap components.

**Q40. What is technical debt?**

Technical debt is the accumulated cost of shortcuts, quick fixes, and poor design decisions made in the past. Like financial debt, it accrues interest — the longer you leave it, the harder and more expensive it becomes to fix.

It happens when teams prioritize speed over quality: skipping tests to ship faster, copying code instead of refactoring, using a quick hack instead of a proper solution.

Signs of high technical debt: adding a new feature takes weeks because the codebase is tangled, every change breaks something unexpected, no one wants to touch certain parts of the code, and onboarding new developers takes months.

Managing it requires discipline: allocate time in every sprint for refactoring, write tests before changing risky code, and document decisions so future developers understand why things were done a certain way.

---

> **Quick Revision Summary:**
> HTTP: GET (read), POST (create), PUT (replace), PATCH (partial update), DELETE (remove). Status codes: 2xx success, 4xx client error, 5xx server error. REST is stateless, resource-based, uniform interface. JWT is stateless auth token. OAuth is delegated authorization. SQL for structured relational data, NoSQL for flexible/high-scale. ACID ensures transaction reliability. Redis for caching. Docker packages apps into containers. CI/CD automates testing and deployment. SOLID principles guide maintainable OOP design.



---

**Q3. What are HTTP status codes? What do the different ranges mean?**

HTTP status codes are 3-digit numbers the server sends back to tell the client what happened with the request. They are grouped into five ranges.

1xx (Informational) — the request was received and is being processed. Rarely seen in practice.

2xx (Success) — the request was successful. 200 OK is the standard success response. 201 Created means a new resource was created (used after POST). 204 No Content means success but there is nothing to return (used after DELETE).

3xx (Redirection) — the client needs to take further action. 301 Moved Permanently means the URL has changed forever. 304 Not Modified means the cached version is still valid, no need to re-download.

4xx (Client Error) — the client made a mistake. 400 Bad Request means invalid input. 401 Unauthorized means the client is not authenticated (not logged in). 403 Forbidden means the client is authenticated but does not have permission. 404 Not Found means the resource does not exist. 429 Too Many Requests means the client is being rate limited.

5xx (Server Error) — the server failed. 500 Internal Server Error means something crashed on the server. 503 Service Unavailable means the server is overloaded or down for maintenance.

The most important ones to know: 200, 201, 204, 400, 401, 403, 404, 422, 429, 500, 503.

---

**Q4. What is a REST API?**

REST (Representational State Transfer) is an architectural style for designing APIs. It is not a protocol or a standard — it is a set of principles that, when followed, make APIs predictable, scalable, and easy to use.

The key principles are: Stateless — every request must contain all the information needed to process it. The server does not store any session state between requests. This makes the server easier to scale because any server can handle any request. Client-Server — the client and server are separate and communicate only through the API. Resource-based — everything is a resource (a user, a post, an order) identified by a URL. You use HTTP methods to perform actions on those resources. Uniform Interface — consistent URL structure and HTTP method usage across the entire API.

A well-designed REST API is intuitive. Just by looking at `GET /api/users/1/posts`, you know it fetches the posts of user 1. You do not need documentation to guess what it does.

```
GET    /api/users          -- list all users
GET    /api/users/1        -- get user with id 1
POST   /api/users          -- create a new user
PUT    /api/users/1        -- replace user 1
PATCH  /api/users/1        -- partially update user 1
DELETE /api/users/1        -- delete user 1
GET    /api/users/1/posts  -- get all posts by user 1
```

---

**Q5. What is the difference between REST and GraphQL?**
```
REST:
  - Multiple endpoints (/users, /posts, /comments)
  - Fixed response structure (may over-fetch or under-fetch)
  - Simple, widely understood
  - Good for simple CRUD APIs

GraphQL:
  - Single endpoint (/graphql)
  - Client specifies exactly what data it needs
  - No over-fetching or under-fetching
  - Better for complex, nested data
  - Steeper learning curve
```

---

**Q6. What is an API? What is the difference between API and REST API?**
- API (Application Programming Interface) is any interface that allows two systems to communicate.
- REST API is a specific type of API that follows REST architectural principles over HTTP.
- Other API types: SOAP, GraphQL, gRPC, WebSocket

---

**Q7. What is JSON and why is it used in APIs?**
```json
{
  "name": "Sujit",
  "age": 21,
  "skills": ["Python", "AI", "FastAPI"],
  "address": {
    "city": "Pune",
    "country": "India"
  }
}
```
- JSON (JavaScript Object Notation) is a lightweight, human-readable data format.
- Used in APIs because it's language-agnostic, easy to parse, and supported everywhere.

---

**Q8. What is the difference between synchronous and asynchronous programming?**
```python
# Synchronous -- blocking, one task at a time
def fetch_data():
    data = requests.get(url)  # waits here until response
    return data

# Asynchronous -- non-blocking, can do other work while waiting
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        data = await session.get(url)  # yields control while waiting
    return data
```
> Use async for I/O-bound tasks (API calls, DB queries). Use sync for CPU-bound tasks.

---

**Q9. What is a database? Difference between SQL and NoSQL?**
```
SQL (Relational):
  - Structured tables with rows and columns
  - Fixed schema
  - ACID transactions
  - Examples: PostgreSQL, MySQL, SQLite
  - Use when: data is structured, relationships matter, consistency is critical

NoSQL (Non-Relational):
  - Flexible schema
  - Types: Document (MongoDB), Key-Value (Redis), Column (Cassandra), Graph (Neo4j)
  - Scales horizontally
  - Use when: unstructured data, high write volume, flexible schema needed
```

---

**Q10. What is CRUD?**
```
C -- Create  (INSERT / POST)
R -- Read    (SELECT / GET)
U -- Update  (UPDATE / PUT/PATCH)
D -- Delete  (DELETE / DELETE)
```

---

**Q11. What is an ORM?**
ORM (Object-Relational Mapper) lets you interact with a database using Python objects instead of raw SQL.
```python
# Without ORM (raw SQL)
cursor.execute("SELECT * FROM users WHERE id = 1")

# With ORM (SQLAlchemy)
user = session.query(User).filter(User.id == 1).first()
```
Examples: SQLAlchemy, Django ORM, Tortoise ORM

---

**Q12. What is indexing in databases?**
An index is a data structure that speeds up data retrieval at the cost of extra storage and slower writes.
```sql
-- Without index: full table scan O(n)
SELECT * FROM users WHERE email = 'sujit@email.com';

-- With index on email: O(log n)
CREATE INDEX idx_email ON users(email);
```
> Add indexes on columns you frequently filter, sort, or join on.

---

**Q13. What is a primary key vs foreign key?**
```
Primary Key -- uniquely identifies each row in a table. Cannot be NULL.
Foreign Key -- references the primary key of another table. Creates a relationship.

users table:  id (PK), name, email
posts table:  id (PK), user_id (FK -> users.id), content
```

---

**Q14. What is normalization in databases?**
Normalization is the process of organizing a database to reduce data redundancy and improve integrity.
```
1NF -- each column has atomic values, no repeating groups
2NF -- no partial dependencies (non-key columns depend on full PK)
3NF -- no transitive dependencies (non-key columns don't depend on other non-key columns)
```

---

**Q15. What is a transaction? What is ACID?**
```
A transaction is a group of operations that execute as a single unit.

ACID properties:
  Atomicity   -- all operations succeed or all fail (no partial updates)
  Consistency -- database moves from one valid state to another
  Isolation   -- concurrent transactions don't interfere with each other
  Durability  -- committed transactions survive system failures
```

---

**Q16. What is caching? What is Redis?**
Caching stores frequently accessed data in fast memory to reduce database load and response time.
```
Redis -- in-memory key-value store, extremely fast (microsecond reads)
Use cases:
  - Cache API responses
  - Session storage
  - Rate limiting
  - Pub/Sub messaging
  - Leaderboards (sorted sets)

Cache strategies:
  Cache-aside:  app checks cache first, if miss, reads DB and stores in cache
  Write-through: write to cache and DB simultaneously
  Write-back:   write to cache only, sync to DB later (risk of data loss)
```

---

**Q17. What is a message queue? What is Kafka/RabbitMQ?**
A message queue decouples producers (who send messages) from consumers (who process them).
```
Use cases:
  - Async task processing (send email after signup)
  - Microservice communication
  - Event streaming
  - Load leveling (handle traffic spikes)

Kafka   -- high-throughput, distributed, persistent log. Good for event streaming.
RabbitMQ-- traditional message broker, supports complex routing. Good for task queues.
```

---

**Q18. What is Docker?**
Docker is a platform for packaging applications and their dependencies into containers.
```
Container -- isolated, lightweight runtime environment
Image     -- blueprint for a container (like a class)
Dockerfile-- instructions to build an image

# Basic Dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]

docker build -t myapp .
docker run -p 8000:8000 myapp
```
> Containers ensure "works on my machine" = "works everywhere"

---

**Q19. What is CI/CD?**
```
CI (Continuous Integration):
  - Developers merge code frequently
  - Automated tests run on every push
  - Catches bugs early

CD (Continuous Delivery/Deployment):
  - Delivery: code is always in deployable state, manual trigger to deploy
  - Deployment: automatically deploys to production after tests pass

Tools: GitHub Actions, Jenkins, GitLab CI, CircleCI
```

---

**Q20. What is Git? Difference between `git merge` and `git rebase`?**
```
git merge  -- combines branches, creates a merge commit, preserves history
git rebase -- moves commits on top of another branch, linear history, cleaner log

# merge (safe, preserves context)
git checkout main
git merge feature-branch

# rebase (cleaner history, but rewrites commits)
git checkout feature-branch
git rebase main
```
> Use merge for shared branches. Use rebase for local cleanup before merging.

---

**Q21. What is microservices architecture vs monolithic?**
```
Monolithic:
  - Single deployable unit
  - Simple to develop and test initially
  - Hard to scale individual parts
  - One bug can crash everything

Microservices:
  - Each service is independent (own DB, own deployment)
  - Scale individual services independently
  - Technology flexibility per service
  - More complex (network calls, distributed tracing, service discovery)
```

---

**Q22. What is load balancing?**
Load balancing distributes incoming traffic across multiple servers to prevent any single server from being overwhelmed.
```
Algorithms:
  Round Robin     -- requests go to each server in turn
  Least Connections -- send to server with fewest active connections
  IP Hash         -- same client always goes to same server (sticky sessions)

Tools: Nginx, AWS ALB, HAProxy
```

---

**Q23. What is authentication vs authorization?**
```
Authentication -- who are you? (login, verify identity)
  Methods: username/password, OAuth, JWT, API keys, biometrics

Authorization  -- what can you do? (permissions, access control)
  Methods: RBAC (Role-Based), ABAC (Attribute-Based), ACL

Example:
  Auth: You log in as Sujit (authentication)
  Authz: Sujit can read posts but cannot delete other users' posts (authorization)
```

---

**Q24. What is JWT (JSON Web Token)?**
```
JWT = Header.Payload.Signature (base64 encoded, dot-separated)

Header:  {"alg": "HS256", "typ": "JWT"}
Payload: {"user_id": 1, "role": "admin", "exp": 1234567890}
Signature: HMAC(header + payload, secret_key)

Flow:
  1. User logs in with credentials
  2. Server creates JWT and sends to client
  3. Client stores JWT (localStorage or cookie)
  4. Client sends JWT in Authorization header: "Bearer <token>"
  5. Server verifies signature, extracts user info

Pros: stateless, no DB lookup needed
Cons: cannot invalidate before expiry (use refresh tokens + blacklist)
```

---

**Q25. What is OAuth 2.0?**
OAuth 2.0 is an authorization framework that lets users grant third-party apps access to their data without sharing passwords.
```
Flow (Authorization Code):
  1. User clicks "Login with Google"
  2. App redirects to Google with client_id and scope
  3. User approves on Google
  4. Google redirects back with authorization code
  5. App exchanges code for access token (server-to-server)
  6. App uses access token to call Google APIs

Roles:
  Resource Owner  -- the user
  Client          -- your app
  Authorization Server -- Google/GitHub/etc.
  Resource Server -- API that holds user data
```

---

**Q26. What is CORS?**
CORS (Cross-Origin Resource Sharing) is a browser security mechanism that blocks requests from a different origin (domain/port/protocol) unless the server explicitly allows it.
```python
# FastAPI CORS setup
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myfrontend.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```

---

**Q27. What is the difference between cookies and localStorage?**
```
Cookies:
  - Sent automatically with every HTTP request
  - Can be HttpOnly (not accessible by JS -- safer)
  - Can be Secure (HTTPS only)
  - Has expiry
  - Good for: session tokens, auth

localStorage:
  - Stored in browser, NOT sent with requests
  - Accessible by JavaScript
  - Persists until manually cleared
  - Good for: user preferences, non-sensitive data

sessionStorage:
  - Same as localStorage but cleared when tab closes
```

---

**Q28. What is SQL injection and how do you prevent it?**
```python
# VULNERABLE -- never do this
query = f"SELECT * FROM users WHERE name = '{user_input}'"
# If user_input = "'; DROP TABLE users; --"  --> disaster

# SAFE -- use parameterized queries
cursor.execute("SELECT * FROM users WHERE name = %s", (user_input,))

# With SQLAlchemy ORM -- safe by default
user = session.query(User).filter(User.name == user_input).first()
```

---

**Q29. What is XSS (Cross-Site Scripting)?**
XSS is an attack where malicious scripts are injected into web pages viewed by other users.
```
Stored XSS:  attacker saves script in DB, runs when other users view it
Reflected XSS: script in URL, runs when user clicks malicious link

Prevention:
  - Escape/sanitize all user input before rendering
  - Use Content Security Policy (CSP) headers
  - Use HttpOnly cookies (JS can't steal them)
  - Modern frameworks (React, Vue) escape by default
```

---

**Q30. What is the difference between TCP and UDP?**
```
TCP (Transmission Control Protocol):
  - Connection-oriented (3-way handshake)
  - Reliable -- guarantees delivery and order
  - Slower due to overhead
  - Use: HTTP, HTTPS, email, file transfer

UDP (User Datagram Protocol):
  - Connectionless
  - No guarantee of delivery or order
  - Faster, lower latency
  - Use: video streaming, gaming, DNS, VoIP
```

---

**Q31. What is DNS?**
DNS (Domain Name System) translates human-readable domain names to IP addresses.
```
You type: www.google.com
DNS resolves: 142.250.195.46

Steps:
  1. Browser checks local cache
  2. OS checks /etc/hosts
  3. Query goes to recursive resolver (ISP)
  4. Resolver queries root nameserver -> TLD (.com) -> authoritative nameserver
  5. Returns IP address
```

---

**Q32. What is a CDN?**
CDN (Content Delivery Network) is a distributed network of servers that delivers content from the server closest to the user.
```
Benefits:
  - Faster load times (geographically closer)
  - Reduced load on origin server
  - DDoS protection
  - Caches static assets (images, CSS, JS)

Examples: Cloudflare, AWS CloudFront, Akamai
```

---

**Q33. What is WebSocket?**
WebSocket is a protocol that provides full-duplex (two-way) communication over a single TCP connection.
```
HTTP:      client sends request --> server responds --> connection closes
WebSocket: connection stays open, both sides can send messages anytime

Use cases:
  - Real-time chat
  - Live notifications
  - Collaborative editing (Google Docs)
  - Live stock prices
  - Multiplayer games
```

---

**Q34. What is the difference between horizontal and vertical scaling?**
```
Vertical Scaling (Scale Up):
  - Add more CPU/RAM to existing server
  - Simple but has hardware limits
  - Single point of failure

Horizontal Scaling (Scale Out):
  - Add more servers
  - No theoretical limit
  - Requires load balancer
  - More complex (distributed state, session management)
```

---

**Q35. What is a design pattern? Name a few common ones.**
```
Creational:
  Singleton   -- only one instance of a class exists
  Factory     -- create objects without specifying exact class
  Builder     -- construct complex objects step by step

Structural:
  Decorator   -- add behavior to objects dynamically (Python @decorator)
  Adapter     -- make incompatible interfaces work together
  Proxy       -- control access to another object

Behavioral:
  Observer    -- notify multiple objects when state changes (event system)
  Strategy    -- swap algorithms at runtime
  Command     -- encapsulate requests as objects
```

---

**Q36. What is SOLID principles?**
```
S -- Single Responsibility: a class should have only one reason to change
O -- Open/Closed: open for extension, closed for modification
L -- Liskov Substitution: subclasses should be replaceable for their parent
I -- Interface Segregation: don't force classes to implement unused methods
D -- Dependency Inversion: depend on abstractions, not concrete implementations
```

---

**Q37. What is the difference between an abstract class and an interface?**
```python
from abc import ABC, abstractmethod

# Abstract class -- can have both abstract and concrete methods
class Animal(ABC):
    @abstractmethod
    def speak(self):  # must be implemented by subclass
        pass

    def breathe(self):  # concrete method, inherited as-is
        return "inhale/exhale"

class Dog(Animal):
    def speak(self):
        return "Woof"

# Python has no formal "interface" keyword
# Use ABC with all abstract methods to simulate an interface
```

---

**Q38. What is recursion? What is the base case?**
```python
def factorial(n):
    if n == 0:      # base case -- stops recursion
        return 1
    return n * factorial(n - 1)  # recursive case

# Without base case = infinite recursion = RecursionError
# Python default recursion limit: sys.getrecursionlimit() = 1000
```

---

**Q39. What is Big O notation?**
```
O(1)       -- constant time (dict lookup, array index)
O(log n)   -- logarithmic (binary search)
O(n)       -- linear (loop through list)
O(n log n) -- linearithmic (merge sort, quicksort average)
O(n²)      -- quadratic (nested loops)
O(2^n)     -- exponential (recursive fibonacci without memoization)

Rule: always analyze worst case unless stated otherwise
```

---

**Q40. What is the difference between a stack and a queue?**
```
Stack -- LIFO (Last In, First Out)
  push() -- add to top
  pop()  -- remove from top
  Use: function call stack, undo/redo, DFS

Queue -- FIFO (First In, First Out)
  enqueue() -- add to back
  dequeue() -- remove from front
  Use: task scheduling, BFS, print queue

Python:
  Stack: use list with .append() and .pop()
  Queue: use collections.deque with .append() and .popleft()
```

---

**Q41. What is the difference between compiled and interpreted languages?**
```
Compiled (C, C++, Go, Rust):
  - Source code -> machine code before execution
  - Faster at runtime
  - Platform-specific binary

Interpreted (Python, JavaScript, Ruby):
  - Source code executed line by line at runtime
  - Slower but more flexible
  - Platform-independent

Python is technically compiled to bytecode (.pyc) then interpreted by CPython VM.
```

---

**Q42. What is version control and why is it important?**
Version control (Git) tracks changes to code over time.
```
Benefits:
  - Revert to any previous state
  - Collaborate without overwriting each other's work
  - Branching for features/experiments
  - Audit trail of who changed what and why

Key commands:
  git init, git clone, git add, git commit, git push, git pull
  git branch, git checkout, git merge, git rebase
  git log, git diff, git stash
```

---

**Q43. What is the difference between `==` and `.equals()` in Java / `==` in Python?**
```
Java:
  == checks reference equality (same object in memory)
  .equals() checks value equality

Python:
  == checks value equality (calls __eq__)
  is checks reference equality (same object)

This is a common trick question -- know the language you're being interviewed for!
```

---

**Q44. What is an environment variable and why use it?**
```python
import os

# Store sensitive config outside code
DB_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

# .env file (never commit to git!)
# DATABASE_URL=postgresql://user:pass@localhost/mydb
# OPENAI_API_KEY=sk-...

# Load with python-dotenv
from dotenv import load_dotenv
load_dotenv()
```
> Never hardcode secrets in code. Use environment variables or secret managers.

---

**Q45. What is the difference between `PUT` and `PATCH`?**
```
PUT   -- replace the ENTIRE resource
PATCH -- update ONLY the specified fields

Example:
User: {name: "Sujit", age: 21, city: "Pune"}

PUT /users/1 with {name: "Sujit", age: 22}
Result: {name: "Sujit", age: 22}  -- city is GONE

PATCH /users/1 with {age: 22}
Result: {name: "Sujit", age: 22, city: "Pune"}  -- only age changed
```

---

**Q46. What is rate limiting?**
Rate limiting restricts how many requests a client can make in a given time window.
```
Purpose:
  - Prevent abuse and DDoS attacks
  - Ensure fair usage
  - Protect server resources

Common strategies:
  Fixed Window:  100 requests per minute, counter resets every minute
  Sliding Window: 100 requests in any 60-second window
  Token Bucket:  tokens refill at fixed rate, each request uses one token

Response: 429 Too Many Requests
Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
```

---

**Q47. What is the difference between `async/await` and threads?**
```
Threads:
  - OS-level concurrency
  - True parallelism (on multi-core for CPU-bound)
  - Higher memory overhead
  - Race conditions possible

async/await:
  - Single-threaded concurrency (event loop)
  - No true parallelism -- cooperative multitasking
  - Very efficient for I/O-bound tasks
  - No race conditions (single thread)
  - Lower memory overhead

Rule: async for I/O-bound (API calls, DB), threads/multiprocessing for CPU-bound
```

---

**Q48. What is a webhook?**
A webhook is a way for one system to notify another system when an event happens, via an HTTP POST request.
```
Traditional polling:
  Your app: "Any new orders?" (every 5 seconds)
  Server: "No... No... No... Yes!"

Webhook (push model):
  Server: "Hey, new order just came in!" (sends POST to your URL)

Use cases:
  - Payment notifications (Stripe, Razorpay)
  - GitHub events (push, PR, issue)
  - Slack notifications
  - CI/CD triggers
```

---

**Q49. What is the difference between monorepo and polyrepo?**
```
Monorepo -- all projects in one repository
  Pros: easy code sharing, atomic commits across projects, single CI/CD
  Cons: large repo size, complex tooling, slower CI
  Examples: Google, Meta, Nx, Turborepo

Polyrepo -- each project in its own repository
  Pros: independent deployments, smaller repos, team autonomy
  Cons: code sharing is harder, dependency management complex
  Examples: most microservices setups
```

---

**Q50. What is technical debt?**
Technical debt is the implied cost of future rework caused by choosing a quick/easy solution now instead of a better approach that would take longer.
```
Types:
  Deliberate:  "We know this is a shortcut, we'll fix it later"
  Accidental:  "We didn't know better at the time"

Signs of technical debt:
  - Hard to add new features without breaking things
  - No tests, so every change is risky
  - Duplicated code everywhere
  - No documentation
  - Slow build/deploy times

Managing it:
  - Refactor regularly (don't let it accumulate)
  - Write tests before refactoring
  - Document decisions (Architecture Decision Records)
  - Allocate 20% of sprint time to debt reduction
```

---

> **Quick Revision Summary:**
> Python basics traps: mutable defaults, `is` vs `==`, shallow vs deep copy, GIL, generators, closures, decorators.
> Software engineering must-knows: HTTP methods + status codes, REST principles, SQL vs NoSQL, ACID, JWT, OAuth, CORS, Docker, CI/CD, SOLID, Big O, async vs threads.
> These questions come up in every interview regardless of role. Know them cold.


---

## 23. DSA Patterns - Interview Must-Know

> These are the patterns that cover 80% of coding interview questions. Know the pattern, explain it clearly, then code it.

---

### Q1. What is the Two Pointers pattern? When do you use it?

Two Pointers is a technique where you use two index variables that move through a data structure — usually from both ends toward the middle, or both moving in the same direction at different speeds.

You use it when:
- The array or string is sorted (or can be sorted)
- You need to find a pair that satisfies some condition (sum, difference)
- You need to remove duplicates or partition elements
- You need to check if a string is a palindrome

Why it works: instead of checking every pair with a nested loop (O(n²)), two pointers reduce it to a single pass O(n) because the sorted order tells you which direction to move.

```python
# Find two numbers that sum to target in a sorted array
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            return [left, right]
        elif s < target:
            left += 1   # sum too small, move left pointer right
        else:
            right -= 1  # sum too big, move right pointer left
    return []

# Check palindrome
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

---

### Q2. What is the Sliding Window pattern? When do you use it?

Sliding Window is a technique where you maintain a "window" (a contiguous subarray or substring) and slide it across the data instead of recomputing from scratch each time.

There are two types:
- Fixed size window: window size k is given. You add one element on the right and remove one on the left as you slide.
- Variable size window: you expand the window until a condition breaks, then shrink from the left.

You use it when:
- The problem asks for max/min/sum of a subarray of size k
- The problem asks for the longest/shortest subarray or substring satisfying some condition
- Brute force would be O(n²) nested loops — sliding window brings it to O(n)

```python
# Fixed window: max sum subarray of size k
def max_sum_subarray(nums, k):
    window_sum = sum(nums[:k])  # compute first window
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]  # slide: add new, remove old
        max_sum = max(max_sum, window_sum)
    return max_sum

# Variable window: longest substring without repeating characters
def length_of_longest_substring(s):
    char_set = set()
    left = max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:   # shrink window until no duplicate
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len
```

---

### Q3. What is the Fast and Slow Pointers pattern?

Fast and Slow Pointers (also called Floyd's Cycle Detection) uses two pointers that move at different speeds through a linked list or sequence. The slow pointer moves one step at a time, the fast pointer moves two steps.

Key insight: if there is a cycle, the fast pointer will eventually lap the slow pointer and they will meet inside the cycle. If there is no cycle, the fast pointer reaches the end.

You use it when:
- Detecting a cycle in a linked list
- Finding the middle of a linked list (when fast reaches end, slow is at middle)
- Finding the start of a cycle

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Detect cycle -- O(n) time, O(1) space (no extra set needed)
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next        # moves 1 step
        fast = fast.next.next   # moves 2 steps
        if slow == fast:        # they met -- cycle exists
            return True
    return False  # fast reached end -- no cycle

# Find middle of linked list
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # when fast is at end, slow is exactly at middle
```

---

### Q4. What is Binary Search? When do you use it beyond simple search?

Binary Search is an algorithm that finds a target in a sorted array by repeatedly halving the search space. Each comparison eliminates half the remaining elements, giving O(log n) time.

The key insight is: whenever you can answer "is the answer in the left half or right half?" based on a condition, you can use binary search.

Beyond simple search, you use it for:
- Finding the first or last occurrence of a value
- Finding the minimum value that satisfies a condition (search on answer space)
- Problems like "find the minimum capacity to ship packages in D days"

```python
# Classic binary search
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2  # avoids integer overflow
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1   # target is in right half
        else:
            right = mid - 1  # target is in left half
    return -1

# Find leftmost (first) position of target
def search_first(nums, target):
    left, right, result = 0, len(nums) - 1, -1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            result = mid
            right = mid - 1  # found it, but keep searching left for earlier occurrence
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result
```

---

### Q5. What is BFS? When do you use it over DFS?

BFS (Breadth-First Search) explores a graph level by level. It starts at a source node, visits all its neighbors first, then their neighbors, and so on. It uses a queue (FIFO).

DFS (Depth-First Search) explores as far as possible down one path before backtracking. It uses a stack (or recursion).

Use BFS when:
- You need the shortest path in an unweighted graph (BFS guarantees shortest path, DFS does not)
- Level-order traversal of a tree
- Finding all nodes within a certain distance

Use DFS when:
- You need to explore all possible paths (backtracking, permutations)
- Detecting cycles
- Topological sort
- When the solution is likely deep in the tree

```python
from collections import deque

# BFS -- shortest path in unweighted graph
def bfs_shortest_path(graph, start, end):
    queue = deque([(start, [start])])  # (current node, path so far)
    visited = set([start])
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path  # first time we reach end = shortest path
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []  # no path found

# DFS -- explore all paths (recursive)
def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
```

---

### Q6. What is Dynamic Programming? How do you identify a DP problem?

Dynamic Programming is a technique for solving problems by breaking them into overlapping subproblems, solving each subproblem once, and storing the result to avoid recomputation.

Two conditions must hold for DP to apply:
1. Optimal substructure: the optimal solution to the problem contains optimal solutions to its subproblems
2. Overlapping subproblems: the same subproblems are solved multiple times in a naive recursive approach

How to identify a DP problem:
- The question asks for maximum, minimum, count of ways, or whether something is possible
- The problem has choices at each step (take or skip, go left or right)
- Brute force recursion has exponential time but many repeated calls

Two approaches:
- Top-down (memoization): write recursion, cache results
- Bottom-up (tabulation): fill a table iteratively from base cases

```python
# Fibonacci -- classic DP example
# Naive recursion: O(2^n) -- recomputes same values
# With memoization: O(n)

from functools import lru_cache

@lru_cache(maxsize=None)  # top-down: cache every result
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

# Bottom-up tabulation
def fib_dp(n):
    if n <= 1: return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# 0/1 Knapsack -- classic DP
# dp[i][w] = max value using first i items with capacity w
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]  # option 1: skip item i
            if weights[i-1] <= w:
                # option 2: take item i
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    return dp[n][capacity]
```

---

### Q7. What is a Stack? What problems does it solve?

A stack is a data structure that follows LIFO (Last In, First Out). The last element added is the first one removed. Think of a stack of plates — you add and remove from the top.

Operations: push (add to top), pop (remove from top), peek (look at top without removing) — all O(1).

Stacks are used for:
- Matching brackets/parentheses (push open, pop on close)
- Undo/redo functionality
- Function call stack (how recursion works internally)
- Next greater element problems (monotonic stack)
- DFS iteratively

```python
# Valid parentheses -- classic stack problem
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:           # closing bracket
            top = stack.pop() if stack else '#'
            if mapping[char] != top:  # doesn't match
                return False
        else:
            stack.append(char)        # opening bracket, push
    return not stack  # valid if nothing left unmatched

# Next greater element -- monotonic stack
# For each element, find the next element that is greater
def next_greater(nums):
    result = [-1] * len(nums)
    stack = []  # stores indices of elements waiting for their next greater
    for i in range(len(nums)):
        # pop all elements smaller than current -- current is their next greater
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    return result
```

---

### Q8. What is a Heap? When do you use it?

A heap is a complete binary tree that satisfies the heap property:
- Min-heap: every parent is smaller than its children (root = minimum element)
- Max-heap: every parent is larger than its children (root = maximum element)

Key property: you can always get the min (or max) in O(1), and insert/delete in O(log n).

Python's `heapq` module implements a min-heap.

You use a heap when:
- You need to repeatedly find and remove the minimum or maximum element
- K largest or K smallest elements from a large dataset
- Merging K sorted lists
- Scheduling problems (always process the highest priority task next)

```python
import heapq

# Min heap -- heapq is always a min-heap in Python
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)
print(heapq.heappop(heap))  # 1 -- always pops the smallest

# Max heap trick: negate values
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -1)
print(-heapq.heappop(max_heap))  # 5 -- largest

# K largest elements -- O(n log k) instead of O(n log n) sort
def k_largest(nums, k):
    return heapq.nlargest(k, nums)

# Top K frequent elements
from collections import Counter
def top_k_frequent(nums, k):
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
```

---

### Q9. What is a Hash Map? Why is lookup O(1)?

A hash map (dict in Python) stores key-value pairs. When you insert a key, a hash function converts the key into an index in an underlying array. Lookup works the same way — hash the key, go directly to that index.

This gives O(1) average time for insert, lookup, and delete because you jump directly to the location instead of searching.

Collisions (two keys hash to same index) are handled by chaining (linked list at each slot) or open addressing. In the worst case (all keys collide), lookup degrades to O(n), but this is extremely rare with a good hash function.

You use hash maps when:
- You need fast lookup by key
- Counting frequencies
- Caching/memoization
- Grouping elements (group anagrams)
- Two-sum style problems (store complement, check if it exists)

```python
# Two Sum -- O(n) with hash map vs O(n²) brute force
def two_sum(nums, target):
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:       # O(1) lookup
            return [seen[complement], i]
        seen[num] = i
    return []

# Group anagrams -- same sorted characters = same group
def group_anagrams(strs):
    groups = {}
    for s in strs:
        key = tuple(sorted(s))       # "eat", "tea", "ate" all sort to ('a','e','t')
        groups.setdefault(key, []).append(s)
    return list(groups.values())

# Frequency count
from collections import Counter
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
count = Counter(words)
print(count.most_common(2))  # [('apple', 3), ('banana', 2)]
```

---

### Time Complexity Cheat Sheet
```
Data Structure         | Access  | Search  | Insert  | Delete
-----------------------|---------|---------|---------|--------
Array / List           | O(1)    | O(n)    | O(n)    | O(n)
Linked List            | O(n)    | O(n)    | O(1)*   | O(1)*
Hash Map / Set         | -       | O(1)    | O(1)    | O(1)
Binary Search Tree     | O(log n)| O(log n)| O(log n)| O(log n)
Heap (min/max)         | O(1)**  | O(n)    | O(log n)| O(log n)
Sorted Array           | O(1)    | O(log n)| O(n)    | O(n)

* at known position
** only for min or max

Algorithm              | Best    | Average | Worst
-----------------------|---------|---------|--------
Binary Search          | O(1)    | O(log n)| O(log n)
Bubble Sort            | O(n)    | O(n²)   | O(n²)
Merge Sort             | O(n log n)| O(n log n)| O(n log n)
Quick Sort             | O(n log n)| O(n log n)| O(n²)
Python sort (Timsort)  | O(n)    | O(n log n)| O(n log n)
```

---

---

### Pattern 1: Two Pointers
Use when: sorted array, find pairs, remove duplicates, palindrome check.
```python
# Find two numbers that sum to target
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            return [left, right]
        elif s < target:
            left += 1
        else:
            right -= 1
    return []

# Check if string is palindrome
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

---

### Pattern 2: Sliding Window
Use when: subarray/substring of fixed or variable size, max/min sum, longest substring.
```python
# Max sum subarray of size k
def max_sum_subarray(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum

# Longest substring without repeating characters
def length_of_longest_substring(s):
    char_set = set()
    left = max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len
```

---

### Pattern 3: Fast & Slow Pointers
Use when: linked list cycle detection, find middle of linked list.
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Detect cycle in linked list
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Find middle of linked list
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # slow is at middle
```

---

### Pattern 4: Binary Search
Use when: sorted array, search for value, find boundary condition.
```python
# Classic binary search
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2  # avoids overflow
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Find first position of target (leftmost)
def search_first(nums, target):
    left, right, result = 0, len(nums) - 1, -1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            result = mid
            right = mid - 1  # keep searching left
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result
```

---

### Pattern 5: BFS (Breadth-First Search)
Use when: shortest path, level-order traversal, minimum steps.
```python
from collections import deque

# BFS on a graph
def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result

# Shortest path in unweighted graph
def shortest_path(graph, start, end):
    queue = deque([(start, [start])])
    visited = set([start])
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []
```

---

### Pattern 6: DFS (Depth-First Search)
Use when: explore all paths, tree traversal, connected components, backtracking.
```python
# DFS on a graph (iterative)
def dfs(graph, start):
    visited = set()
    stack = [start]
    result = []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            stack.extend(graph[node])
    return result

# DFS recursive
def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
    return visited
```

---

### Pattern 7: Dynamic Programming
Use when: overlapping subproblems, optimal substructure, count ways, min/max cost.
```python
# Fibonacci with memoization (top-down)
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

# Fibonacci with tabulation (bottom-up)
def fib_dp(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# 0/1 Knapsack
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]  # don't take item i
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    return dp[n][capacity]

# Longest Common Subsequence
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

---

### Pattern 8: Stack-based Problems
Use when: matching brackets, next greater element, monotonic stack.
```python
# Valid parentheses
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    return not stack

# Next greater element
def next_greater(nums):
    result = [-1] * len(nums)
    stack = []  # stores indices
    for i in range(len(nums)):
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    return result
```

---

### Pattern 9: Heap / Priority Queue
Use when: k largest/smallest elements, merge k sorted lists, top-k frequent.
```python
import heapq

# K largest elements
def k_largest(nums, k):
    return heapq.nlargest(k, nums)

# K smallest elements
def k_smallest(nums, k):
    return heapq.nsmallest(k, nums)

# Top K frequent elements
from collections import Counter
def top_k_frequent(nums, k):
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# Min heap operations
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
print(heapq.heappop(heap))  # 1 (smallest)

# Max heap: negate values
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -3)
print(-heapq.heappop(max_heap))  # 5 (largest)
```

---

### Pattern 10: Hash Map Tricks
Use when: frequency count, two-sum, group anagrams, first unique character.
```python
# Two Sum (unsorted)
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Group anagrams
def group_anagrams(strs):
    groups = {}
    for s in strs:
        key = tuple(sorted(s))
        groups.setdefault(key, []).append(s)
    return list(groups.values())

# First non-repeating character
def first_unique(s):
    count = Counter(s)
    for i, c in enumerate(s):
        if count[c] == 1:
            return i
    return -1
```

---

### Time Complexity Cheat Sheet
```
Operation              | List    | Dict/Set | Sorted List | Heap
-----------------------|---------|----------|-------------|------
Access by index        | O(1)    | -        | O(1)        | -
Search                 | O(n)    | O(1)     | O(log n)    | O(n)
Insert at end          | O(1)    | O(1)     | O(log n)    | O(log n)
Insert at beginning    | O(n)    | -        | O(n)        | -
Delete by value        | O(n)    | O(1)     | O(log n)    | O(n)
Min/Max                | O(n)    | -        | O(1)        | O(1)

Sorting: O(n log n) -- Python's Timsort
Binary Search: O(log n) -- only on sorted data
```

---

---

## 24. System Design Basics - Interview Questions

> System design questions test whether you can think at scale. Even for junior roles, interviewers want to see structured thinking. Answer with definitions first, then reasoning, then design.

---

### Q1. How do you approach a system design question?

Most candidates jump straight into drawing boxes. Interviewers want to see structured thinking, not just a diagram.

The right approach:

Step 1 — Clarify requirements (2 minutes). Ask: What are the core features? How many users? What is the expected read/write ratio? Does it need to be real-time? What is the acceptable latency?

Step 2 — Estimate scale. Back-of-envelope math. If 1 million DAU each make 10 requests/day, that is about 115 requests/second. If each request stores 1KB, that is 10GB/day of storage. This tells you whether you need caching, sharding, or a CDN.

Step 3 — High-level design. Draw the main components: Client, Load Balancer, API Servers, Database, Cache. Keep it simple first.

Step 4 — Deep dive. Go into the database schema, API design, and the hardest part of the problem (usually consistency, scale, or real-time delivery).

Step 5 — Address bottlenecks. Where will the system break under load? How do you fix it?

---

### Q2. What is a Load Balancer and why do you need it?

A load balancer sits between the client and your servers. It distributes incoming requests across multiple servers so no single server gets overwhelmed.

Without a load balancer, all traffic hits one server. When that server goes down, the entire service goes down. With a load balancer, you can have 10 servers behind it — if one dies, the load balancer stops sending traffic to it and the other 9 handle the load.

Load balancing algorithms:
- Round Robin: requests go to each server in turn (1, 2, 3, 1, 2, 3...). Simple, works when all servers are equal.
- Least Connections: send to the server with the fewest active connections. Better when requests have varying processing times.
- IP Hash: the same client IP always goes to the same server. Used for sticky sessions (when the server stores session state).

Load balancers also do health checks — they ping each server every few seconds and remove unhealthy servers from the pool automatically.

---

### Q3. What is caching and when should you use it?

Caching is storing the result of an expensive operation in fast memory so future requests can be served without repeating the work.

The most common cache is Redis — an in-memory key-value store that can respond in microseconds, compared to milliseconds for a database query.

When to use caching:
- Data that is read frequently but changes rarely (user profile, product catalog)
- Expensive computations (LLM responses, complex DB aggregations)
- Session data (store user session in Redis instead of DB)

Cache strategies:
- Cache-aside (lazy loading): the application checks the cache first. On a miss, it reads from the DB and writes to the cache. Most common pattern.
- Write-through: every write goes to both cache and DB simultaneously. Cache is always fresh but writes are slower.
- Write-back: write to cache only, sync to DB later. Fastest writes but risk of data loss if cache crashes.

Cache invalidation is the hard part. When the underlying data changes, you need to either delete the cache entry (so the next read fetches fresh data) or update it. TTL (time-to-live) is a simple way to expire stale data automatically.

---

### Q4. What is the difference between SQL and NoSQL? When do you choose each?

SQL databases (PostgreSQL, MySQL) store data in structured tables with rows and columns. They enforce a fixed schema — every row must have the same columns. They support ACID transactions, which guarantee that operations are atomic, consistent, isolated, and durable. They are excellent for relational data where you need joins across tables.

NoSQL databases (MongoDB, Cassandra, Redis, DynamoDB) store data in flexible formats — documents, key-value pairs, wide columns, or graphs. They do not enforce a fixed schema, so you can store different fields in different documents. They scale horizontally more easily than SQL databases.

Choose SQL when:
- Your data is structured and relational (users, orders, products with foreign keys)
- You need ACID transactions (banking, payments — you cannot afford partial updates)
- Your schema is stable and well-defined

Choose NoSQL when:
- Your data is unstructured or semi-structured (user activity logs, JSON documents)
- You need to scale writes massively (Cassandra handles millions of writes/second)
- Your schema changes frequently (early-stage product with evolving data model)
- You need a specific data model (Redis for caching, Neo4j for graphs)

In practice, most production systems use both — SQL for core business data, Redis for caching, and maybe MongoDB or Cassandra for specific high-volume use cases.

---

### Q5. What is database sharding? What problems does it solve and create?

Sharding is splitting your database horizontally across multiple servers. Instead of one server holding all rows of a table, you split the rows across multiple servers called shards.

For example, users with IDs 1–1,000,000 go to shard 1, users 1,000,001–2,000,000 go to shard 2, and so on. Each shard is a separate database server with its own storage and compute.

Why you need it: a single database server has limits — disk space, RAM, CPU, and maximum connections. When your data grows beyond what one server can handle, you shard.

Problems sharding creates:
- Cross-shard queries are expensive. A query that joins data from two shards requires fetching from both and merging in the application layer.
- Rebalancing is painful. If one shard gets too much data, you need to move data to a new shard, which is complex and risky.
- Hot spots. If you shard by user ID range, a viral event might send all traffic to one shard. Hash-based sharding distributes more evenly.

Hash-based sharding: `shard = hash(user_id) % number_of_shards`. This distributes data evenly but makes range queries harder.

---

### Q6. What is CAP Theorem? Why does it matter?

CAP Theorem states that a distributed system can only guarantee two of these three properties at the same time:

Consistency means every read returns the most recent write. All nodes see the same data at the same time.

Availability means every request gets a response (not necessarily the most recent data).

Partition Tolerance means the system continues to work even when network failures cause some nodes to be unable to communicate with others.

The key insight is that network partitions always happen in distributed systems — cables fail, data centers lose connectivity. So partition tolerance is not optional. You must choose between consistency and availability when a partition occurs.

CP systems (choose consistency over availability): when a partition happens, the system refuses to respond rather than return stale data. MongoDB, HBase, ZooKeeper. Use for banking, inventory — where stale data causes real problems.

AP systems (choose availability over consistency): when a partition happens, the system responds with potentially stale data. Cassandra, DynamoDB, CouchDB. Use for social media feeds, product catalogs — where slightly stale data is acceptable.

---

### Q7. What is a message queue? Why use it instead of direct API calls?

A message queue is a buffer that sits between services. A producer sends a message to the queue, and a consumer reads from it and processes it. The producer and consumer do not need to be running at the same time or at the same speed.

Why use it instead of direct API calls:

Decoupling: if Service B is down, Service A can still send messages to the queue. When B comes back up, it processes the backlog. With direct API calls, A would fail immediately if B is down.

Handling traffic spikes: if 10,000 orders come in at once, the queue absorbs the spike. The order processing service works through them at its own pace instead of crashing under load.

Async processing: some tasks do not need to be done synchronously. Sending a welcome email after signup does not need to block the signup response. Put it in a queue and process it in the background.

Kafka is used for high-throughput event streaming — it retains messages for days and allows multiple consumers to read the same messages independently. Good for analytics pipelines, event sourcing.

RabbitMQ is a traditional message broker with complex routing rules. Good for task queues where each message should be processed by exactly one consumer.

---

### Q8. What is the difference between horizontal and vertical scaling?

Vertical scaling (scaling up) means adding more resources to your existing server — more CPU cores, more RAM, faster storage. It is simple because you do not change your application architecture. But it has a hard limit: there is only so much CPU and RAM you can add to one machine. It is also a single point of failure — if that one powerful server goes down, everything goes down.

Horizontal scaling (scaling out) means adding more servers and distributing the load across them. There is no theoretical limit — you can keep adding servers. It requires a load balancer to distribute traffic and means your application must be stateless (no session data stored on the server itself, because the next request might go to a different server).

Most modern systems start with vertical scaling (it is simpler) and move to horizontal scaling when they hit the limits of a single machine.

---

### Q9. What is a CDN and when do you use it?

A CDN (Content Delivery Network) is a globally distributed network of servers that caches and serves static content from the location closest to the user.

Without a CDN, a user in Mumbai requesting an image stored on a server in the US waits for the round trip across the ocean — maybe 200ms just for the network. With a CDN, that image is cached on a server in Mumbai, and the user gets it in 5ms.

CDNs are used for:
- Static assets: images, CSS, JavaScript files, fonts
- Video streaming: Netflix uses CDNs to serve video from servers near each viewer
- DDoS protection: CDN providers absorb attack traffic before it reaches your origin server

When you deploy a new version of a static file, you either use cache-busting (change the filename, e.g., `app.v2.js`) or invalidate the CDN cache manually.

---

### Q10. What is the difference between monolithic and microservices architecture?

A monolithic application is a single deployable unit where all features — user management, payments, notifications, search — live in one codebase and are deployed together.

Pros of monolith: simple to develop, test, and deploy initially. No network calls between components. Easy to debug because everything is in one place.

Cons of monolith: as the team and codebase grow, it becomes hard to change one part without breaking another. You must deploy the entire application to change one feature. One bug can crash everything. You cannot scale individual components — if the search feature needs more CPU, you must scale the entire application.

Microservices splits the application into small, independent services. Each service owns its own database, is deployed independently, and communicates with other services via APIs or message queues.

Pros of microservices: independent deployments, independent scaling, technology flexibility per service, fault isolation (one service crashing does not bring down others).

Cons of microservices: much more complex. Network calls between services add latency and failure points. Distributed tracing and debugging is harder. You need service discovery, API gateways, and more infrastructure.

The honest answer: start with a monolith. Move to microservices only when the monolith's pain points are real and specific, not hypothetical.

---

---

### How to Approach Any System Design Question
```
1. Clarify requirements (2 min)
   - Functional: what does it do?
   - Non-functional: scale, latency, availability?

2. Estimate scale (1 min)
   - Users: 1M DAU?
   - Requests: 1000 req/sec?
   - Storage: 1TB/year?

3. High-level design (5 min)
   - Draw boxes: Client > Load Balancer > API Servers > DB > Cache

4. Deep dive (10 min)
   - Database schema
   - API design
   - Bottlenecks and how to solve them

5. Wrap up
   - Monitoring, failure handling, future scaling
```

---

### Design a URL Shortener (like bit.ly)

**Requirements:**
- Given a long URL, return a short URL
- Redirect short URL to original
- 100M URLs created/day, 10B redirects/day

**Design:**
```
Client --> Load Balancer --> API Servers --> Cache (Redis) --> DB (PostgreSQL)

Schema:
  urls table: id (BIGINT), short_code (VARCHAR 7), long_url (TEXT), created_at

Short code generation:
  Option 1: Base62 encode auto-increment ID (a-z, A-Z, 0-9 = 62 chars)
  Option 2: MD5 hash of long URL, take first 7 chars

API:
  POST /shorten  { "url": "https://very-long-url.com" } --> { "short": "abc1234" }
  GET  /{code}   --> 301 redirect to long URL

Caching:
  Cache short_code -> long_url in Redis (TTL 24h)
  Cache hit rate ~80% for popular URLs

DB:
  Index on short_code for O(log n) lookup
  Read replicas for 10B redirects/day
```

---

### Design a Chat Application (like WhatsApp)

**Key components:**
```
Client --> WebSocket Server --> Message Queue (Kafka) --> DB (Cassandra)
                                                      --> Push Notification Service

Why WebSocket?
  HTTP is request-response. Chat needs real-time bidirectional communication.
  WebSocket keeps connection open, server can push messages anytime.

Why Cassandra?
  Write-heavy (millions of messages/sec)
  Scales horizontally
  Time-series data (messages ordered by timestamp)

Message delivery:
  Online user:  deliver via WebSocket directly
  Offline user: store in DB, send push notification (FCM/APNs)

Schema:
  messages: message_id, chat_id, sender_id, content, timestamp, status
  Index on (chat_id, timestamp) for fetching chat history
```

---

### Design a Rate Limiter

**Algorithms:**
```
Token Bucket:
  - Bucket holds max N tokens
  - Tokens refill at fixed rate (e.g., 10/sec)
  - Each request consumes 1 token
  - If bucket empty: reject request
  - Allows bursts up to bucket size

Sliding Window Counter:
  - Track request count in rolling time window
  - More accurate than fixed window
  - Slightly more memory

Implementation with Redis:
  key = f"rate_limit:{user_id}"
  count = redis.incr(key)
  if count == 1:
      redis.expire(key, 60)  # 60 second window
  if count > 100:
      return 429 Too Many Requests
```

---

### Design a Notification System

```
Producers (events) --> Message Queue (Kafka) --> Notification Workers --> Channels

Channels:
  Push notification: FCM (Android), APNs (iOS)
  Email: SendGrid, AWS SES
  SMS: Twilio
  In-app: WebSocket or polling

Why Kafka?
  Decouples event producers from notification senders
  Handles traffic spikes (buffer)
  Retry failed notifications
  Multiple consumers (email + push + SMS all consume same event)

Schema:
  notifications: id, user_id, type, title, body, status, created_at
  user_preferences: user_id, email_enabled, push_enabled, sms_enabled
```

---

### Key System Design Concepts

**CAP Theorem:**
```
In a distributed system, you can only guarantee 2 of 3:
  C -- Consistency: all nodes see same data at same time
  A -- Availability: every request gets a response
  P -- Partition Tolerance: system works despite network failures

Network partitions always happen in distributed systems, so you choose:
  CP: consistent + partition tolerant (MongoDB, HBase) -- may reject requests
  AP: available + partition tolerant (Cassandra, DynamoDB) -- may return stale data
```

**Database Sharding:**
```
Sharding = splitting data across multiple DB servers (horizontal partitioning)

Strategies:
  Range-based:  users 1-1M on shard 1, 1M-2M on shard 2
  Hash-based:   shard = hash(user_id) % num_shards (even distribution)
  Directory:    lookup table maps key to shard

Problems:
  Hot spots: one shard gets all traffic
  Cross-shard queries: expensive joins
  Rebalancing: adding shards requires data migration
```

**Consistent Hashing:**
```
Used in distributed caches and load balancers.
Servers and keys are placed on a virtual ring.
Each key goes to the nearest server clockwise.
Adding/removing a server only affects neighboring keys (not all keys).
Used by: Amazon DynamoDB, Apache Cassandra, Memcached
```

**Read Replicas:**
```
Primary DB handles writes.
Read replicas handle reads (can have many).
Replication lag: replicas may be slightly behind primary.
Use for: read-heavy workloads, analytics, reporting.
```

**Event-Driven Architecture:**
```
Services communicate via events (messages) instead of direct API calls.
Producer emits event --> Message Queue --> Consumer processes it

Benefits:
  Loose coupling -- services don't know about each other
  Scalability -- consumers can scale independently
  Resilience -- queue buffers traffic spikes

Tools: Kafka, RabbitMQ, AWS SQS/SNS
```

---

---

## 25. Python OOP - Deep Dive

> OOP questions come up in almost every Python interview. Know the concept, explain it clearly, then show the code.

---

### Q1. What are the four pillars of OOP?

Encapsulation means bundling data (attributes) and the methods that operate on that data into a single unit (a class), and restricting direct access to some of the object's components. In Python, you signal private attributes with a leading underscore (`_name`) or double underscore (`__name`). The idea is that the internal state of an object should only be changed through its own methods, not from outside.

Abstraction means hiding complex implementation details and exposing only what is necessary. You interact with a car through the steering wheel and pedals — you do not need to know how the engine works internally. In Python, abstract base classes (ABC) enforce this by requiring subclasses to implement certain methods.

Inheritance means a class (child) can inherit attributes and methods from another class (parent), allowing code reuse and establishing an "is-a" relationship. A Dog is an Animal. A Dog inherits the `breathe()` method from Animal and adds its own `speak()` method.

Polymorphism means the same interface can be used for different underlying types. You can call `speak()` on a Dog and a Cat — both respond, but differently. This lets you write code that works with any Animal without knowing the specific type.

```python
class Animal:
    def __init__(self, name):
        self.name = name          # encapsulation: data inside the class

    def speak(self):              # abstraction: subclasses must define this
        raise NotImplementedError

class Dog(Animal):                # inheritance: Dog IS-A Animal
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

# Polymorphism: same call, different behavior
for animal in [Dog("Rex"), Cat("Whiskers")]:
    print(f"{animal.name}: {animal.speak()}")
```

---

### Q2. What is the difference between inheritance and composition? Which is better?

Inheritance models an "is-a" relationship. A Dog is an Animal. The child class gets all the parent's behavior and can override or extend it.

Composition models a "has-a" relationship. A Car has an Engine. Instead of inheriting from Engine, the Car class holds an Engine object as an attribute and delegates to it.

Composition is generally preferred because:
- It is more flexible. You can swap components at runtime (give the car a different engine).
- It avoids the fragile base class problem — changes to a parent class can unexpectedly break all child classes.
- Deep inheritance chains (A → B → C → D) become very hard to understand and maintain.
- It is easier to test because you can mock individual components.

The rule of thumb: use inheritance when the relationship is truly "is-a" and the child genuinely needs all of the parent's behavior. Use composition for everything else.

```python
# Inheritance -- tight coupling, hard to change Engine behavior
class ElectricCar(Engine):  # wrong -- a car is not an engine
    pass

# Composition -- loose coupling, flexible
class Engine:
    def start(self): return "Engine started"

class Car:
    def __init__(self, engine: Engine):
        self.engine = engine  # Car HAS-A Engine

    def start(self):
        return self.engine.start()  # delegates to engine

# Now you can swap engines without changing Car
car = Car(engine=ElectricEngine())
```

---

### Q3. What is Method Resolution Order (MRO) in Python?

MRO is the order in which Python searches for a method when you call it on an object. It matters in multiple inheritance — when a class inherits from two parents that both have the same method, Python needs to know which one to use.

Python uses the C3 linearization algorithm to compute MRO. The rule is: left to right, depth first, but a class is never visited before all its subclasses have been visited.

You can see the MRO of any class with `ClassName.__mro__` or `ClassName.mro()`.

```python
class A:
    def hello(self): return "A"

class B(A):
    def hello(self): return "B"

class C(A):
    def hello(self): return "C"

class D(B, C):  # inherits from both B and C
    pass

print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
# Python searches: D -> B -> C -> A -> object
# So D().hello() returns "B" (found in B before C)
```

The practical takeaway: when you have multiple inheritance, always check the MRO to understand which method will actually be called. Use `super()` to properly chain calls through the MRO instead of calling parent methods directly.

---

### Q4. What is the Singleton pattern? When and why do you use it?

The Singleton pattern ensures that a class has only one instance throughout the entire application, and provides a global point of access to it.

You use it when exactly one object is needed to coordinate actions across the system — a database connection pool, a configuration manager, a logger, or a cache client. Creating multiple instances of these would waste resources or cause inconsistency.

In Python, you implement it by overriding `__new__`, which is called before `__init__` and is responsible for creating the object. If an instance already exists, return it instead of creating a new one.

```python
class DatabasePool:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.connections = []  # initialize once
        return cls._instance  # always return the same object

pool1 = DatabasePool()
pool2 = DatabasePool()
print(pool1 is pool2)  # True -- same object, not two separate pools
```

Be careful with Singletons in multi-threaded code — two threads could both check `_instance is None` simultaneously and both create an instance. Use a threading lock to prevent this.

---

### Q5. What is an Abstract Base Class? How is it different from a regular class?

An Abstract Base Class (ABC) is a class that cannot be instantiated directly. It exists only to be subclassed. It defines a contract — a set of methods that every subclass must implement. If a subclass does not implement all abstract methods, Python raises a `TypeError` when you try to instantiate it.

The difference from a regular class: a regular class can be instantiated even if it has methods that do nothing. An ABC enforces that subclasses provide real implementations.

Why use ABCs: they make your code's intent explicit. When you see a class inheriting from an ABC, you immediately know what interface it must provide. It also catches missing implementations at class definition time rather than at runtime when the method is actually called.

```python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    @abstractmethod
    def charge(self, amount: float) -> bool:
        pass  # no implementation here -- subclass MUST provide it

    @abstractmethod
    def refund(self, amount: float) -> bool:
        pass

    def log(self, message: str):  # concrete method -- shared by all subclasses
        print(f"[Payment] {message}")

class StripeProcessor(PaymentProcessor):
    def charge(self, amount):
        self.log(f"Charging ${amount} via Stripe")
        return True

    def refund(self, amount):
        self.log(f"Refunding ${amount} via Stripe")
        return True

# PaymentProcessor()  # TypeError: Can't instantiate abstract class
p = StripeProcessor()  # works fine -- all abstract methods implemented
```

---

### Q6. What is the Singleton vs Factory pattern?

Singleton ensures one instance exists. Factory is about creating objects without specifying the exact class.

The Factory pattern defines an interface for creating objects but lets subclasses or configuration decide which class to instantiate. This is useful when the exact type of object to create depends on runtime conditions.

```python
# Factory pattern -- create the right LLM client based on config
class LLMFactory:
    @staticmethod
    def create(provider: str):
        if provider == "openai":
            return OpenAIClient()
        elif provider == "groq":
            return GroqClient()
        elif provider == "anthropic":
            return AnthropicClient()
        else:
            raise ValueError(f"Unknown provider: {provider}")

# Caller does not need to know which class is being created
llm = LLMFactory.create("groq")
response = llm.chat("What is RAG?")
```

The benefit: if you add a new provider, you only change the factory. All the code that calls `LLMFactory.create()` stays the same.

---

---

### Inheritance
```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement speak()")

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

# Polymorphism -- same interface, different behavior
animals = [Dog("Rex"), Cat("Whiskers")]
for animal in animals:
    print(f"{animal}: {animal.speak()}")
```

---

### Multiple Inheritance & MRO
```python
class A:
    def hello(self):
        return "A"

class B(A):
    def hello(self):
        return "B"

class C(A):
    def hello(self):
        return "C"

class D(B, C):
    pass

d = D()
print(d.hello())       # "B" -- follows MRO
print(D.__mro__)       # D -> B -> C -> A -> object
# MRO = Method Resolution Order, determined by C3 linearization
# Python searches left to right, depth first, but respects the order
```

---

### Composition vs Inheritance
```python
# Inheritance: "is-a" relationship
class ElectricCar(Car):  # ElectricCar IS-A Car
    pass

# Composition: "has-a" relationship (preferred in most cases)
class Engine:
    def start(self):
        return "Engine started"

class Car:
    def __init__(self):
        self.engine = Engine()  # Car HAS-A Engine

    def start(self):
        return self.engine.start()

# Prefer composition over inheritance:
# - More flexible (swap components at runtime)
# - Avoids deep inheritance chains
# - Easier to test (mock components)
```

---

### Singleton Pattern
```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True -- same object

# Thread-safe singleton
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
```

---

### Abstract Base Classes
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

    def describe(self):  # concrete method
        return f"Area: {self.area():.2f}, Perimeter: {self.perimeter():.2f}"

class Rectangle(Shape):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h

    def perimeter(self):
        return 2 * (self.w + self.h)

# Shape()  # TypeError: Can't instantiate abstract class
r = Rectangle(4, 5)
print(r.describe())  # Area: 20.00, Perimeter: 18.00
```

---

### Dunder Methods - Practical Examples
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):       # v1 + v2
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):      # v * 3
        return Vector(self.x * scalar, self.y * scalar)

    def __len__(self):              # len(v)
        return 2

    def __getitem__(self, idx):     # v[0], v[1]
        return (self.x, self.y)[idx]

    def __iter__(self):             # for val in v
        yield self.x
        yield self.y

    def __eq__(self, other):        # v1 == v2
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)    # Vector(4, 6)
print(v1 * 3)     # Vector(3, 6)
print(list(v1))   # [1, 2]
```

---

### Metaclasses (Advanced)
```python
# A metaclass is the class of a class
# type is the default metaclass of all classes

class Meta(type):
    def __new__(mcs, name, bases, namespace):
        # Auto-add a class_name attribute to every class using this metaclass
        namespace['class_name'] = name
        return super().__new__(mcs, name, bases, namespace)

class MyClass(metaclass=Meta):
    pass

print(MyClass.class_name)  # "MyClass"

# Common use: enforce that subclasses implement certain methods
class InterfaceMeta(type):
    def __new__(mcs, name, bases, namespace):
        required = ['process', 'validate']
        for method in required:
            if name != 'Interface' and method not in namespace:
                raise TypeError(f"Class {name} must implement {method}()")
        return super().__new__(mcs, name, bases, namespace)
```

---

## 26. Python Concurrency - Interview Questions

---

### Q1. What is the difference between concurrency and parallelism?

Concurrency means dealing with multiple tasks at the same time — but not necessarily executing them simultaneously. Tasks take turns using the CPU. When one task is waiting (for a network response, a file read), another task runs. This is what asyncio does.

Parallelism means actually executing multiple tasks simultaneously on multiple CPU cores at the same time. This is what multiprocessing does.

The analogy: a single chef cooking multiple dishes by switching between them is concurrency. Multiple chefs each cooking their own dish simultaneously is parallelism.

---

### Q2. What is the difference between Threading, Multiprocessing, and AsyncIO in Python?

Threading creates multiple threads within a single process. Threads share the same memory space, which makes communication easy but requires careful synchronization to avoid race conditions. The problem in Python is the GIL — the Global Interpreter Lock prevents more than one thread from executing Python bytecode at the same time. This means threading does NOT give you true parallelism for CPU-bound tasks. However, the GIL is released during I/O operations, so threading works well for I/O-bound tasks like making API calls or reading files.

Multiprocessing creates multiple separate processes, each with its own Python interpreter and memory space. Because each process has its own GIL, they can truly run in parallel on multiple CPU cores. This is the right choice for CPU-bound tasks like image processing, number crunching, or running ML inference. The downside is higher memory overhead and more complex communication between processes.

AsyncIO is single-threaded concurrency using an event loop. You write `async` functions that can `await` I/O operations. When a coroutine hits an `await`, it yields control back to the event loop, which runs another coroutine. No threads, no processes — just one thread switching between tasks cooperatively. This is the most efficient approach for handling thousands of concurrent I/O operations (API calls, database queries) because there is no thread creation overhead and no risk of race conditions.

Rule of thumb:
- I/O-bound with many concurrent operations → AsyncIO
- I/O-bound with simpler code → Threading
- CPU-bound → Multiprocessing

---

### Q3. What is a race condition and how do you prevent it?

A race condition occurs when two or more threads access shared data simultaneously and the final result depends on the order of execution — which is unpredictable.

Classic example: two threads both read a counter (value = 5), both add 1, and both write back 6. The counter should be 7 but it is 6 because both threads read the old value before either wrote the new one.

You prevent race conditions with a Lock (mutex). A lock ensures only one thread can execute a critical section at a time. The other thread waits until the lock is released.

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:      # only one thread enters this block at a time
            counter += 1

threads = [threading.Thread(target=increment) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()
print(counter)  # always 500000 with lock, random without it
```

The `with lock:` syntax automatically acquires the lock on entry and releases it on exit, even if an exception occurs.

---

### Q4. What is asyncio and how does the event loop work?

AsyncIO is Python's built-in library for writing concurrent code using the async/await syntax. It is built around an event loop — a loop that continuously checks for tasks that are ready to run and executes them.

When you `await` something (like an HTTP request), the coroutine is suspended and the event loop picks up another coroutine that is ready to run. When the awaited operation completes, the original coroutine is resumed.

This is cooperative multitasking — tasks voluntarily yield control at `await` points. Unlike threads, there is no preemption (the OS forcibly switching tasks), so there are no race conditions on shared data.

```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()  # yields control while waiting for response

async def main():
    async with aiohttp.ClientSession() as session:
        # All 10 requests start almost simultaneously
        # Total time ≈ time of slowest request, not sum of all
        tasks = [fetch(session, f"https://api.example.com/{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
    return results

asyncio.run(main())
```

Without asyncio, 10 sequential requests taking 1 second each = 10 seconds total. With asyncio, all 10 run concurrently = ~1 second total.

---
```
Threading:
  - Multiple threads in ONE process
  - Shared memory (careful with race conditions)
  - GIL limits true parallelism for CPU-bound
  - Good for: I/O-bound tasks (file, network)

Multiprocessing:
  - Multiple processes, each with own memory
  - True parallelism (bypasses GIL)
  - Higher memory overhead
  - Good for: CPU-bound tasks (image processing, ML inference)

AsyncIO:
  - Single thread, event loop
  - Cooperative multitasking (tasks yield control)
  - No race conditions
  - Good for: many concurrent I/O operations (API calls, DB queries)
```

```python
# Threading example
import threading

def download(url):
    # simulate I/O
    import time; time.sleep(1)
    print(f"Downloaded {url}")

threads = [threading.Thread(target=download, args=(f"url_{i}",)) for i in range(5)]
for t in threads: t.start()
for t in threads: t.join()

# Multiprocessing example
from multiprocessing import Pool

def cpu_heavy(n):
    return sum(i**2 for i in range(n))

with Pool(processes=4) as pool:
    results = pool.map(cpu_heavy, [1000000, 2000000, 3000000, 4000000])

# AsyncIO example
import asyncio, aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, f"https://api.example.com/{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
    return results
```

---

### Race Conditions and Locks
```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:  # only one thread at a time
            counter += 1

threads = [threading.Thread(target=increment) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()
print(counter)  # 500000 (correct with lock, random without)
```

---

## 27. Common Python Interview Traps

> These are the "gotcha" questions interviewers use to catch people off guard. Know the trap, explain why it happens, then show the fix.

---

### Trap 1: Mutable Default Arguments

This is one of the most common Python bugs. When you use a mutable object (list, dict, set) as a default argument, Python creates it once when the function is defined — not each time the function is called. So all calls that use the default share the same object.

Why it happens: default argument values are evaluated at function definition time, not at call time. The list `[]` is created once and stored as part of the function object.

```python
# WRONG -- the list is created once and reused across all calls
def add_item(item, lst=[]):
    lst.append(item)
    return lst

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2]  -- unexpected! should be [2]
print(add_item(3))  # [1, 2, 3]

# CORRECT -- use None as sentinel, create fresh list inside
def add_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

---

### Trap 2: Late Binding in Closures

When a closure captures a variable from an outer scope, it captures the variable itself — not its value at the time the closure was created. So if the variable changes later, the closure sees the new value.

Why it happens: Python closures close over variables by reference, not by value. The lambda `lambda: i` does not capture the current value of `i` — it captures the variable `i`. By the time the lambdas are called, the loop has finished and `i` is 4.

```python
# WRONG -- all lambdas see the final value of i (which is 4)
funcs = [lambda: i for i in range(5)]
print([f() for f in funcs])  # [4, 4, 4, 4, 4]

# CORRECT -- capture the current value using a default argument
# Default arguments ARE evaluated at definition time
funcs = [lambda i=i: i for i in range(5)]
print([f() for f in funcs])  # [0, 1, 2, 3, 4]
```

---

### Trap 3: Modifying a List While Iterating Over It

When you remove elements from a list while iterating over it with a `for` loop, the loop's internal index keeps advancing even though the list is shrinking. This causes elements to be skipped.

Why it happens: Python's `for` loop uses an internal counter. When you remove an element at index 2, the element that was at index 3 shifts to index 2. But the counter advances to 3, so the element that shifted to 2 is never visited.

```python
# WRONG -- skips elements silently
lst = [1, 2, 3, 4, 5, 6]
for item in lst:
    if item % 2 == 0:
        lst.remove(item)
print(lst)  # [1, 3, 5] looks right but 4 was skipped

# CORRECT option 1: iterate over a copy
for item in lst[:]:
    if item % 2 == 0:
        lst.remove(item)

# CORRECT option 2: list comprehension (cleanest)
lst = [item for item in lst if item % 2 != 0]
```

---

### Trap 4: Integer Caching (Small Integer Interning)

Python caches (interns) small integers from -5 to 256. For these values, Python reuses the same object instead of creating a new one. So `is` returns `True` for small integers but `False` for larger ones.

Why it happens: small integers are used so frequently that creating new objects for them every time would be wasteful. Python pre-creates them at startup and reuses them.

```python
a = 256
b = 256
print(a is b)  # True  -- same cached object

a = 257
b = 257
print(a is b)  # False -- different objects (outside cache range)

# The lesson: NEVER use `is` to compare integers or strings
# Always use == for value comparison
```

---

### Trap 5: String Concatenation in a Loop

Using `+=` to build a string in a loop creates a new string object on every iteration because strings are immutable. This is O(n²) — each concatenation copies all the characters accumulated so far.

Why it happens: strings are immutable in Python. `result += word` does not modify `result` in place — it creates a brand new string containing all the old characters plus the new ones, then rebinds `result` to the new string.

```python
# SLOW -- O(n²), creates a new string object every iteration
result = ""
for word in words:
    result += word

# FAST -- O(n), builds a list then joins once at the end
result = "".join(words)

# For large lists of strings, join() is dramatically faster
# 10,000 words: += takes ~50ms, join() takes ~0.5ms
```

---

### Trap 6: `==` vs `is` for None

You should always use `is None` and `is not None` to check for None, never `== None`.

Why: `==` calls the `__eq__` method, which can be overridden by a custom class to return `True` even when the object is not None. `is` checks identity — whether the object is literally the None singleton. Since there is only one None object in Python, `is None` is always correct and cannot be fooled.

```python
# Correct
if x is None:
    pass

if x is not None:
    pass

# Avoid -- can be overridden by custom __eq__
if x == None:
    pass
```

---

### Trap 7: `__eq__` Without `__hash__`

If you define `__eq__` in a class, Python automatically sets `__hash__` to `None`, making instances of that class unhashable. This means you cannot use them as dictionary keys or put them in a set.

Why it happens: Python's rule is that objects that compare equal must have the same hash. If you define custom equality, Python cannot guarantee this anymore, so it removes the default hash to prevent incorrect behavior.

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    # Python now sets __hash__ = None automatically

p = Point(1, 2)
{p}  # TypeError: unhashable type: 'Point'

# Fix: define __hash__ alongside __eq__
def __hash__(self):
    return hash((self.x, self.y))  # hash the tuple of values
```

---

### Trap 8: `is` vs `==` for Strings (String Interning)

Python automatically interns (reuses) short strings that look like identifiers. So `is` may return `True` for short strings but `False` for longer ones. This behavior is an implementation detail and should never be relied upon.

```python
a = "hello"
b = "hello"
print(a is b)  # True  -- Python interned this short string

a = "hello world"
b = "hello world"
print(a is b)  # May be False -- longer strings may not be interned

# Always use == for string comparison
print(a == b)  # True -- always correct
```

---

---

### Trap 1: Mutable Default Arguments
```python
# WRONG -- list is created ONCE and shared across all calls
def add_item(item, lst=[]):
    lst.append(item)
    return lst

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2]  <-- unexpected!

# CORRECT -- use None as default, create new list inside
def add_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

---

### Trap 2: Late Binding in Closures
```python
# WRONG -- all lambdas capture the same variable i
funcs = [lambda: i for i in range(5)]
print([f() for f in funcs])  # [4, 4, 4, 4, 4]  <-- all 4!

# CORRECT -- capture current value with default argument
funcs = [lambda i=i: i for i in range(5)]
print([f() for f in funcs])  # [0, 1, 2, 3, 4]
```

---

### Trap 3: Chained Comparisons
```python
# Python supports chained comparisons (unlike most languages)
x = 5
print(1 < x < 10)   # True  -- works as expected
print(1 < x > 3)    # True  -- also works

# This is NOT the same as:
print((1 < x) and (x < 10))  # same result but different syntax
```

---

### Trap 4: Integer Caching
```python
# Python caches small integers (-5 to 256)
a = 256
b = 256
print(a is b)  # True  -- same cached object

a = 257
b = 257
print(a is b)  # False -- different objects (outside cache range)
# This is why you should NEVER use `is` for integer comparison
```

---

### Trap 5: String Concatenation in Loop
```python
# SLOW -- creates new string object every iteration O(n²)
result = ""
for word in words:
    result += word

# FAST -- join at end O(n)
result = "".join(words)
```

---

### Trap 6: `is not None` vs `!= None`
```python
# Correct way to check for None
if x is None:      # checks identity (correct)
    pass

if x is not None:  # correct
    pass

# Avoid:
if x == None:      # calls __eq__, can be overridden by custom class
    pass
```

---

### Trap 7: Modifying a List While Iterating
```python
# WRONG -- skips elements
lst = [1, 2, 3, 4, 5]
for item in lst:
    if item % 2 == 0:
        lst.remove(item)  # modifies list during iteration!
print(lst)  # [1, 3, 5] -- looks right but 4 was skipped

# CORRECT -- iterate over a copy
for item in lst[:]:
    if item % 2 == 0:
        lst.remove(item)

# OR use list comprehension
lst = [item for item in lst if item % 2 != 0]
```

---

### Trap 8: `__eq__` without `__hash__`
```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    # If you define __eq__, Python sets __hash__ = None
    # This makes Point unhashable (can't use in set or as dict key)

p = Point(1, 2)
s = {p}  # TypeError: unhashable type: 'Point'

# Fix: also define __hash__
def __hash__(self):
    return hash((self.x, self.y))
```

---

## 28. Git - Interview Questions

---

### Q1. What is Git and why do we use it?

Git is a distributed version control system. It tracks every change made to your code over time, lets multiple developers work on the same codebase simultaneously without overwriting each other's work, and allows you to go back to any previous state of the code.

Distributed means every developer has a complete copy of the entire repository history on their machine — not just the latest version. You can commit, branch, and view history entirely offline. The remote (GitHub, GitLab) is just another copy that the team uses as the shared source of truth.

Without Git, teams would email zip files of code, overwrite each other's changes, and have no way to trace who changed what or why. Git solves all of this.

---

### Q2. What is the difference between `git merge` and `git rebase`?

Both combine changes from one branch into another, but they do it differently and produce different histories.

`git merge` creates a new "merge commit" that has two parents — the tip of both branches. The history is preserved exactly as it happened. You can see when branches diverged and when they came back together. This is safe for shared branches because it never rewrites history.

`git rebase` takes the commits from your branch and replays them on top of another branch, one by one. The result is a linear history — it looks like you developed your feature directly on top of the latest main branch, with no branching at all. This makes the history cleaner and easier to read, but it rewrites commit hashes, which is dangerous on shared branches.

Rule: use `merge` for shared branches (main, develop). Use `rebase` to clean up your local feature branch before merging it.

---

### Q3. What is the difference between `git reset` and `git revert`?

`git reset` moves the branch pointer backward to a previous commit. It rewrites history. The commits after the reset point are gone from the branch (though the changes may still be in your working directory depending on the flag used). Never use `git reset` on commits that have already been pushed to a shared branch — it will cause problems for everyone else.

`git revert` creates a new commit that undoes the changes of a specific commit. The original commit stays in the history. This is safe to use on shared branches because it does not rewrite history — it just adds a new commit that reverses the changes.

```bash
# Reset -- dangerous on shared branches, rewrites history
git reset HEAD~1          # undo last commit, keep changes in working directory
git reset --hard HEAD~1   # undo last commit, DISCARD all changes

# Revert -- safe, creates a new "undo" commit
git revert abc1234        # creates a new commit that undoes commit abc1234
```

---

### Q4. What is `git stash` and when do you use it?

`git stash` temporarily saves your uncommitted changes (both staged and unstaged) and reverts your working directory to the last commit. Your changes are stored in a stack and can be reapplied later.

You use it when you are in the middle of working on something and need to quickly switch to another branch (to fix a bug, review a PR) without committing half-done work.

```bash
git stash           # save current changes
git checkout main   # switch to another branch
# fix the bug, commit it
git checkout feature-branch
git stash pop       # restore your saved changes
```

---

### Q5. What is the difference between `git fetch` and `git pull`?

`git fetch` downloads changes from the remote repository but does NOT merge them into your local branch. Your working directory is unchanged. You can inspect what changed before deciding to merge.

`git pull` is `git fetch` followed by `git merge`. It downloads the changes and immediately merges them into your current branch.

`git fetch` is safer because it lets you review changes before integrating them. `git pull` is more convenient for simple workflows.

---

### Q6. What is a Git branch and why do we use branches?

A branch is a lightweight, movable pointer to a specific commit. Creating a branch does not copy any files — it just creates a new pointer. This makes branching in Git extremely fast and cheap.

Branches let you work on a new feature or bug fix in isolation without affecting the main codebase. When the work is done and tested, you merge the branch back into main. If the work is abandoned, you just delete the branch.

Common branching strategies:
- Feature branches: one branch per feature (`feature/user-auth`, `feature/payment`)
- Gitflow: separate branches for features, releases, and hotfixes
- Trunk-based development: everyone commits to main frequently, using feature flags to hide incomplete work

```bash
git checkout -b feature/user-auth   # create and switch to new branch
# do your work, commit
git checkout main
git merge feature/user-auth         # merge when done
git branch -d feature/user-auth     # delete the branch
```

---

### Quick Reference - Most Used Git Commands
```bash
git status                  # see what changed
git add file.py             # stage a specific file
git add .                   # stage all changes
git commit -m "message"     # commit with message
git push origin branch-name # push to remote
git pull origin main        # fetch + merge from remote
git log --oneline           # compact history
git diff                    # see unstaged changes
git diff --staged           # see staged changes
git stash / git stash pop   # save/restore uncommitted work
git reset HEAD~1            # undo last commit (keep changes)
git revert <hash>           # safely undo a pushed commit
```

---
# Setup
git config --global user.name "Sujit"
git config --global user.email "sujit@email.com"

# Basic workflow
git init                    # initialize repo
git clone <url>             # clone remote repo
git status                  # see changed files
git add file.py             # stage specific file
git add .                   # stage all changes
git commit -m "message"     # commit staged changes
git push origin main        # push to remote
git pull origin main        # fetch + merge from remote

# Branching
git branch feature-x        # create branch
git checkout feature-x      # switch to branch
git checkout -b feature-x   # create + switch in one step
git branch -d feature-x     # delete branch (safe)
git branch -D feature-x     # force delete branch

# Merging
git merge feature-x         # merge feature-x into current branch
git rebase main             # rebase current branch onto main

# Undoing
git restore file.py         # discard unstaged changes
git restore --staged file.py# unstage a file
git reset HEAD~1            # undo last commit, keep changes staged
git reset --hard HEAD~1     # undo last commit, DISCARD changes (destructive)
git revert <commit-hash>    # create new commit that undoes a commit (safe)

# Stashing
git stash                   # save uncommitted changes temporarily
git stash pop               # restore stashed changes
git stash list              # see all stashes

# Inspection
git log --oneline           # compact commit history
git log --graph --oneline   # visual branch graph
git diff                    # unstaged changes
git diff --staged           # staged changes
git blame file.py           # who changed each line

# Remote
git remote -v               # list remotes
git fetch origin            # download changes without merging
git remote add origin <url> # add remote

# Tags
git tag v1.0.0              # create tag
git push origin v1.0.0      # push tag
```

---

## 29. Linux / Terminal - Interview Questions

> Many interviewers ask basic Linux commands, especially for backend/DevOps roles. Know the command, know what it does and why.

---

### Q1. What is the difference between a process and a thread?

A process is an independent program running in its own memory space. Each process has its own heap, stack, and resources. Processes are isolated — one process crashing does not affect another.

A thread is a unit of execution within a process. Multiple threads share the same memory space of their parent process. This makes communication between threads fast (shared memory) but requires synchronization to avoid race conditions.

Creating a process is expensive (copy memory, set up resources). Creating a thread is cheap. This is why web servers use threads (or async) to handle concurrent requests rather than spawning a new process per request.

---

### Q2. What does `chmod 755` mean?

File permissions in Linux are represented as three groups of three bits: owner, group, others. Each group has read (4), write (2), execute (1) permissions.

`chmod 755` means:
- Owner: 7 = 4+2+1 = read + write + execute
- Group: 5 = 4+0+1 = read + execute (no write)
- Others: 5 = 4+0+1 = read + execute (no write)

This is the standard permission for executable scripts and web server directories — the owner can modify it, everyone else can read and run it.

`chmod +x script.sh` is a shortcut to add execute permission for everyone.

---

### Q3. What is the difference between `>` and `>>` in the terminal?

`>` redirects output to a file and overwrites it. If the file exists, its content is replaced.

`>>` redirects output to a file and appends to it. If the file exists, new content is added at the end.

```bash
echo "line 1" > file.txt   # file.txt now contains only "line 1"
echo "line 2" >> file.txt  # file.txt now contains "line 1" and "line 2"
```

`2>&1` redirects stderr (file descriptor 2) to wherever stdout (file descriptor 1) is going. Used to capture both normal output and error messages in the same place.

---

### Q4. What does `grep` do? What are its most useful flags?

`grep` searches for a pattern in files or input and prints matching lines. It is one of the most used commands for debugging and log analysis.

```bash
grep "error" app.log              # find lines containing "error"
grep -i "error" app.log           # case-insensitive search
grep -r "TODO" ./src              # recursive search in all files under ./src
grep -n "error" app.log           # show line numbers
grep -v "DEBUG" app.log           # show lines that do NOT match (invert)
grep -c "error" app.log           # count matching lines
grep "error\|warning" app.log     # match either "error" or "warning"
tail -f app.log | grep "ERROR"    # live log monitoring, filter for errors only
```

---

### Q5. What is `ps aux` and how do you kill a process?

`ps aux` lists all running processes on the system with their PID (Process ID), CPU usage, memory usage, and the command that started them.

`kill <PID>` sends a SIGTERM signal to the process, asking it to terminate gracefully. The process can catch this signal and do cleanup before exiting.

`kill -9 <PID>` sends SIGKILL, which immediately terminates the process with no chance for cleanup. Use this only when the process is unresponsive to SIGTERM.

```bash
ps aux | grep python      # find all Python processes
kill 12345                # graceful termination
kill -9 12345             # force kill
```

---

### Q6. What is `curl` and how do you use it to test an API?

`curl` is a command-line tool for making HTTP requests. It is the fastest way to test an API endpoint without writing code.

```bash
# GET request
curl https://api.example.com/users

# POST with JSON body
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Sujit", "email": "sujit@email.com"}'

# With auth header
curl -H "Authorization: Bearer your-token" https://api.example.com/profile

# See response headers
curl -I https://api.example.com/health

# Save response to file
curl -o response.json https://api.example.com/data
```

---

### Q7. What is `tail -f` and when do you use it?

`tail -f` follows a file in real time, printing new lines as they are added. It is the standard way to monitor live logs.

```bash
tail -f app.log                    # watch logs in real time
tail -f app.log | grep "ERROR"     # watch only error lines
tail -n 100 app.log                # show last 100 lines
```

You use it when debugging a running application — you can watch exactly what is happening as requests come in.

---

### Quick Reference - Most Used Linux Commands
```bash
# Navigation
pwd / ls -la / cd /path / cd ..

# Files
touch file.txt          # create file
mkdir -p dir/subdir     # create directory
cp src dest             # copy
mv src dest             # move/rename
rm -rf dir/             # delete directory (careful!)
cat file.txt            # print file
head -n 20 / tail -n 20 # first/last 20 lines

# Search
grep -r "pattern" ./    # search in files
find . -name "*.py"     # find files by name

# Processes
ps aux | grep python    # find process
kill -9 <PID>           # force kill

# Permissions
chmod 755 script.sh     # set permissions
chmod +x script.sh      # make executable

# Network
curl -X GET url         # HTTP request
ping google.com         # test connectivity
netstat -tulpn          # list open ports

# Disk/Memory
df -h                   # disk usage
free -h                 # memory usage
du -sh ./               # directory size
```

---

> **Summary of Sections 23-29:** These cover the complete picture for any software engineering interview — DSA patterns with explanations (not just code), system design with structured thinking, OOP pillars and patterns, concurrency models, Python gotchas with root cause explanations, Git concepts, and Linux fundamentals. Combined with the AI/ML sections, this is everything from fresher to experienced roles.


```bash
# Navigation
pwd                    # print working directory
ls -la                 # list files with details
cd /path/to/dir        # change directory
cd ..                  # go up one level
cd ~                   # go to home directory

# File operations
touch file.txt         # create empty file
mkdir -p dir/subdir    # create directory (and parents)
cp source dest         # copy file
mv source dest         # move/rename file
rm file.txt            # delete file
rm -rf dir/            # delete directory recursively (careful!)
cat file.txt           # print file content
head -n 20 file.txt    # first 20 lines
tail -n 20 file.txt    # last 20 lines
tail -f log.txt        # follow file in real time (logs)

# Search
grep "pattern" file.txt        # search in file
grep -r "pattern" ./           # recursive search
grep -i "pattern" file.txt     # case insensitive
find . -name "*.py"            # find files by name
find . -type f -size +1M       # find files larger than 1MB

# Process management
ps aux                 # list all processes
kill -9 <PID>          # force kill process
top                    # real-time process monitor
htop                   # better process monitor

# Permissions
chmod 755 script.sh    # rwxr-xr-x
chmod +x script.sh     # make executable
chown user:group file  # change owner

# Networking
curl https://api.example.com/data          # HTTP GET
curl -X POST -d '{"key":"val"}' url        # HTTP POST
wget https://example.com/file.zip          # download file
ping google.com                            # test connectivity
netstat -tulpn                             # list open ports
ss -tulpn                                  # modern netstat

# Environment
export KEY=value       # set env variable
echo $KEY              # print env variable
env                    # list all env variables
source .env            # load .env file into shell

# Disk & Memory
df -h                  # disk usage
du -sh ./              # directory size
free -h                # memory usage

# Pipes and redirection
cat file.txt | grep "error"        # pipe output to grep
command > output.txt               # redirect stdout to file
command >> output.txt              # append to file
command 2>&1 | tee log.txt         # redirect stderr + stdout, also print
```

---

> **Final Note:** These sections (23-29) cover DSA patterns, system design, OOP deep dive, concurrency, Python traps, Git, and Linux — the complete picture for any software engineering interview. Combined with the AI/ML sections earlier, this covers everything from fresher to experienced roles.
