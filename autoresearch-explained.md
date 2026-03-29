# Autoresearch: A Complete Guide for Listening

## What Is Autoresearch?

Autoresearch is a project by Andrej Karpathy that turns an AI coding assistant into an autonomous machine learning researcher. The core idea is simple but powerful: you give an AI agent a small but real neural network training setup, tell it the rules, and then go to sleep. While you sleep, the agent runs experiments on its own. It changes the code, trains the model for five minutes, checks whether the change made things better or worse, keeps the good changes, throws away the bad ones, and repeats. When you wake up in the morning, you have a log of about a hundred experiments and, hopefully, a better model than what you started with.

Think of it like this. Imagine you had a very tireless research assistant who could try a new idea every five minutes, around the clock, without ever getting bored or needing a break. That is what autoresearch does.

---

## The Three Files That Matter

The entire project is deliberately kept tiny. Only three files matter, and understanding what each one does is the key to understanding the whole system.

### File One: prepare.py

This is the foundation file. It handles all the boring but essential setup work. It downloads training data from Hugging Face, which is a large collection of text documents stored as parquet files. It trains a byte pair encoding tokenizer, which is the tool that converts raw text into numbers that a neural network can understand. And it provides the evaluation function that measures how good the model is.

The critical thing about prepare.py is that nobody is allowed to change it. Not the human, not the AI agent. It is read-only. This is intentional. By keeping the data pipeline and the evaluation metric fixed, every experiment is measured on exactly the same playing field. If the agent were allowed to change the evaluation function, it could cheat by making the metric easier rather than making the model better.

Prepare.py defines three important constants. MAX_SEQ_LEN is set to 2048, which is the context length, meaning how many tokens the model can see at once. TIME_BUDGET is set to 300 seconds, which is five minutes. And EVAL_TOKENS defines how many tokens are used for evaluation.

It also provides three runtime utilities that train.py imports and uses. First, the Tokenizer class, which loads the pre-trained tokenizer and can convert text to token IDs and back. Second, the make_dataloader function, which creates an efficient data loading pipeline that packs documents together without any wasted padding. Third, the evaluate_bpb function, which is the single most important function in the entire project. This function measures the model's quality using a metric called bits per byte, which I will explain in detail later.

### File Two: train.py

This is the playground. This is the only file the AI agent is allowed to modify. It contains everything about the neural network itself: the model architecture, the optimizer, the hyperparameters, and the training loop.

The model is a GPT, a generative pre-trained transformer. In its default configuration, it has eight transformer layers, an embedding dimension of 512, six attention heads, and about fifty million parameters. It uses several modern techniques. It uses rotary position embeddings, sometimes called RoPE, which help the model understand the position of tokens in a sequence. It uses RMS normalization instead of the more traditional layer normalization. It uses a sliding window attention pattern where most layers only look at half the context, with the last layer always looking at the full context. It uses an activation function called ReLU squared, which is just the standard ReLU but with the output squared. And it uses value embeddings, a technique from a paper called ResFormer, where the model mixes in learned embeddings into the value vectors of attention, controlled by a learned gate.

The optimizer is a hybrid called MuonAdamW. For the large matrix parameters inside the transformer layers, it uses an optimizer called Muon, which applies a technique called polar express orthogonalization to the gradients. For everything else, like the token embeddings and the output projection, it uses the more standard AdamW optimizer. Different parameter groups get different learning rates, which is a common practice in modern deep learning.

The training loop itself is straightforward. It runs a while-true loop. Each iteration, it processes a batch of data, computes the loss, accumulates gradients across multiple micro-batches, applies the learning rate schedule, takes an optimizer step, and checks how much wall clock time has elapsed. Once five minutes of actual training time have passed, it stops. It then runs the evaluation function and prints a summary of results including the validation bits per byte, the peak GPU memory used, the number of steps completed, and other statistics.

The hyperparameters section at the top of train.py is where most of the agent's changes will happen. Things like the learning rate, the model depth, the batch size, the warmup and warmdown ratios for the learning rate schedule, the weight decay, and the aspect ratio which controls how wide the model is relative to its depth.

### File Three: program.md

This is the most conceptually interesting file. It is a plain markdown document that contains instructions for the AI agent. There is no Python orchestration code, no agent framework, no LangChain, no complex pipeline. The entire agent is just a language model reading a markdown file and following the instructions.

Program.md is divided into sections. The setup section tells the agent how to initialize a new experiment run: pick a date-based tag, create a git branch, read all the files, check that the data is downloaded, and create the results log file. The experimentation section explains what the agent can and cannot do. It can modify train.py in any way it wants. It cannot modify prepare.py, cannot install new packages, and cannot change the evaluation metric. The output format section shows the agent what the training script prints when it finishes, so the agent knows how to parse the results. The logging section explains the tab-separated results file format. And the experiment loop section describes the exact steps of each cycle, which I will walk through in detail next.

The human's job is to write and refine program.md. You are essentially programming the researcher, not the model. As Karpathy puts it, the "code" you write is the markdown instructions, and the "computer" that runs it is the AI agent.

---

## The Experiment Loop: Step by Step

Now let's walk through exactly what happens during one complete cycle of the experiment loop. Each cycle takes roughly five to six minutes.

### Step One: The Agent Inspects the Current State

The agent starts by looking at the current git state. It checks which branch it is on, what the latest commit is, and reviews the results so far. This gives it context about what has already been tried and what the current best validation score is.

### Step Two: The Agent Proposes an Experiment

This is where the AI's knowledge of deep learning comes in. Based on its understanding of neural network architectures, optimization, and the results of previous experiments, the agent comes up with an idea to try. This could be anything. It might increase the learning rate. It might add more transformer layers. It might change the activation function. It might switch the attention pattern. It might adjust the batch size to fit a larger model in memory. It might try a completely different architectural idea.

The agent then directly edits train.py to implement this change. It might change a single number, like bumping the learning rate from 0.04 to 0.06. Or it might rewrite entire sections of the model code, like replacing the standard multi-layer perceptron with a gated linear unit variant.

### Step Three: The Agent Commits the Change

Before running the experiment, the agent creates a git commit with a descriptive message like "increase learning rate to 0.06" or "replace ReLU squared with SwiGLU activation." This is important because it creates a clean snapshot that can be reverted if the experiment fails.

### Step Four: The Agent Runs the Training

The agent executes the command: uv run train.py, with all output redirected to a file called run.log. The redirect is important. If the output went directly to the agent's context window, the thousands of lines of training progress would flood the agent's memory and waste its context budget. By writing to a file, the agent can selectively read only the parts it needs.

Now the actual training happens. The script loads the tokenizer, builds the model according to whatever configuration the agent has set, initializes the weights, sets up the optimizer, compiles the model with torch.compile for performance, and starts the training loop.

During training, the model processes batches of tokenized text. Each batch goes through a forward pass where the model predicts the next token at every position, a loss computation using cross-entropy, and a backward pass that computes gradients. These gradients are accumulated across multiple micro-batches to simulate a larger effective batch size, and then the optimizer updates the model weights.

The learning rate follows a schedule. It can optionally warm up from zero at the beginning, stays constant in the middle, and then cools down during the last portion of training. The weight decay also decreases over time.

After exactly five minutes of training time have elapsed, the loop breaks. Note that the timer ignores the first ten steps, because those steps include the overhead of torch.compile actually compiling the model, which can take a significant amount of time but is a one-time cost.

Then the model is switched to evaluation mode, and the evaluate_bpb function is called.

### Step Five: The Agent Parses the Results

The agent runs the command: grep val_bpb run.log. This extracts the validation bits per byte number from the log file. It also extracts the peak VRAM usage. If grep returns nothing, that means the script crashed before reaching the evaluation stage.

If the run crashed, the agent reads the last fifty lines of the log file to see the error message. Depending on the error, it might fix the code and try again. For example, if it was an out-of-memory error, it might reduce the batch size or model size. If it was a typo, it fixes the typo. If the idea was fundamentally broken, it gives up on that particular experiment.

### Step Six: The Agent Logs the Result

Regardless of the outcome, the agent appends a row to results.tsv. This file is tab-separated and has five columns: the git commit hash, the validation bits per byte score, the peak memory usage in gigabytes, the status which is either keep or discard or crash, and a short text description of what the experiment tried.

Importantly, results.tsv is not tracked by git. It is listed in the gitignore file. This means it accumulates across all experiments without creating noisy commits.

### Step Seven: Keep or Revert

This is the critical decision point.

If the validation bits per byte improved, meaning it went down compared to the previous best, the agent keeps the commit. The branch advances. This new version of train.py becomes the starting point for the next experiment. All future experiments will build on top of this improvement.

If the validation bits per byte stayed the same or got worse, the agent reverts the commit using git reset. The code goes back to exactly how it was before this experiment. The failed idea is discarded, but the result is still logged in the TSV file so there is a record of what was tried.

There is also a simplicity criterion. The instructions in program.md tell the agent that if a change gives only a tiny improvement but adds a lot of ugly complexity, it is probably not worth keeping. Conversely, if removing code gives equal or better results, that is a great outcome because it simplifies the codebase. This prevents the code from accumulating unnecessary complexity over many experiments.

### Step Eight: Repeat

The agent goes back to step one and does it all again. And again. And again. The instructions in program.md are explicit: never stop. Never ask the human if you should continue. The human might be asleep. Keep going until you are manually interrupted. If you run out of ideas, think harder.

At five minutes per experiment, the agent can run about twelve experiments per hour. Over an eight-hour night of sleep, that is roughly a hundred experiments, all automatically logged, all building on top of each other's successes.

---

## The Metric: Bits Per Byte

Let me explain the evaluation metric in more detail, because it is central to the whole system.

Bits per byte, or BPB, measures how well the model can predict the next piece of text, normalized by the number of raw bytes in the text. Here is how it works.

First, the model processes a large amount of validation text, about twenty million tokens worth. For each token, the model outputs a probability distribution over the entire vocabulary, and we compute the cross-entropy loss, which measures how surprised the model was by the actual next token. Cross-entropy is measured in nats, which is the natural logarithm unit.

Then, we look at each target token and count how many bytes of UTF-8 text it represents. A simple ASCII character like the letter "a" is one byte. A Chinese character might be three bytes. A special token like the beginning-of-sequence marker has zero bytes and is excluded from the calculation.

Finally, we divide the total cross-entropy in nats by the total number of bytes, and convert from nats to bits by dividing by the natural logarithm of two. The result is how many bits of information the model needs, on average, to encode one byte of text.

Why use bits per byte instead of the more common perplexity or cross-entropy loss? Because it is vocabulary-size-independent. If the agent decides to change the vocabulary size, say from 32,000 tokens to 16,000 tokens, the per-token cross-entropy would change even if the model is equally good. But bits per byte normalizes everything down to the raw byte level, so any architectural change, including vocabulary changes, can be fairly compared.

Lower bits per byte is better. A typical baseline might score around 1.0, meaning the model needs about one bit per byte of text. Improvements might bring it down to 0.99, 0.98, and so on. Each small reduction means the model has gotten measurably better at predicting text.

---

## The Model Architecture in Detail

Let me walk through the neural network architecture in train.py, since this is what the agent is iterating on.

### Token Embeddings

The first layer converts token IDs into dense vectors. Each token in the vocabulary gets a 768-dimensional vector, in the default configuration. These vectors are learned during training.

### Transformer Blocks

The core of the model is a stack of transformer blocks. By default, there are eight of them. Each block has two sub-layers: a causal self-attention layer and a multi-layer perceptron, sometimes called a feed-forward network.

In the attention layer, the input is projected into three sets of vectors: queries, keys, and values. The queries and keys are used to compute attention scores, which determine how much each position should attend to every other position. The values are then weighted by these attention scores and summed up to produce the output.

Several modern techniques are applied in the attention layer. Rotary position embeddings encode positional information directly into the query and key vectors using rotations in the complex plane. Query and key normalization applies RMS normalization to the query and key vectors before computing attention scores, which stabilizes training. Sliding window attention means that most layers only attend to the nearest 1024 tokens instead of all 2048, which reduces computation. The pattern "SSSL" means three short-window layers followed by one long-window layer, repeating. Value embeddings, from the ResFormer paper, add learned token-level embeddings to the value vectors, controlled by a small learned gate.

The attention computation itself uses Flash Attention 3, a highly optimized CUDA kernel that computes exact attention much faster than a naive implementation.

In the multi-layer perceptron layer, the input is projected to four times the model dimension, passed through the ReLU-squared activation function, and projected back down. ReLU-squared means taking the standard ReLU, which zeros out negative values, and then squaring the result. This tends to produce sparser activations compared to smooth activations like GELU.

Between the attention and MLP sub-layers, there are residual connections with learned per-layer scaling factors. The model learns two scalars per layer: one that scales the residual stream and one that mixes in the original input embedding. This is a technique that helps with training stability in deep models.

### Output Head

The final layer projects the transformer's output back to vocabulary size and applies a logit soft-capping at fifteen, which prevents any logit from becoming too extreme. The output is a probability distribution over the vocabulary for the next token prediction.

---

## The Optimizer in Detail

The optimizer is a hybrid of two algorithms, hence the name MuonAdamW.

### AdamW for Non-Matrix Parameters

For the token embeddings, the output projection (called lm_head), the value embeddings, and the per-layer scalar parameters, the optimizer uses AdamW. This is the standard adaptive optimizer used in most deep learning. It maintains a running average of the gradient and a running average of the squared gradient for each parameter. The learning rates for these different parameter groups are tuned separately. Token embeddings get a relatively high learning rate of 0.6. The output projection gets a much lower learning rate of 0.004. The per-layer scalars get a learning rate of 0.5.

### Muon for Matrix Parameters

For the large weight matrices inside the transformer blocks, like the query, key, value, and output projections in attention, and the two projections in the MLP, the optimizer uses Muon. Muon is a more exotic optimizer that applies polar decomposition to the gradients, effectively orthogonalizing them before taking the step. The idea is that orthogonalized updates are more efficient in high-dimensional parameter spaces. It also uses Nesterov momentum and a second-moment-based variance reduction technique called NorMuon.

Additionally, Muon applies cautious weight decay, where the weight decay term is masked so it only applies when the gradient and the parameter have the same sign. This prevents the regularization from fighting against the optimization direction.

---

## The Git Workflow

The git workflow is a crucial part of the system. It acts as a version control mechanism that enables the keep-or-revert decision.

At the start of a run, the agent creates a fresh branch from master, something like autoresearch/mar21. Every experiment that improves the score advances this branch with a new commit. Every experiment that fails gets reverted, so the branch only contains the chain of successful improvements.

After a night of experiments, you can look at the git log of the branch and see a clean history of every improvement that was made. You can look at the diff between any two commits to see exactly what change led to each improvement. And you can look at results.tsv to see all experiments, including the failures, giving you a complete picture of what was tried.

This is essentially hill climbing on the space of possible train.py modifications, where the fitness function is the validation bits per byte metric. The agent explores the landscape by making changes, evaluates each change, and only moves to the new position if it is better than where it was.

---

## What the Agent Is Not

It is worth being clear about what the agent is not, because the simplicity of the design can be surprising.

There is no reinforcement learning happening at the meta level. The agent does not learn from its past experiments in any formal sense. It does not update its own weights based on the outcomes. It simply reads the results log and uses its general knowledge of deep learning to propose the next experiment. Its "learning" comes from having the experiment history in its context window.

There is no search algorithm. There is no Bayesian optimization, no evolutionary strategy, no neural architecture search framework. The agent is just a language model using its judgment, the same way a human researcher would.

There is no agent framework. No LangChain, no AutoGPT, no complex orchestration pipeline. The agent is just an LLM following a markdown document, using the standard tool-use capabilities that modern coding assistants provide: running shell commands and editing files.

The elegance of autoresearch is precisely this simplicity. The "agent" is just a prompt. The "orchestration" is just a markdown file. The "evaluation" is just a Python function. And the "search strategy" is just a smart language model doing what it thinks is best.

---

## Why This Matters

Autoresearch represents a philosophical shift in how machine learning research could work. Traditionally, a human researcher comes up with an idea, implements it, runs the experiment, waits, analyzes the results, and then thinks about what to try next. This cycle might take hours or days per idea.

With autoresearch, the cycle is compressed to five minutes. And because the agent does not need to sleep, eat, or attend meetings, it can run continuously. A hundred experiments overnight is not an unreasonable estimate.

Of course, the agent is not as creative as a top human researcher. It is unlikely to come up with a truly novel architectural innovation. But it is excellent at systematic exploration: trying different hyperparameter combinations, testing small variations, finding the optimal learning rate, adjusting model sizes, and discovering which techniques help and which hurt. These are exactly the kinds of tedious, time-consuming experiments that human researchers often skip because they do not have enough GPU hours or patience.

The key insight is that you are not programming the model. You are programming the researcher. The quality of the experiments depends on the quality of the instructions in program.md. A more sophisticated program.md might tell the agent to read recent papers, to try specific techniques, to focus on certain aspects of the architecture, or to use more systematic exploration strategies. This is the layer where human creativity matters.

As Karpathy writes in the README, somewhat tongue-in-cheek: "One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of group meeting. That era is long gone."

Whether or not you take that literally, autoresearch is a concrete, working demonstration of what autonomous AI research looks like in practice. And it fits in three files.
