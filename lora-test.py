from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. Configuration ---
# Adjust these paths to where you've saved your models.

# Path to your base model in GGUF format
MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Path to your LoRA adapter file (can be .bin or a GGUF LoRA)
# NOTE: Make sure the LoRA is compatible with your base model.
LORA_PATH = "./loras/python-expert-lora.gguf" # Fictional LoRA for this example

# --- 2. Initialize the LLMs ---

# A. Initialize the base model WITHOUT the LoRA
print("Initializing base model...")
llm_base = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, # Offload all layers to GPU
    n_batch=512,
    n_ctx=4096,      # Context window
    verbose=False,   # Set to True for more logs
)

# B. Initialize the same base model WITH the LoRA adapter
print("Initializing model with LoRA...")
llm_lora = LlamaCpp(
    model_path=MODEL_PATH,
    lora_path=LORA_PATH, # This is the key parameter!
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=4096,
    verbose=False,
)

# --- 3. Create a Simple Chain ---
# We'll use the same prompt and chain for both models.

template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)
chain = prompt | StrOutputParser()

# --- 4. Invoke and Compare ---
question = "Write a Python function to calculate the factorial of a number."

print("\n--- Invoking Base Model (Without LoRA) ---")
response_base = chain.invoke({"question": question}, config={"llm": llm_base})
print(response_base)


print("\n--- Invoking Fine-Tuned Model (With LoRA) ---")
response_lora = chain.invoke({"question": question}, config={"llm": llm_lora})
print(response_lora)