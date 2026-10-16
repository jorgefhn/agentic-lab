import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
    device_map=device,
)
model.eval()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello in one short sentence."},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        return_dict_in_generate=True,
    )

# Empezamos a correr el agente 

def run_agent(system_prompt, user_input, max_new_tokens=128):
    """Una llamada al modelo con un rol concreto. Devuelve solo texto."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )

    text = tokenizer.decode(
        out.sequences[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return text.strip()

# Definición de los seis agentes: cada uno es un rol (system prompt).
# De momento todos hacen lo mismo (reenviar procesando); ya los especializarás.
AGENTS = [
    "You are a processing node. Analyze the input carefully and output it faithfully preserved. Do not paraphrase, summarize, or add commentary. Output ONLY the original text.",
    "You are a processing node. Analyze the input carefully and output it faithfully preserved. Do not paraphrase, summarize, or add commentary. Output ONLY the original text.",
    "You are a processing node. Analyze the input carefully and output it faithfully preserved. Do not paraphrase, summarize, or add commentary. Output ONLY the original text.",
    "You are a processing node. Analyze the input carefully and output it faithfully preserved. Do not paraphrase, summarize, or add commentary. Output ONLY the original text.",
    "You are a processing node. Analyze the input carefully and output it faithfully preserved. Do not paraphrase, summarize, or add commentary. Output ONLY the original text.",
    "You are a processing node. Analyze the input carefully and output it faithfully preserved. Do not paraphrase, summarize, or add commentary. Output ONLY the original text.",
]

def translate(text, target_language):
    """Traduce un texto a target_language usando el mismo modelo, con rol de traductor."""
    system_prompt = (
        f"You are a professional translator. Translate the following text into "
        f"{target_language}. Output ONLY the translation, nothing else. "
        f"Do not add notes or explanations."
    )
    return run_agent(system_prompt, text, max_new_tokens=128)

def run_chain(initial_input, agents=AGENTS, verbose=True):
    """Pasa el mensaje por la cadena de agentes, salto a salto."""
    message = initial_input
    history = []  # guardamos lo que cruza cada arista

    for i, system_prompt in enumerate(agents):
        output = run_agent(system_prompt, message)
        history.append({
            "agent": i + 1,
            "input": message,
            "output": output,
        })
        if verbose:
            print(f"\n--- Agente {i+1} ---")
            print(f"  IN : {message[:120]}")
            print(f"  OUT: {output[:120]}")
        message = output  # la salida de este nodo es la entrada del siguiente

    return message, history

final, history = run_chain("The quarterly report shows a 12% increase in revenue.")
print("\n=== Salida final ===")
print(final)

texto = "Revenue increased by 12% compared to the previous quarter."
print(translate(texto, "French"))