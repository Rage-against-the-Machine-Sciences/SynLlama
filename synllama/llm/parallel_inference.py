import torch
import json, pickle, argparse, os, random
import multiprocessing as mp
from synllama.llm.vars import *
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

instruction = TEMPLATE["instruction"]
input_template = TEMPLATE["input"]


def _extract_last_product(json_text):
    """Extract the final product SMILES from a generated pathway JSON.

    The last reaction in the chain is the one closest to the target molecule,
    so its product is the best proxy for what analog the model is proposing.
    Returns None if parsing fails or the JSON has no reactions.
    """
    try:
        parsed = json.loads(json_text)
        reactions = parsed.get("reactions", [])
        if reactions:
            return reactions[-1].get("product")
    except Exception:
        pass
    return None


def generate_text(smiles, tokenizer, model, stopping_ids, sampling_params, max_length=1600, sequential=False):
    base_input = input_template.replace("SMILES_STRING", smiles)
    generated_texts = []

    for params in sampling_params:
        temp = params["temp"]
        top_p = params["top_p"]
        repeat = params["repeat"]

        if sequential:
            # Generate one pathway at a time, feeding the products of previous
            # iterations back into the prompt so the model is nudged toward
            # structurally distinct analogs on each subsequent call.
            prev_products = []
            for _ in range(repeat):
                if prev_products:
                    products_str = ", ".join(prev_products)
                    diversity_hint = (
                        f"\nPreviously generated pathways target these molecules: {products_str}. "
                        "Generate a pathway targeting a DIFFERENT, structurally distinct analog."
                    )
                    input_text = base_input + diversity_hint
                else:
                    input_text = base_input

                prompt = "### Instruction:\n" + instruction + "\n\n### Input:\n" + input_text + "\n\n### Response: \n"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                prompt_length = inputs.input_ids.shape[1]

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=temp,
                        top_p=top_p,
                        num_return_sequences=1,
                        eos_token_id=stopping_ids,
                        pad_token_id=tokenizer.eos_token_id
                    )

                generated_text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
                generated_texts.append(generated_text)

                product = _extract_last_product(generated_text)
                if product and product not in prev_products:
                    prev_products.append(product)
        else:
            prompt = "### Instruction:\n" + instruction + "\n\n### Input:\n" + base_input + "\n\n### Response: \n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temp,
                    top_p=top_p,
                    num_return_sequences=repeat,
                    eos_token_id=stopping_ids,
                    pad_token_id=tokenizer.eos_token_id
                )
            for output in outputs:
                generated_text = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
                generated_texts.append(generated_text.strip())

    return generated_texts

def process_batch(args):
    gpu_id, model_path, smiles_batch, sampling_params, sequential = args
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
        device_map={'': device}
    )

    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    results = {}
    for smiles in tqdm(smiles_batch, desc=f"Processing on {device.upper()}"):
        try:
            response = generate_text(smiles, tokenizer, model, stopping_ids, sampling_params, sequential=sequential)
            json_responses = []
            for r in response:
                try:
                    json_responses.append(json.loads(r))
                except json.JSONDecodeError:
                    json_responses.append("json format error")
            results[smiles] = json_responses
        except Exception as e:
            results[smiles] = f"Error: {str(e)}"

    return results

def main(model_path, smiles_path, save_path, sampling_params, gpus=None, n_samples=None, seed=42, sequential=False):
    with open(smiles_path, "r") as f:
        smiles_list = [line.strip() for line in f]

    if n_samples is not None and n_samples < len(smiles_list):
        random.seed(seed)
        smiles_list = random.sample(smiles_list, n_samples)

    num_gpus = torch.cuda.device_count() if gpus is None else gpus
    print(f"Number of available GPUs: {num_gpus}")

    if num_gpus > 1:
        pool = mp.Pool(num_gpus)
        try:
            # Process batches on different GPUs
            batches = [smiles_list[i::num_gpus] for i in range(num_gpus)]
            results = pool.map(process_batch, [(i, model_path, batch, sampling_params, sequential) for i, batch in enumerate(batches)])

            # Combine results from all GPUs
            combined_results = {}
            for r in results:
                combined_results.update(r)

        finally:
            # Ensure pool cleanup happens even if an error occurs
            pool.close()  # Prevent any more tasks from being submitted
            pool.join()   # Wait for all processes to finish
            pool.terminate()  # Terminate all worker processes
    else:
        # If only one GPU, process all SMILES on that GPU
        combined_results = process_batch((0, model_path, smiles_list, sampling_params, sequential))

    # Save results
    with open(save_path, "wb") as f:
        pickle.dump(combined_results, f)

    # close pool
    return combined_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference pipeline for reaction prediction")
    parser.add_argument("--model_path", type=str, help="Path to the model", default="data/model/SynLlama-1B-2M")
    parser.add_argument("--smiles_path", type=str, help="Path to the SMILES file")
    parser.add_argument("--save_path", type=str, help="Pickle file path to save the results", default = None)
    parser.add_argument("--sample_mode", type=str, default=None, help="Sampling mode, choose from: greedy, frugal, frozen_only, low_only, medium_only, high_only")
    parser.add_argument("--temp", type=float, default=None, help="Temperature for the model")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p for the model")
    parser.add_argument("--repeat", type=int, default=None, help="Number of times to repeat the model")
    parser.add_argument("--gpus", type=int, default=None, help="name of the cuda device to use, default is all available GPUs")
    parser.add_argument("--n_samples", type=int, default=None, help="Randomly sample N molecules from the SMILES file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for molecule sampling")
    parser.add_argument("--sequential", action="store_true", help="Generate pathways sequentially, feeding previous products back as diversity hints")
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    if args.save_path is None:
        args.save_path = args.smiles_path.replace(".smi", "_results.pkl")
    directory = os.path.dirname(args.save_path)
    os.makedirs(directory, exist_ok=True)
    sample_mode_mapping = {
        "greedy": sampling_params_greedy,
        "frugal": sampling_params_frugal,
        "frozen_only": sampling_params_frozen_only,
        "low_only": sampling_params_low_only,
        "medium_only": sampling_params_medium_only,
        "high_only": sampling_params_high_only
    }
    if args.sample_mode is None:
        assert args.temp is not None and args.top_p is not None and args.repeat is not None, "Please provide a sample mode or all the sampling parameters"
        sampling_params = [
            {"temp": args.temp, "top_p": args.top_p, "repeat": args.repeat}
        ]
    else:
        assert args.sample_mode in sample_mode_mapping, f"Invalid sample mode: {args.sample_mode}"
        sampling_params = sample_mode_mapping[args.sample_mode]

    main(args.model_path, args.smiles_path, args.save_path, sampling_params=sampling_params, gpus=args.gpus, n_samples=args.n_samples, seed=args.seed, sequential=args.sequential)
