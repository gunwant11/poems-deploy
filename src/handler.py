""" Example handler file. """

import runpod
from unsloth import FastLanguageModel
# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

custom_prompt = """###Human:write a poem in style of rumi, {} ### Asstitent:  Title {}

{}"""


model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "bankai11/mistral-7b-poem", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        token = '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'
    )
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    inputs = tokenizer([
    custom_prompt.format(
        job_input, # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

    name = job_input.get('name', 'World')

    return f"Hello, {name}!"


runpod.serverless.start({"handler": handler})
