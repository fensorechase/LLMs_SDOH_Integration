import os
import json
import sys
import math
import torch
import argparse
# import textwrap
import transformers
from peft import PeftModel
# NOTE: if version conflicts, can change transformers to ==4.36.2
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
# from transformers import GemmaTokenizer, GemmaConfig, GemmaForCausalLM, pipeline
from llama_attn_replace import replace_llama_attn
# from supervised_fine_tune import PROMPT_DICT
from tqdm import tqdm
# from queue import Queue
# from threading import Thread
# import gradio as gr
from huggingface_hub import login
from datasets import load_dataset
import numpy as np
import pandas as pd
from collections import Counter


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-70b-hf")

    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning') # -1
    parser.add_argument('--task', type=str, default='domain_CTA', help='either topic_CTA or domain_CTA.') ###
    parser.add_argument('--feat_set', type=str, default='a', help='1 options: a, b, or c, 2 options: ab, ac, or bc, 3 options: abc') ###
    parser.add_argument('--json_format', type=bool, default=False, help='False for freetext prompt. True for JSON prompt.') ###

    parser.add_argument('--num_shots', type=int, default=0, help='either 0 for 0 shot, 1 for 1 shot, 3 for 3 shot.') ###

    # parser.add_argument('--flash_attn', type=bool, default=True, help='') # False
    parser.add_argument('--temperature', type=float, default=0.01, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=128, help='') # 64 (cut some responses off), 512, 4096

    parser.add_argument('--input_data_file', type=str, default='INPUT_NaNDA.csv', help='') # INPUT_subset_NaNDA_domain_column_classification.csv
    parser.add_argument('--output_data_file', type=str, required=False, default='', help='') # Default will be: (args.feat_set) + ("zeroshot" / "oneshot") + (args.base_model after "/") + "domain" + (args.input_data_file before ".csv") + "outputs.csv"

    args = parser.parse_args()
    return args


# Feature options: (a) Var name, (b) dataset, (c) column description
"""
Zero shot: 
- 1 feature: (a), (b), (c), 
- 2 features: (a) + (b), (a) + (c), (b) + (c), 
- all 3 features: (a) + (b) + (c)

One shot: 
- 1 feature: (a), (b), (c)
- 2 features: (a) + (b), (a) + (c), (b) + (c), 
- all 3 features: (a) + (b) + (c)

"""

# Full input: " <<SYS>>\nYou are a researcher tasked with annotating a variable describing social determinants of health (SDOH).\nThis is a multi-class classification task. The five domain options are (1) social_and_community_context, (2) economic_stability, (3) education_access_and_quality, (4) neighborhood_and_built_environment, (5) health_care_and_quality.\nThe name of a variable, description of the variable, and the original data source provide important information for choosing the correct class.\nOutput your answer in JSON using the following format: {{\"domain\": domain, \"explanation\": explanation}}\n\nWhat is the domain of this variable?\n{{\"variable_name\": \"{variable_name}\"}}. Output your answer in JSON using the following format: {{\"domain\": domain, \"explanation\": explanation}}\n[/INST]"

"""
NOTE: Overall, we adpot prompt format according to these instructions: https://huggingface.co/meta-llama/Llama-2-70b-hf
"""

FREETEXT_SYSTEM_PROMPT = "You are a researcher tasked with annotating a variable describing social determinants of health (SDOH). This is a multi-class classification task. Respond with a number (1-5) or \"?\" if unsure. Give a one number response. The five domain options are:\n\n (1) Social and Community Context, \n(2) Economic Stability, \n(3) Education Access and Quality, \n(4) Neighborhood and Built Environment, \n(5) Healthcare Access and Quality.\n"


A_ONE_SHOTS_FREETEXT = "Variable name: \"ACS_PCT_OTHER_FEMALE\".\nDomain: (1)\nVariable name: \"OPP_INCAR_HSP_F_HH_50PT\".\nDomain: (2)\nVariable name: \"ACS_PCT_POSTHS_ED\".\nDomain: (3)\nVariable name: \"ACS_PCT_RENTER_HU_ABOVE65\".\nDomain: (4)\nVariable name: \"CDCP_KIDNEY_DISEASE_ADULT_C\".\nDomain: (5)\n"
B_ONE_SHOTS_FREETEXT = "Variable description: \"Percentage of population reporting some other race alone and female\".\nDomain: (1)\nVariable description: \"Fraction incarcerated on April 1st 2010: Hispanic female child with parents from 50th household income percentile\".\nDomain: (2)\nVariable description: \"Percentage of population with any postsecondary education (ages 25 and over)\".\nDomain: (3)\nVariable description: \"Percentage of renter-occupied housing units occupied by householders aged 65 and above\".\nDomain: (4)\nVariable description: \"Crude prevalence of adults aged 18 years and older who report ever having been told by a doctor, nurse, or other health professional that they have kidney disease (%)\".\nDomain: (5)\n"
C_ONE_SHOTS_FREETEXT = "Data source: \"American Community Survey (ACS)\".\nDomain: (1)\nData source: \"The Opportunity Atlas (OPP)\".\nDomain: (2)\nData source: \"American Community Survey (ACS)\".\nDomain: (3)\nData source: \"American Community Survey (ACS)\".\nDomain: (4)\nData source: \"Centers for Disease Control and Prevention Population Level Analysis and Community Estimates: Local Data for Better Health (CDCP)\".\nDomain: (5)\n"
AB_ONE_SHOTS_FREETEXT = "Variable name: \"ACS_PCT_OTHER_FEMALE\".\nVariable description: \"Percentage of population reporting some other race alone and female\".\nDomain: (1)\nVariable name: \"OPP_INCAR_HSP_F_HH_50PT\".\nVariable description: \"Fraction incarcerated on April 1st 2010: Hispanic female child with parents from 50th household income percentile\".\nDomain: (2)\nVariable name: \"ACS_PCT_POSTHS_ED\".\nVariable description: \"Percentage of population with any postsecondary education (ages 25 and over)\".\nDomain: (3)\nVariable name: \"ACS_PCT_RENTER_HU_ABOVE65\".\nVariable description: \"Percentage of renter-occupied housing units occupied by householders aged 65 and above\".\nDomain: (4)\nVariable name: \"CDCP_KIDNEY_DISEASE_ADULT_C_census_tractlevel\".\nVariable description: \"Crude prevalence of adults aged 18 years and older who report ever having been told by a doctor, nurse, or other health professional that they have kidney disease (%)\".\nDomain: (5)\n"
AC_ONE_SHOTS_FREETEXT = "Variable name: \"ACS_PCT_OTHER_FEMALE\".\nData source: \"American Community Survey (ACS)\".\nDomain: (1)\nVariable name: \"OPP_INCAR_HSP_F_HH_50PT\".\nData source: \"The Opportunity Atlas (OPP)\".\nDomain: (2)\nVariable name: \"ACS_PCT_POSTHS_ED\".\nData source: \"American Community Survey (ACS)\".\nDomain: (3)\nVariable name: \"ACS_PCT_RENTER_HU_ABOVE65\".\nData source: \"American Community Survey (ACS)\".\nDomain: (4)\nVariable name: \"CDCP_KIDNEY_DISEASE_ADULT_C_census_tractlevel\".\nData source: \"Centers for Disease Control and Prevention Population Level Analysis and Community Estimates: Local Data for Better Health (CDCP)\".\nDomain: (5)\n"
BC_ONE_SHOTS_FREETEXT = "Variable description: \"Percentage of population reporting some other race alone and female\".\nData source: \"American Community Survey (ACS)\".\nDomain: (1)\nVariable description: \"Fraction incarcerated on April 1st 2010: Hispanic female child with parents from 50th household income percentile\".\nData source: \"The Opportunity Atlas (OPP)\".\nDomain: (2)\nVariable description: \"Percentage of population with any postsecondary education (ages 25 and over)\".\nData source: \"American Community Survey (ACS)\".\nDomain: (3)\nVariable description: \"Percentage of renter-occupied housing units occupied by householders aged 65 and above\".\nData source: \"American Community Survey (ACS)\".\nDomain: (4)\nVariable description: \"Crude prevalence of adults aged 18 years and older who report ever having been told by a doctor, nurse, or other health professional that they have kidney disease (%)\".\nData source: \"Centers for Disease Control and Prevention Population Level Analysis and Community Estimates: Local Data for Better Health (CDCP)\".\nDomain: (5)\n"
ABC_ONE_SHOTS_FREETEXT = "Variable name: \"ACS_PCT_OTHER_FEMALE\".\nVariable description: \"Percentage of population reporting some other race alone and female\".\nData source: \"American Community Survey (ACS)\".\nDomain: (1)\nVariable name: \"OPP_INCAR_HSP_F_HH_50PT\".\nVariable description: \"Fraction incarcerated on April 1st 2010: Hispanic female child with parents from 50th household income percentile\".\nData source: \"The Opportunity Atlas (OPP)\".\nDomain: (2)\nVariable name: \"ACS_PCT_POSTHS_ED\".\nVariable description: \"Percentage of population with any postsecondary education (ages 25 and over)\".\nData source: \"American Community Survey (ACS)\".\nDomain: (3)\nVariable name: \"ACS_PCT_RENTER_HU_ABOVE65\".\nVariable description: \"Percentage of renter-occupied housing units occupied by householders aged 65 and above\".\nData source: \"American Community Survey (ACS)\".\nDomain: (4)\nVariable name: \"CDCP_KIDNEY_DISEASE_ADULT_C\".\nVariable description: \"Crude prevalence of adults aged 18 years and older who report ever having been told by a doctor, nurse, or other health professional that they have kidney disease (%)\".\nData source: \"Centers for Disease Control and Prevention Population Level Analysis and Community Estimates: Local Data for Better Health (CDCP)\".\nDomain: (5)\n"



PROMPT_DICT = { 
        "freetext_zeroshot_prompt_input_a": 
            "\n[INST] Variable name: \"{variable_name}\".\nDomain:[/INST]"
        ,
        "freetext_zeroshot_prompt_input_b": 
            "\n[INST] Variable description: \"{variable_label}\".\nDomain:[/INST]"
        , 
        "freetext_zeroshot_prompt_input_c": 
            "\n[INST] Data source: \"{data_source}\".\nDomain:[/INST]"
        , 
        
        "freetext_zeroshot_prompt_input_ab": 
            "\n[INST] Variable name: \"{variable_name}\".\nVariable description: \"{variable_label}\".\nDomain:[/INST]"
        , 
        "freetext_zeroshot_prompt_input_ac": 
            "\n[INST] Variable name: \"{variable_name}\".\nData source: \"{data_source}\".\nDomain:[/INST]"
        ,
        "freetext_zeroshot_prompt_input_bc": 
            "\n[INST] Variable description: \"{variable_label}\".\nData source: \"{data_source}\".\nDomain:[/INST]"
        ,
        "freetext_zeroshot_prompt_input_abc": 
            "\n[INST] Variable name: \"{variable_name}\".\nVariable description: \"{variable_label}\".\nData source: \"{data_source}\".\nDomain:[/INST]"
        ,


        "freetext_oneshot_prompt_input_a": 
            "{oneshot_prefix}\n[INST] Variable name: \"{variable_name}\".\nDomain:[/INST]"
        ,
        "freetext_oneshot_prompt_input_b": 
            "{oneshot_prefix}\n[INST] Variable label: \"{variable_label}\".\nDomain:[/INST]"
        ,
       "freetext_oneshot_prompt_input_c": 
            "{oneshot_prefix}\n[INST] Data source: \"{data_source}\".\nDomain:[/INST]"
        ,

        "freetext_oneshot_prompt_input_ab": 
            "{oneshot_prefix}\n[INST] Variable name: \"{variable_name}\".\nVariable description: \"{variable_label}\".\nDomain:[/INST]."
        ,
        "freetext_oneshot_prompt_input_ac": 
            "{oneshot_prefix}\n[INST] Variable name: \"{variable_name}\".\nData source: \"{data_source}\".\nDomain:[/INST]"
        ,
         "freetext_oneshot_prompt_input_bc": 
            "{oneshot_prefix}\n[INST] Variable description: \"{variable_label}\".\nData source: \"{data_source}\".\nDomain:[/INST]"
        ,
        "freetext_oneshot_prompt_input_abc": 
            "{oneshot_prefix}\n[INST] Variable name: \"{variable_name}\".\nVariable description: \"{variable_label}\".\nData source: \"{data_source}\".\nDomain:[/INST]" 
        ,

    }


def get_text_after_last_occurrence(text, delimiter):
    parts = text.rsplit(delimiter, 1)
    if len(parts) == 2:
        return parts[1]
    else:
        return "ERROR: No occurrences of "+delimiter+" in the text"
    

def generate_prompt(variable_name, variable_label, data_source):
    #if input:
    #  return PROMPT_DICT["prompt_input"].format(instruction=instruction, input_seg=input_seg, question=question)
    #else:
    #  return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
    prompt_input = ""
    format_prefix = ""
    system_prompt = ""
    #if args.json_format == True:
    #    format_prefix = "json_"
    #    system_prompt = str(JSON_SYSTEM_PROMPT)
    if args.json_format == False:
        format_prefix = "freetext_"
        system_prompt = str(FREETEXT_SYSTEM_PROMPT)

    if args.num_shots == 0:
        prompt_input = str(format_prefix) + "zeroshot_prompt_input_" + str(args.feat_set)
        return system_prompt + PROMPT_DICT[prompt_input].format(variable_name=variable_name, variable_label=variable_label, data_source=data_source)
    elif args.num_shots == 1:   
        prompt_input = str(format_prefix) + "oneshot_prompt_input_" + str(args.feat_set)

        oneshot_prefix = str(args.feat_set.upper()) + "_ONE_SHOTS_FREETEXT"

        oneshot_prefix = globals()[oneshot_prefix] # Grabs the value of the varaible with *name* x+"_ONE_SHOTS_FREETEXT"

        return system_prompt + PROMPT_DICT[prompt_input].format(oneshot_prefix=oneshot_prefix, variable_name=variable_name, variable_label=variable_label, data_source=data_source)
    
    
    return None # PROMPT_DICT[prompt_input].format(variable_name=variable_name, variable_label=variable_label, data_source=data_source)




def build_generator(
    item, model, tokenizer, temperature=0.01, top_p=0.9, max_gen_len=4096, use_cache=True
):
    # def response(material, question, material_type="", material_title=None):
    # material = read_txt_file(material)
    # prompt = format_prompt(material, question, material_type, material_title)
    prompt = generate_prompt(item["variable_name"], variable_label = item["variable_label"], data_source = item["data_source"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


   
    output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache
            ) 

    raw_output = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True) # Previously: skip_special_tokens=False, clean_up_tokenization_spaces=False

    # Gets text after the *LAST* occurrence of the delimeter "Domain:"
    output = get_text_after_last_occurrence(raw_output, delimiter="Domain:")
   

    # out = out.split(prompt)[0].strip() # 1: Caused index oor error
    return output, raw_output, prompt




def convert_to_written_number(number):
    """
    Output filename helper.
    """
    written_numbers = {
        0: 'zero',
        1: 'one',
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine'
    }
    return written_numbers.get(number, "Invalid number")




def main(args):
    #if args.flash_attn:
    #    replace_llama_attn()

    #orig_ctx_len = getattr(config, "max_position_embeddings", None)

    # Set RoPE scaling factor
    #if orig_ctx_len and args.context_size > orig_ctx_len:
    #    scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
    #    config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer

    tokenizer = None
    model = None

    # GEMMA CHAT FORMAT: https://huggingface.co/google/gemma-7b/discussions/62
    # Also: https://huggingface.co/google/gemma-7b-it
    """
    Other works have suggested that Gemma instruction tuned models may not reliably follow instructions:
    "Sorry for the long delay in responding -- right now, our models don't support use of a system instruction or a special context area. 
    For now, I would suggest placing all of the relevant inside the user prompt and prompting it to only use information here, 
    and unfortunately having to include it in each question. Note that we find the models are not always perfect at following instructions, but we are working on improving this!"
    - Discussion: https://huggingface.co/google/gemma-7b/discussions/62
    """
    if "gemma" in args.base_model: 
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            # model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
            # padding_side="right",
            cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=False,
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )


    if "2-70b" in args.base_model: 
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            # model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
            # padding_side="right",
            padding_side="right",
            use_fast=False,
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # Resize embeddings:
        # model.resize_token_embeddings(32001)

    elif "llama" in args.base_model: 
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            # model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
            # padding_side="right",
            padding_side="right",
            use_fast=False,
        )

       
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # Resize embeddings:
        # model.resize_token_embeddings(32001)

    if "flan" in args.base_model:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            # model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
            # padding_side="right",
            padding_side="right",
            use_fast=False,
        )

       
        model = transformers.T5ForConditionalGeneration.from_pretrained( 
            args.base_model,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    elif  "t5" in args.base_model:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            # model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
            # padding_side="right",
            padding_side="right",
            use_fast=False,
        )

        
        # https://huggingface.co/google-t5/t5-small/discussions/10
        model = transformers.T5ForConditionalGeneration.from_pretrained( 
            args.base_model,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # mistralai/Mistral-7B-v0.1
    if  "mistralai" in args.base_model:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            # model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
            # padding_side="right",
            padding_side="right",
            use_fast=False,
        )

        
        model = transformers.AutoModelForCausalLM.from_pretrained( 
            args.base_model,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        

    

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    #with open(args.input_data_file, "r") as f:
    #    test_data = json.load(f)
    access_token_read = "TODO: input your HuggingFace access token"
    login(token = access_token_read)
    # Load dataset
    test_data = load_dataset("anonymous/SDOH_CTA", data_files=args.input_data_file)  # cache_dir=args.cache_dir, AHRQ inputs OLD: INPUT_domain_column_classification
    # Select all data (stored as 'train' by huggingface dataset object)
    test_data = test_data['train']


    # import random
    # test_data = random.sample(test_data, k=2)

    test_data_pred = []
    for i in tqdm(range(len(test_data))):
        item = test_data[i]


        new_item = {}
        output, raw_output, prompt = build_generator(item, model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                              max_gen_len=args.max_gen_len)
        # output = respond(item)

        new_item["idx"] = i
        #new_item["table_id"] = test_data[i]["table_id"]
        new_item["prompt"] = prompt

        new_item["variable_name"] = test_data[i]["variable_name"]

        new_item["data_source"] = test_data[i]["data_source"]



        new_item["variable_label"] = test_data[i]["variable_label"]
        new_item["raw_output"] = raw_output
        new_item["output"] = output # Parsed for last occurrence of "Domain:"

        if args.task == 'topic_CTA':
            new_item["SDOH_topic"] = test_data[i]["SDOH_topic"] # The SDOH topic true label for example i
        elif args.task == 'domain_CTA':
            new_item["domain"] = test_data[i]["domain"] # The domain true label for example i
        # new_item["ground_truth"] = test_data[i]["ground_truth"]
        # new_item["output"] = test_data[i]["output"]

        test_data_pred.append(new_item)
        # import pdb
        # pdb.set_trace() 
    
    # After evaluation, convert to pd df, print to csv:
    final_preds_df = pd.DataFrame(test_data_pred)

    # If output filename is not explicity specified...
    # # Default will be: (args.feat_set) + ("zeroshot" / "oneshot") + (args.base_model after "/") + "domain" + (args.input_data_file before ".csv") + "outputs.csv"

    output_fname = ""
    if len(args.output_data_file) == 0:
        shots_str = str(convert_to_written_number(args.num_shots)) + "shot"
        model_str = args.base_model.split('/', 1)[-1]
        task_str = "domain" if args.task == "domain_CTA" else "topic_CTA" # Domain or Topic SDOH variable classification.
        inputfile_str = args.input_data_file.split('.', 1)[0]
        output_fname = str(args.feat_set) + shots_str + model_str + task_str + inputfile_str + "outputs.csv"
    else: 
        output_fname = args.output_data_file

    final_preds_df.to_csv(output_fname, index=False)
    print("------ PREDICTIONS COMPLETED ------")
    
    #with open(args.output_data_file, "w") as f:
    #    json.dump(test_data_pred, f, indent = 2)




if __name__ == "__main__":
    args = parse_config()
    main(args)