import os
import sys
import argparse
import torch
import tqdm
import json

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="models/atomic-generation/iteration-500-50000/transformer/categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/model_transformer-nL_12-nH_12-hSize_768-edpt_0.1-adpt_0.1-rdpt_0.1-odpt_0.1-pt_gpt-afn_gelu-init_pt-vSize_40542/exp_generation-seed_123-l2_0.01-vl2_T-lrsched_warmup_linear-lrwarm_0.002-clip_1-loss_nll-b2_0.999-b1_0.9-e_1e-08/bs_1-smax_40-sample_greedy-numseq_1-gs_1000-es_1000-categories_oEffect#oReact#oWant#xAttr#xEffect#xIntent#xNeed#xReact#xWant/6.25e-05_adam_64_22000.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="beam-5")
    parser.add_argument("--dataset_file", type=str, default="dataset/socialiqa")

    args = parser.parse_args()

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"
    
    category = "all"

    sampler = interactive.set_sampler(opt, args.sampling_algorithm, data_loader)

    with open(args.file) as f_in:
        for line in tqdm.tqdm(f_in):
            fields = json.loads(line.strip())
            input_event = fields["context"]
            outputs = interactive.get_atomic_sequence(
                    input_event, model, sampler, data_loader, text_encoder, category)

            print(outputs)