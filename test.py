import os
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.out import bcolors
from lib.dataset import AVDataset
from lib.preprocess.utils import init_tokenizer
from lib.models import UnimodalDecoder, MultimodalDecoder
from lib.evaluation_metrics import WordErrorRate, RecoveryRate


def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    # Model and Test Split Parameters
    parser.add_argument('--run', type=str,
                        help='ID of the run to load weights from.')
    parser.add_argument('--id_noise', type=str,
                        help='Noise ID. Choices are: {"{speaker_label}_clean", "{speaker_label}_mask_{p_mask}"}.')
    parser.add_argument('--eval_set', type=str,
                        help='Evaluation set to use. Choices are {"test_(un)seen_(un)heard"}.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Beam size for beam search decoder.')
    parser.add_argument('--out', type=str, default='results.csv',
                        help='Output file path.')
    parser.add_argument('--dir_source', type=str, default="data",
                        help='Preprocessed source directory for file reading.')

    # Pipeline Parameters
    parser.add_argument('--d_audio', type=int, default=[312, 768], nargs=2,
                        help='Dimension of the audio embedding.')
    parser.add_argument('--d_vision', type=int, default=512,
                        help='Dimension of the vision embedding.')
    parser.add_argument('--max_target_len', type=int, default=25,
                        help='Maximum sequence length of a target transcript.')
    parser.add_argument('--depth', type=int, default=4,
                        help='Depth of the TransformerDecoder')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout of the TransformerDecoderLayer')

    # Torch Parameters
    parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        help='Device to use for torch.')

    args = parser.parse_args()

    # Assertions, to make sure params are valid.
    assert len(args.d_audio) == 2, \
           "d_audio must have length 2"
    assert os.path.exists(f"models/{args.run}.pt"), \
           f"run '{args.run}' not found"
    assert os.path.isdir(f"{args.dir_source}/train/audio/{args.id_noise}"), \
           f"id_noise '{args.id_noise}' not found"

    return args


def init_pipeline(args: argparse.Namespace) -> nn.Module:
    """
    Initializes and loads the model weights of the ASR pipeline.
    """
    if args.pipeline == 'unimodal':
        pipeline = UnimodalDecoder(
            args.d_audio, 
            args.n_tokens, 
            args.depth, 
            args.max_target_len, 
            args.dropout
        )
    elif args.pipeline == 'multimodal':
        pipeline = MultimodalDecoder(
            args.d_audio, 
            args.d_vision, 
            args.n_tokens, 
            args.depth, 
            args.max_target_len, 
            args.dropout
        )
    else:
        assert False, f"Pipeline {args.pipeline} not implemented."

    pipeline.load_state_dict(
        torch.load(f'models/{args.run}.pt', 
                   map_location=torch.device('cpu'))['pipeline_state_dict']
    )
    pipeline.to(args.device)
    return pipeline


def init_dataloader() -> DataLoader:
    """
    Initializes the dataloader for the evaluation split.
    """
    dataset = AVDataset(f'{args.dir_source}/{args.eval_set}',
                        args.id_noise,
                        pad_token=PAD_token,
                        max_target_length=args.max_target_len,
                        load_noise=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader


def start_testing(loader: DataLoader) -> None:
    """
    Performs beam search to generate predicted transcriptions
    of the spoken instructions.
    """
    n_batches = len(loader)
    pred_str_list = []
    target_str_list = []
    noise_indices_list = []

    for (audio, vision), (_, target_str, noise_indices) in tqdm(loader, total=n_batches):
        audio = audio.repeat_interleave(args.top_k, dim=0)
        audio = audio.to(args.device)
        if args.pipeline == 'multimodal':
            vision = vision.unsqueeze(1).repeat_interleave(args.top_k, dim=0)
            vision = vision.to(args.device)
        target_str_list.append(target_str[0])
        noise_indices_list.append(noise_indices[0])

        # Start decoder
        top_input = torch.tensor([[BOS_token]], device=args.device).repeat_interleave(args.top_k, dim=0)
        top_sequence_list = [([BOS_token], 0)]
        with torch.no_grad():
            for _ in range(args.max_target_len):
                # Get source mask
                tgt_mask = pipeline.get_tgt_mask(top_input.shape[1]).to(args.device)
                # Standard training
                if args.pipeline == 'unimodal':
                    pred_list = pipeline(audio, top_input, tgt_mask)
                elif args.pipeline == 'multimodal':
                    pred_list = pipeline(audio, vision, top_input, tgt_mask)
                pred_list = F.log_softmax(pred_list[:, -1, :], dim=-1)
                
                new_sequences = []
                for top_sequence, pred in zip(top_sequence_list, pred_list):
                    old_seq, old_score = top_sequence
                    p_word, tok_word = pred.topk(args.top_k)
                    for idx_word in range(args.top_k):
                        new_seq = old_seq + [tok_word[idx_word].item()]
                        new_score = old_score + p_word[idx_word].item()
                        new_sequences.append((new_seq, new_score))

                # Sort new sequences decreasing
                top_sequence_list = sorted(new_sequences, key=lambda val: val[1], reverse=True)
                # Select top-k based on score
                top_sequence_list = top_sequence_list[:args.top_k]
                top_input = torch.tensor([seq[0] for seq in top_sequence_list], device=args.device)
                
            top_sequence = [
                token for token in top_sequence_list[0][0] if token not in [BOS_token, EOS_token, PAD_token]
            ]
            pred_str_list.append(" ".join(tokenizer.inverse_transform(top_sequence)))

    wer_score = wer(pred_str_list, target_str_list) * 100
    rr_score = rr(pred_str_list, target_str_list, noise_indices_list) * 100

    # Save WER
    print(f"\t- WER: {bcolors.OKCYAN}{wer_score}{bcolors.ENDC}")
    print(f"\t- RR: {bcolors.OKCYAN}{rr_score}{bcolors.ENDC}")
    if os.path.exists(args.out):
        logs = pd.read_csv(args.out)
    else:
        logs = pd.DataFrame(columns=['run', 'id_noise', 'set', 'wer', 'rr'])
    logs.loc[len(logs)] = [args.run, args.id_noise, args.eval_set, wer_score, rr_score]
    logs.to_csv(args.out, index=None)


if __name__ == "__main__":
    """
    Evaluates a trained ASR model on a noised variation
    of an evaluation split. Produces WER and RR metrics.
    """
    args = parse_args()

    # Determine whether the model is unimodal or multimodal
    if args.run[:8] == 'unimodal':
        args.pipeline = 'unimodal'
    elif args.run[:10] == 'multimodal':
        args.pipeline = 'multimodal'
    else:
        assert False, f"Pipeline type of {args.run} not recognized."
    # Print device
    print(f"> Using device: {bcolors.OKGREEN}{args.device}{bcolors.ENDC}")
    print(f"> Testing run {bcolors.OKCYAN}{args.run}{bcolors.ENDC}")

    # Initialize pipeline and logger
    tokenizer, BOS_token, EOS_token, PAD_token, n_tokens = init_tokenizer(args.dir_source)
    args.n_tokens = n_tokens
    pipeline = init_pipeline(args)
    pipeline.eval()
    loader = init_dataloader()
    # Decoder for WER
    wer = WordErrorRate()
    rr = RecoveryRate()

    # Start training
    start_testing(loader)
