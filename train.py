import os
import random
import argparse
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from lib.out import bcolors
from lib.dataset import AVDataset
from lib.preprocess.utils import init_tokenizer
from lib.models import UnimodalDecoder, MultimodalDecoder
from lib.evaluation_metrics import WordErrorRate


def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Data Extraction Parameters
    parser.add_argument('--pipeline', type=str,
                        help='Unimodal or multimodal pipeline. Choices are: {"unimodal", "multimodal"}.')
    parser.add_argument('--id_noise', type=str,
                        help='Noise id. Choices are: {"{speaker_label}_clean", "{speaker_label}_mask_{p_mask}"}.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train. If loading from a checkpoint,' + 
                        'this number is added to the already trained epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size to run with.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate to train with.')
    parser.add_argument('--phases', type=str, default=['train', 'valid[teacher]', 'valid'], nargs="*",
                        help='Which phases to run with.')
    parser.add_argument('--checkpoint', type=str, default="",
                        help='Name of run to load checkpoint from.')
    parser.add_argument('--dir_source', type=str, default="data",
                        help='Name of directory to load the data from.')

    # Pipeline Parameters
    parser.add_argument('--max_target_len', type=int, default=25,
                        help='Maximum sequence length of a target transcript.')
    parser.add_argument('--d_audio', type=int, default=[312, 768], nargs=2,
                        help='Dimension of the audio embedding.')
    parser.add_argument('--d_vision', type=int, default=512,
                        help='Dimension of the vision embedding.')
    parser.add_argument('--depth', type=int, default=4,
                        help='Depth of the TransformerDecoder')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout of the TransformerDecoderLayer')

    # Torch Parameters
    parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        help='Device to use for torch.')
    parser.add_argument('--seed_random', type=int, default=0,
                        help='Random seed for reproducibility.')

    # Logging Parameters
    parser.add_argument('--ignore_wer', action='store_true',
                        help='If flag is called, ignore WER when logging.')
    parser.add_argument('--cp_frequency', type=int, default=10,
                        help='Frequency to save a model checkpoint.')

    args = parser.parse_args()

    # Assertions, to make sure params are valid.
    assert args.epochs > 0, "epochs must be positive"
    assert args.batch_size > 0, "batch_size must be positive"
    assert args.cp_frequency > 0, "cp_frequency must be positive"
    assert (args.checkpoint == '' or 
            os.path.exists(f"models/checkpoints/{args.checkpoint}.pt")), \
           f"checkpoint '{args.checkpoint}' not found. " + \
           "checkpoint must be of format [{pipeline}]_[{id_noise}]_cp{epoch}"
    assert len(args.d_audio) == 2, "d_audio must have length 2"
    assert os.path.isdir(f"{args.dir_source}/train/audio/{args.id_noise}"), \
           f"id_noise '{args.id_noise}' not found"

    return args


def init_pipeline(args: argparse.Namespace) -> nn.Module:
    """
    Initializes and loads the model weights of the ASR pipeline,
    using a checkpoint if indicated.
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
        assert False, "pipeline flag must be one of {'unimodal', 'multimodal'}."

    if args.checkpoint != "":
        pipeline.load_state_dict(
            torch.load(f'models/checkpoints/{args.checkpoint}.pt', 
                       map_location=torch.device('cpu'))['pipeline_state_dict']
        )
    pipeline.to(args.device)
    return pipeline


def init_dataloaders() -> List[DataLoader]:
    """
    Initializes the dataloader for the 
    training and validation splits.
    """
    loaders = {}
    for phase in args.phases:
        # Skip loading valid[teacher], since
        # it is just valid
        if phase == 'valid[teacher]':
            if 'valid' in args.phases:
                continue
            else:
                phase = 'valid'
        # Load dataset and dataloader
        dataset = AVDataset(
            f'{args.dir_source}/{phase}',
            args.id_noise,
            pad_token=PAD_token,
            max_target_length=args.max_target_len
        )
        loaders[phase] = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
    return loaders


def calculate_loss(audio: torch.tensor, 
                   vision: torch.tensor, 
                   target: torch.tensor,
                   teacher_forcing: bool) -> Tuple[torch.tensor, torch.tensor]:
    """
    Performs a forward pass of the audio 
    (and optionally vision) for the pipeline.
    """
    batch_size = audio.shape[0]
    input_decoder = torch.tensor(
        [[BOS_token]], device=args.device
    ).repeat_interleave(batch_size, dim=0)

    if teacher_forcing:
        input_decoder = torch.cat([input_decoder, target[:, :-1]], axis=1)
        # Get mask to mask out the next words
        tgt_mask = pipeline.get_tgt_mask(args.max_target_len).to(args.device)
        tgt_pad_mask = pipeline.create_pad_mask(input_decoder, PAD_token)
        # Get predictions
        if args.pipeline == 'unimodal':
            pred = pipeline(audio, input_decoder, tgt_mask, tgt_pad_mask)
        elif args.pipeline == 'multimodal':
            pred = pipeline(audio, vision, input_decoder, tgt_mask, tgt_pad_mask)
        # Permute pred to have vocab size before sequence
        # Calculate loss
        loss = criterion(pred.transpose(1, 2), target)

        return loss, pred.argmax(dim=-1)
    else:
        current_pred = torch.zeros(batch_size, n_tokens, args.max_target_len, device=args.device)
        for di in range(args.max_target_len):
            # Get source mask
            tgt_mask = pipeline.get_tgt_mask(input_decoder.shape[1]).to(args.device)
            # Standard training 
            if args.pipeline == 'unimodal':
                next_pred = pipeline(audio, input_decoder, tgt_mask)
            elif args.pipeline == 'multimodal':
                next_pred = pipeline(audio, vision, input_decoder, tgt_mask)
            # Concatenate previous input with predicted best word
            current_pred[:, :, di] = next_pred[:, -1, :]
            next_token = next_pred.topk(1)[1][:, -1, :].detach()
            input_decoder = torch.cat((input_decoder, next_token), dim=1)

            # Stop if model predicts end of sentence for all items in batch
            if ((next_token == PAD_token) | (next_token == EOS_token)).all():
                break

        # Calculate loss
        loss = criterion(current_pred, target)

        return loss, input_decoder[:, 1:]


def start_training(phases: List[str],
                   loaders: DataLoader,
                   writer: SummaryWriter,
                   epochs: int) -> None:
    """
    Begin and log the model training.
    """
    for epoch in range(epoch_start, epoch_start + epochs):
        print(f"> Epoch {bcolors.OKCYAN}{epoch + 1}{bcolors.ENDC} of " +
              f"{bcolors.OKCYAN}{epoch_start + epochs}{bcolors.ENDC}")

        for phase in phases:
            print(f"\t Begin {bcolors.OKGREEN}{phase}{bcolors.ENDC} phase.")
            phase_loader = phase
            if phase == 'valid[teacher]':
                phase_loader = 'valid'
            dataloader = loaders[phase_loader]
            n_batches = len(dataloader)
            pred_str_list = []
            target_str_list = []

            for id_batch, (feats, targets) in tqdm(enumerate(dataloader),
                                                   total=n_batches):
                audio = feats[0].to(args.device)
                target = targets[0].to(args.device)
                if args.pipeline == 'multimodal':
                    vision = feats[1].unsqueeze(1).to(args.device)
                else:
                    vision = None
                target_str_list.extend(targets[1])

                sample_iter = (epoch * n_batches + id_batch) * args.batch_size
                batch_size = target.shape[0]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        pipeline.train()
                    else:
                        pipeline.eval()
                    loss, pred = calculate_loss(
                        audio, 
                        vision, 
                        target, 
                        teacher_forcing=(phase == 'train' or phase == 'valid[teacher]')
                    )

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if not args.ignore_wer:
                    words_pred = []
                    for i in range(batch_size):
                        eos_pred = (pred[i] == EOS_token).nonzero()
                        if len(eos_pred) > 0:
                            eos_pred = eos_pred[0].item()
                        else:
                            eos_pred = args.max_target_len
                        seq_pred = tokenizer.inverse_transform(
                            pred[i, :eos_pred].cpu().long()
                        )
                        words_pred.append(" ".join(seq_pred))
                    pred_str_list.extend(words_pred)

                writer.add_scalar(f'Loss/{phase}', loss.item(), sample_iter)
            
            if not args.ignore_wer:
                error = wer(pred_str_list, target_str_list)
                writer.add_scalar(f'WER/{phase}', error, sample_iter)

        if (epoch + 1) % args.cp_frequency == 0:
            if not os.path.isdir('models/checkpoints'):
                os.mkdir('models/checkpoints')

            checkpoint = {}
            checkpoint['pipeline_state_dict'] = pipeline.state_dict()
            checkpoint['opt_state_dict'] = optimizer.state_dict()
            torch.save(
                checkpoint,
                f'models/checkpoints/{args.pipeline}_[{args.id_noise}]_cp{epoch + 1}.pt'
            )


if __name__ == "__main__":
    """
    Trains a Unimodal or Multimodal ASR model on a noised variation
    of an training split. Logs CrossEntropyLoss and Word Error Rate.
    """
    args = parse_args()
    print(f"> Using device: {bcolors.OKGREEN}{args.device}{bcolors.ENDC}")
    # Set random seed
    random.seed(args.seed_random)
    np.random.seed(args.seed_random)
    torch.manual_seed(args.seed_random)
    # Create models dir
    if not os.path.isdir('models'):
        os.mkdir('models')
    # Determine run_id
    if args.checkpoint == "":
        epoch_start = 0
    else:
        epoch_start = int(args.checkpoint[args.checkpoint.find('_cp') + 3:])
    run_id = f"{args.pipeline}_[{args.id_noise}]_{args.epochs + epoch_start}"

    # Initialize pipeline and logger
    tokenizer, BOS_token, EOS_token, PAD_token, n_tokens = init_tokenizer(args.dir_source)
    args.n_tokens = n_tokens
    pipeline = init_pipeline(args)
    loaders = init_dataloaders()

    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
    optimizer = optim.Adam(pipeline.parameters(), lr=args.lr)
    if args.checkpoint != "":
        optimizer.load_state_dict(
            torch.load(f'models/checkpoints/{args.checkpoint}.pt', 
                       map_location='cpu')['opt_state_dict']
        )

    # Loggers
    writer = SummaryWriter(f'runs/{run_id}')
    if not args.ignore_wer:
        wer = WordErrorRate()

    start_training(args.phases, loaders, writer, args.epochs)
    
    run_dict = {}
    run_dict['pipeline_state_dict'] = pipeline.state_dict()
    run_dict['opt_state_dict'] = optimizer.state_dict()
    torch.save(run_dict,f'models/{run_id}.pt')
