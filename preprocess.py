import os
import shutil
import math
import random
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
from tqdm import tqdm

from lib.out import bcolors
from lib.preprocess.fileio import DataReader, DataWriter
from lib.preprocess.utils import remove_duplicates, encode_waveforms, encode_words
from lib.preprocess.word_to_wav import WordToWav


def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    # IO parameters
    parser.add_argument('--reset_data', action="store_true",
                        help='If flag is called, then all data is deleted and regenerated.')
    parser.add_argument('--n_tasks', type=int, default=-1,
                        help='Number of data tasks to generate for each data split. The value -1 will generate all data.')
    parser.add_argument('--train_valid_split', type=float, default=[0.9, 0.1], nargs=2,
                        help='Proportion of train-valid instructions to split.')
    parser.add_argument('--dir_source', type=str, default="data/full_2.1.0",
                        help='Source directory for file reading.')
    parser.add_argument('--dir_out', type=str, default="data",
                        help='Output directory for file writing.')
    # Size parameters
    parser.add_argument('--max_wav_length', type=int, default=int(1e5),
                        help='Maximum length of wav to pad to.')
    parser.add_argument('--max_target_length', type=int, default=25,
                        help='Maximum target length.')
    parser.add_argument('--d_audio', type=int, default=[312, 768], nargs=2,
                        help='Dimension of the audio embedding.')
    parser.add_argument('--d_vision', type=int, default=512,
                        help='Dimension of the vision embedding.')
    # Perturbation parameters
    parser.add_argument('--speaker_list', type=str,
                        default=['american:lj_16khz [] []',
                                 'indic:v3_en_indic [tamil_female,bengali_female,malayalam_male,manipuri_female,assamese_female,' +
                                 'gujarati_male,telugu_male,kannada_male,hindi_female,rajasthani_female] ' +
                                 '[kannada_female,bengali_male,tamil_male,gujarati_female,assamese_male]'], nargs="*",
                        help='Label-prefixed list of speakers to use for each set. (i.e., label:speaker [accents_seen] [accents_unseen])')
    parser.add_argument('--k_nouns', type=int, default=100,
                        help='Top-k nouns to be prioritized for perturbations.')
    parser.add_argument('--p_mask_all', type=float, default=[0.2, 0.4], nargs="*",
                        help='Proportions of words to mask.')
    parser.add_argument('--p_mask_nouns', type=float, default=[0.2, 0.4, 1.0], nargs="*",
                        help='Proportions of nouns to mask.')
    # Torch Parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size used for generation.')
    parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        help='Device to use for torch.')
    parser.add_argument('--seed_random', type=int, default=0,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    # Assertions, to make sure params are valid.
    assert (args.n_tasks > 0 or 
            args.n_tasks == -1), \
            "n_tasks must be positive or -1"
    assert sorted(args.train_valid_split)[0] > 0, \
           "train_valid_split proportions must be positive"
    assert sum(args.train_valid_split) == 1, \
           "train_valid_split proportions must sum to 1"
    assert len(args.speaker_list) >= 1, \
           "speaker_list must have at least one member"

    # Parse speaker list
    speaker_list = []
    label_set = set()
    for param in args.speaker_list:
        # Remove white spaces
        param = param.replace(" ", "")

        # Tokenize list
        i_start_speaker = param.find(":")
        i_start_accents_seen = param.find("[")
        i_end_accents_seen = param.find("]")
        i_start_accents_unseen = i_end_accents_seen + \
            1 + param[i_end_accents_seen + 1:].find("[")
        i_end_accents_unseen = i_end_accents_seen + \
            1 + param[i_end_accents_seen + 1:].find("]")

        label = param[:i_start_speaker]
        speaker = param[i_start_speaker + 1:i_start_accents_seen]
        accents_seen = [accent for accent in param[i_start_accents_seen +
                                                   1:i_end_accents_seen].split(',') if accent != ""]
        accents_unseen = [accent for accent in param[i_start_accents_unseen +
                                                     1:i_end_accents_unseen].split(',') if accent != ""]

        # Assertions
        assert label not in label_set, \
               "Speaker labels must be unique"
        label_set.add(label)
        assert speaker in ['lj_16khz', 'v3_en_indic'], \
               "Speaker must be one of: {'lj_16khz', 'v3_en_indic'}"
        speaker_list.append((label, speaker, accents_seen, accents_unseen))
    args.speaker_list = speaker_list

    for param in [args.p_mask_all, args.p_mask_nouns]:
        assert 0 not in param, \
               "p_mask must not contain 0"
        assert (len(param) == 0 or 
                (sorted(param)[0] > 0 and 
                 sorted(param)[0] <= 1)), \
               "p_mask and p_swap must both be in [0, 1]"
        assert len(set(param)) == len(param), \
               "p_mask must contain unique values"

    return args


def init_directories() -> None:
    """
    Creates directories to write preprocessed files.
    """
    for dir_set in dir_out_list:
        # Delete dir if args['reset_data']
        if args.reset_data and os.path.isdir(dir_set):
            shutil.rmtree(dir_set)
        # Assert that dir is missing
        assert not os.path.isdir(dir_set), \
               "Detected an existing data directory. To reset data, call the --reset_data flag"
        os.mkdir(dir_set)


def start_read() -> Tuple[List, List]:
    """
    Begin reading the ALFRED data source directory, 
    and return the target transcripts and CLIP features.
    """
    targets = []
    visions = []
    # Split train into train and test
    valid_tasks = []

    for i_set, dir_out in enumerate(dir_out_list):
        source_subdir = source_subdir_list[i_set]
        # Check if subdir has already been read
        if source_subdir != "train" and source_subdir in source_subdir_list[:i_set]:
            i_previous = source_subdir_list.index(source_subdir)
            print(f"> Copying read from {bcolors.OKGREEN}{sets[i_previous]}{bcolors.ENDC} " +
                  f"to {bcolors.OKCYAN}{sets[i_set]}{bcolors.ENDC}.")

            targets.append(targets[i_previous].copy(deep=True))
            visions.append(visions[i_previous].copy(deep=True))
            continue

        # Define dir_source and n_tasks
        dir_source = os.path.join(args.dir_source, source_subdir)
        if (args.n_tasks == -1):
            n_tasks = len(sorted(os.listdir(dir_source)))
        else:
            n_tasks = args.n_tasks
        if i_set == 0:
            # Training set
            train_subdir = sorted(os.listdir(dir_source))[:n_tasks]
            dir_task_list = random.sample(
                train_subdir, int(n_tasks * args.train_valid_split[0])
            )
            valid_tasks = [
                dir_task for dir_task in train_subdir if dir_task not in dir_task_list
            ]
            n_tasks = len(dir_task_list)
        elif i_set == 1:
            # Validation set
            dir_task_list = valid_tasks
            n_tasks = len(dir_task_list)
        else:
            # Test set
            dir_task_list = [
                dir_task for i, dir_task in enumerate(sorted(os.listdir(dir_source))) if i < n_tasks
            ]

        print(f"> Reading {bcolors.OKGREEN}{n_tasks}{bcolors.ENDC} tasks " +
              f"from {bcolors.OKGREEN}{source_subdir}{bcolors.ENDC} " +
              f"to {bcolors.OKCYAN}{dir_out}{bcolors.ENDC}.")

        target, vision = reader.build_target_and_vision(
            dir_source=dir_source,
            dir_task_list=dir_task_list,
            d_vision=args.d_vision
        )
        targets.append(target)
        visions.append(vision)

    return targets, visions


def start_write(targets: List[pd.DataFrame],
                visions: List[pd.DataFrame],
                batch_size: int,
                word2wav: WordToWav,
                w2v_model: nn.Module) -> None:
    """
    Writes preprocessed ALFRED data to file.
    """
    # For each data split
    for i_target, (target, vision) in enumerate(zip(targets, visions)):
        writer = DataWriter(dir_out=os.path.join(args.dir_out, sets[i_target]),
                            speaker_list=args.speaker_list,
                            is_unheard=("unheard" in sets[i_target]),
                            p_mask_all=args.p_mask_all,
                            p_mask_nouns=args.p_mask_nouns,
                            waveform_sample_rate=word2wav.sample_rate_out)

        print(f"> Writing {bcolors.OKGREEN}{len(target)}{bcolors.ENDC} transcripts " +
              f"for {bcolors.OKCYAN}{sets[i_target]}{bcolors.ENDC} in " +
              f"{bcolors.OKGREEN}{math.ceil(len(target) / batch_size)}{bcolors.ENDC} batches.")
        if len(target) == 0:
            continue
        for target_batch in tqdm(np.array_split(target, math.ceil(len(target) / batch_size))):
            waveforms, indices_masked = word2wav.translate(
                target_batch.transcript_str.tolist(),
                use_accents_seen=("unheard" not in sets[i_target]),
                max_wav_length=args.max_wav_length,
                p_mask_all=args.p_mask_all,
                p_mask_nouns=args.p_mask_nouns
            )
            waveforms_clean, waveforms_masked = waveforms

            # Clean w2v (S, L_audio, H_audio)
            w2v_clean = []
            # Perturbed w2v (S, P_all + P_nouns, L_audio, H_audio)
            w2v_masked = []
            for i_speaker_set in range(len(args.speaker_list)):
                # Encode waveforms
                w2v_clean.append(
                    encode_waveforms(waveforms_clean[i_speaker_set], 
                                     w2v_model, 
                                     args.d_audio, 
                                     args.device)
                )
                # Add to list
                w2v_masked.append([])
                for waveform in waveforms_masked[i_speaker_set]:
                    w2v_masked[-1].append(
                        encode_waveforms(waveform, 
                                         w2v_model, 
                                         args.d_audio, 
                                         args.device)
                    )

            writer.write(target_batch.id_assignment,
                         target_batch.idx_instruction,
                         w2v_clean,
                         w2v_masked,
                         waveforms_clean,
                         waveforms_masked,
                         indices_masked)
        writer.write_buffers(target, vision)


if __name__ == "__main__":
    """
    Reads the ALFRED dataset, synthesizes waveforms, masks waveforms,
    and writes the preprocessed data to file.
    """
    args = parse_args()
    print(f"> Using device: {bcolors.OKGREEN}{args.device}{bcolors.ENDC}")
    # Set random seed
    random.seed(args.seed_random)
    np.random.seed(args.seed_random)
    torch.manual_seed(args.seed_random)

    # Declare and set directories
    sets = [
        "train", "valid", "test_seen_heard", "test_unseen_heard", "test_seen_unheard", "test_unseen_unheard"
    ]
    source_subdir_list = [
        "train", "train", "valid_seen", "valid_unseen", "valid_seen", "valid_unseen"
    ]
    dir_out_list = [os.path.join(args.dir_out, set_str) for set_str in sets]
    init_directories()

    # Read jsons, visions
    reader = DataReader(device=args.device)
    targets, visions = start_read()
    del reader  # Free up CLIP from GPU memory
    conflicts_valid, conflicts_test_seen, conflicts_test_unseen = remove_duplicates(
        sets, targets, visions
    )

    # Generate word tokenizer and identify nouns
    print(f"> Extracting tokens and parts of speech from words.")
    tokenizer, nouns, targets = encode_words(
        targets, args.k_nouns, args.max_target_length
    )

    # Write csv files
    conflicts_valid.to_csv(
        os.path.join(args.dir_out, sets[1], 'conflicts.csv'), index=None
    )
    conflicts_test_seen.to_csv(
        os.path.join(args.dir_out, sets[2], 'conflicts.csv'), index=None
    )
    conflicts_test_unseen.to_csv(
        os.path.join(args.dir_out, sets[3], 'conflicts.csv'), index=None
    )
    nouns.to_csv(os.path.join(args.dir_out, sets[0], 'nouns.csv'), index=None)
    nouns = set(nouns['noun'])
    np.save(
        os.path.join(args.dir_out, sets[0], 'tokenizer.npy'), 
        tokenizer.classes_
    )

    # Create extractor objects
    w2v_bundle = WAV2VEC2_ASR_BASE_960H
    w2v_model = w2v_bundle.get_model()
    word2wav = WordToWav(encoder=w2v_model,
                         encoder_sample_rate=w2v_bundle.sample_rate,
                         encoder_labels=w2v_bundle.get_labels(),
                         nouns=nouns,
                         speaker_list=args.speaker_list,
                         device=args.device)
    start_write(
        targets, visions, args.batch_size, word2wav, w2v_model
    )
