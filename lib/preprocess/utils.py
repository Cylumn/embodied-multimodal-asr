from collections import defaultdict
from typing import Tuple, Set, List

import nltk
import jiwer
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from lib.out import bcolors


def get_target_transform() -> jiwer.Compose:
    """
    Returns the JiWER text transformation class.
    """
    target_transform = jiwer.Compose([
        jiwer.ToUpperCase(),
        jiwer.RemovePunctuation(),
        jiwer.SubstituteRegexes({r"[^ABCDEFGHIJKLMNOPQRSTUVWXYZ\'-| ]": r""}),
        jiwer.ReduceToListOfListOfWords()
    ])

    return target_transform


def init_tokenizer(dir_source: str) -> Tuple[LabelEncoder, int, int, int, int]:
    """
    Loads the word tokenizer and special token values.
    """
    tokenizer = LabelEncoder()
    tokenizer.classes_ = np.load(f'{dir_source}/train/tokenizer.npy')

    BOS_token, EOS_token, PAD_token = tokenizer.transform(
        ['<BOS>', '<EOS>', '<PAD>'])
    n_tokens = len(tokenizer.classes_)

    return tokenizer, BOS_token, EOS_token, PAD_token, n_tokens


def remove_duplicates(sets: List[str],
                      targets: List[pd.DataFrame],
                      visions: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detects and deletes duplicate transcripts in 
    validation or test splits from the training split.
    """
    drop_punctuation = jiwer.Compose([
        jiwer.ToUpperCase(),
        jiwer.RemovePunctuation(),
        jiwer.SubstituteRegexes({r"[^ABCDEFGHIJKLMNOPQRSTUVWXYZ\'-| ]": r""})
    ])

    print(f"> Removing duplicates in {bcolors.OKCYAN}{sets[1]}{bcolors.ENDC}, " +
          f"{bcolors.OKCYAN}{sets[2]}{bcolors.ENDC}, " +
          f"{bcolors.OKCYAN}{sets[3]}{bcolors.ENDC} from " +
          f"{bcolors.OKCYAN}{sets[0]}{bcolors.ENDC}.")
    conflict_list = [[] for _ in range(3)]
    indices_list = []

    # Store all transcripts in valid, test, test_unseen into a set
    transcript_set_list = []
    for i_set in range(3):
        transcript_set = set()
        for transcript in drop_punctuation(targets[i_set + 1].transcript_str.tolist()):
            transcript_set.add(transcript)
        transcript_set_list.append(transcript_set)

    # Check for duplicates
    for i_train, transcript_train in enumerate(tqdm(drop_punctuation(targets[0].transcript_str.tolist()))):
        for i_set in range(3):
            if transcript_train in transcript_set_list[i_set]:
                conflict_list[i_set].append(transcript_train)
                indices_list.append(i_train)
                break
    conflict_list = [pd.Series(conflicts, dtype='object')
                     for conflicts in conflict_list]

    # Remove from train
    targets[0].drop(indices_list, axis=0, inplace=True)
    visions[0].drop(indices_list, axis=0, inplace=True)
    # Reset index
    targets[0].reset_index(drop=True, inplace=True)
    visions[0].reset_index(drop=True, inplace=True)

    print(f"> Found {bcolors.WARNING}{len(conflict_list[0])}{bcolors.ENDC} conflicts " +
          f"with {bcolors.OKCYAN}{sets[1]}{bcolors.ENDC} from {bcolors.OKCYAN}{sets[0]}{bcolors.ENDC}.")
    print(f"> Found {bcolors.WARNING}{len(conflict_list[1])}{bcolors.ENDC} conflicts " +
          f"with {bcolors.OKCYAN}{sets[2]}{bcolors.ENDC} from {bcolors.OKCYAN}{sets[0]}{bcolors.ENDC}.")
    print(f"> Found {bcolors.WARNING}{len(conflict_list[2])}{bcolors.ENDC} conflicts " +
          f"with {bcolors.OKCYAN}{sets[3]}{bcolors.ENDC} from {bcolors.OKCYAN}{sets[0]}{bcolors.ENDC}.")

    return conflict_list


def encode_waveforms(waveform_list: torch.tensor,
                     w2v_model: torch.nn.Module,
                     d_audio: List[int],
                     device: torch.device) -> torch.tensor:
    """
    Encodes waveforms as wav2vec audio embeddings.
    """
    with torch.no_grad():
        lengths = torch.tensor(
            [waveform_list.shape[-1]], device=device
        ).repeat_interleave(waveform_list.shape[0], dim=0)
        w2v_feats, _ = w2v_model.feature_extractor(
            waveform_list.to(device), lengths
        )
        w2v_feats = w2v_model.encoder(w2v_feats)
        w2v_feats = F.adaptive_avg_pool2d(w2v_feats, d_audio)

    return w2v_feats


def encode_words(targets: pd.DataFrame,
                 k_nouns: int,
                 max_target_length: int) -> Tuple[LabelEncoder, Set, List]:
    """
    Tokenizes transcript sequences, while also identifying nouns.
    """
    target_transform = get_target_transform()

    # Fit LabelEncoder
    vocabulary = set(['<BOS>', '<EOS>', '<PAD>'])
    transcript_lists = []
    pos_dict = defaultdict(lambda: defaultdict(int))
    for i_target, target in enumerate(targets):
        transcript_list = target_transform(target.transcript_str.tolist())

        if i_target == 0:
            # Get the part of speeches from the training split
            for transcript in transcript_list:
                transcript_poscase = [
                    word.capitalize() if i == 0 else word.lower() for i, word in enumerate(transcript)
                ]
                for token, pos in nltk.pos_tag(transcript_poscase):
                    pos_dict[token.upper()][pos] += 1
            for token in pos_dict.keys():
                # "Turn" is not a noun
                if token == 'TURN':
                    pos_dict[token] == 'VB'
                else:
                    pos_dict[token] = max(
                        pos_dict[token], key=pos_dict[token].get
                    )

        # Add words to vocabulary
        for word_list in transcript_list:
            for word in word_list:
                vocabulary.add(word)

        # Add list to buffer
        transcript_list = [
            seq[:max_target_length - 1] + ['<EOS>'] for seq in transcript_list
        ]
        transcript_lists.append(transcript_list)

    # Fit LabelEncoder
    word_to_token = LabelEncoder()
    word_to_token = word_to_token.fit(list(vocabulary))

    # Transform to tokens
    for i_target, target in enumerate(targets):
        transcript_flat = [
            word for seq in transcript_lists[i_target] for word in seq
        ]
        transcript_lengths = [len(seq) for seq in transcript_lists[i_target]]

        # If target is train, get most common nouns
        if i_target == 0:
            tokens = pd.DataFrame(
                pd.Series(transcript_flat).value_counts(normalize=True))
            tokens.rename(columns={0: 'value_counts'}, inplace=True)
            tokens.insert(0, "noun", tokens.index)
            tokens['pos'] = [pos_dict[idx] for idx in tokens.index]
            nouns = tokens[tokens.pos.isin(['NN', 'NNP', 'NNS'])]
            nouns = nouns.head(k_nouns)

        # Convert to token
        tokens_flat = word_to_token.transform(transcript_flat)

        i_token = 0
        for i_row, _ in target.iterrows():
            target.at[i_row, 'transcript_tokens'] = list(
                tokens_flat[i_token:i_token + transcript_lengths[i_row]]
            )
            i_token += transcript_lengths[i_row]

    return word_to_token, nouns, targets
