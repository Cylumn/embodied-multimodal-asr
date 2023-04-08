import math
import random
from typing import Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from lib.preprocess.utils import get_target_transform


class WordToWav:
    """
    Synthesizes waveforms from text transcripts using 
    Silero TTS (https://github.com/snakers4/silero-models)
    """

    def __init__(self,
                 encoder: nn.Module,
                 encoder_sample_rate: int,
                 encoder_labels: tuple,
                 nouns: set,
                 speaker_list: List[Tuple],
                 device: torch.device):
        super(WordToWav, self).__init__()

        # Save parameters
        self.encoder = encoder
        self.sample_rate_out = encoder_sample_rate
        self.nouns = nouns
        self.speaker_list = speaker_list
        self.device = device

        # Set to evaluation
        self.encoder.eval()
        self.encoder.to(device)

        # Load TTS model
        self.speaker_models = {}
        for _, speaker, _, _ in speaker_list:
            if speaker in ["lj_16khz", 'v3_en_indic']:
                language = 'en'
            else:
                assert False, f"Speaker {speaker} not recognized."

            # Load model
            if speaker == "v3_en_indic":
                tts_model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language=language,
                    speaker=speaker
                )
                symbols = None
            else:
                tts_model, symbols, _, _, apply_tts = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language=language,
                    speaker=speaker
                )
                tts_model.eval()
                self.apply_tts = apply_tts
            tts_model.to(self.device)
            self.speaker_models[speaker] = (tts_model, symbols)

        # silero-model generates a UserWarning
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        # Get target transform and pipe label
        self.target_transform = get_target_transform()
        # Convert labels to dictionary
        labels = dict(
            (encoder_labels[i], i) for i in range(len(encoder_labels))
        )
        self.pipe_label = labels['|']

    def text_to_speech(self,
                       speaker_bundle: Tuple[nn.Module, List],
                       accent: str,
                       text_list: List[str],
                       max_wav_length: int) -> torch.tensor:
        """
        Synthesizes a waveform from the provided speaker and accent.
        """
        with torch.no_grad():
            if accent is None:
                # American English TTS Speaker
                sample_rate_tts = 16000
                waveforms = self.apply_tts(
                    texts=text_list,
                    model=speaker_bundle[0],
                    sample_rate=sample_rate_tts,
                    symbols=speaker_bundle[1],
                    device=self.device
                )
            else:
                # Indic English TTS Speaker
                sample_rate_tts = 24000
                max_wav_length_tts = (
                    max_wav_length * sample_rate_tts
                ) // self.sample_rate_out
                waveforms = torch.zeros(
                    (len(text_list), max_wav_length_tts), device=self.device
                )
                for i_waveform, text in enumerate(text_list):
                    waveform = speaker_bundle[0].apply_tts(
                        text=text,
                        speaker=accent,
                        sample_rate=sample_rate_tts,
                        put_accent=True,
                        put_yo=True
                    )
                    waveforms[
                        i_waveform, :min(max_wav_length_tts, len(waveform))
                    ] = waveform[:min(max_wav_length_tts, len(waveform))]

            # Resample sample rate
            if sample_rate_tts != self.sample_rate_out:
                waveforms = torchaudio.functional.resample(
                    waveforms, sample_rate_tts, self.sample_rate_out
                )

            # Pad waveforms to wav_length
            waveforms = torch.stack(
                [F.pad(waveform, (0, max_wav_length - len(waveform)))
                 for waveform in waveforms]
            )
        return waveforms.cpu()

    def _segment(self,
                 transcript_list: List[str],
                 waveform: torch.tensor) -> Tuple[float, List, List]:
        """
        Segments words in waveforms using wav2vec ASR.
        """
        # Pass through w2v
        with torch.no_grad():
            w2v_feats, _ = self.encoder(waveform.to(self.device))
        length_ratio = waveform.shape[1] / w2v_feats.shape[1]
        # Greedy and beam search decode
        chars = w2v_feats.argmax(dim=-1)  # (B, H)
        # Find pipes without subsequent pipe
        pipes_list = (chars == self.pipe_label) & \
            (chars.roll(-1, 1) != self.pipe_label)
        pipes_list = [pipe.nonzero().view(-1) for pipe in pipes_list]

        # Determine which words are nouns
        indices_nouns = []
        transcript_list = self.target_transform(transcript_list)
        for transcript, pipes in zip(transcript_list, pipes_list):
            indices_nouns.append(
                [i_word for i_word, word in enumerate(transcript)
                 if word in self.nouns and i_word < len(pipes)]
            )

        return length_ratio, pipes_list, indices_nouns

    def mask_words(self,
                   transcript_list: List[str],
                   waveform_list: torch.tensor,
                   p_mask: int,
                   only_perturb_nouns: bool) -> Tuple[torch.tensor, List]:
        """
        Masks words in the waveform with proportion p_mask.
        """
        waveform_list = waveform_list.clone()

        # Segment waveform
        length_ratio, pipes_list, indices_nouns_list = self._segment(
            transcript_list, waveform_list
        )

        # Apply masking
        indices_list = []
        for waveform, pipes, indices_nouns in zip(waveform_list, pipes_list, indices_nouns_list):
            n_words, n_nouns = len(pipes), len(indices_nouns)
            if only_perturb_nouns:
                n_mask = math.ceil(n_nouns * p_mask)
            else:
                n_mask = math.ceil(n_words * p_mask)

            # Priority: nouns
            # Sample mask indices
            indices_mask = random.sample(indices_nouns, min(n_mask, n_nouns))
            if n_mask > n_nouns:
                indices_mask.extend(
                    random.sample(
                        [i_word for i_word in range(n_words) if i_word not in indices_nouns], 
                        n_mask - n_nouns
                    )
                )
            indices_mask = sorted(indices_mask)
            indices_list.append(indices_mask)

            for i_mask in indices_mask:
                # Determine bounds
                start = 0 if i_mask == 0 else int((pipes[i_mask - 1].item() + 1) * length_ratio)
                end = int(pipes[i_mask].item() * length_ratio)
                # Pad bounds
                pad_length = 1000
                start = max(0, start - pad_length)
                end = min(waveform.shape[-1], end + pad_length)
                # Apply mask
                waveform[start:end] = torch.randn(end - start)

        return waveform_list, indices_list

    def translate(self,
                  transcript_list: List[str],
                  use_accents_seen: bool,
                  max_wav_length: int,
                  p_mask_all: List[float],
                  p_mask_nouns: List[float]) -> Tuple[Tuple, Tuple]:
        """
        Synthesizes clean and masked waveforms from text transcripts.
        """
        # TTS Clean (S, B, L_wav)
        waveforms_clean_list = []
        # TTS Perturbed (S, P_all + P_noun, B, L_wav)
        waveforms_masked_list = []
        # Indices Perturbed (S, P_all + P_noun, B, L_p)
        indices_masked_list = []

        for _, speaker, accents_seen, accents_unseen in self.speaker_list:
            if len(accents_seen) > 0 and len(accents_unseen) > 0:
                accent = random.sample(
                    accents_seen if use_accents_seen else accents_unseen, 1
                )[0]
            else:
                accent = None

            # TTS Clean (B, L_wav)
            waveforms_clean_list.append(
                self.text_to_speech(self.speaker_models[speaker],
                                    accent,
                                    transcript_list,
                                    max_wav_length)
            )
            # Add to TTS Perturbed Lists
            waveforms_masked_list.append([])
            # Indices Perturbed (S, P_all + P_noun, B, L_p)
            indices_masked_list.append([])

            # Convert Masked TTS
            for i_p, p in enumerate(p_mask_all + p_mask_nouns):
                only_perturb_nouns = (i_p >= len(p_mask_all))
                waveforms_masked, indices_masked = self.mask_words(
                    transcript_list, waveforms_clean_list[-1], p, only_perturb_nouns
                )
                waveforms_masked_list[-1].append(waveforms_masked)
                indices_masked_list[-1].append(indices_masked)

        return (waveforms_clean_list, waveforms_masked_list), indices_masked_list
