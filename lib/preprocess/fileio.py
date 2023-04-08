import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from PIL import Image
from tqdm import tqdm

import clip


class DataReader():
    """
    Reads ALFRED data for preprocessing.
    """

    def __init__(self,
                 device: torch.device):
        super(DataReader, self).__init__()

        self.clip_model, self.clip_processer = clip.load("ViT-B/32", device)
        self.clip_model.eval()
        self.device = device

        self.src_keys = {
            'key_annotations': ["turk_annotations", "anns"],
            'key_idx': "assignment_id",
            'key_instructions': "high_descs"
        }

    def _open_traj(self,
                   f_json: str) -> Dict[str, List]:
        """
        Loads a traj_data.json file and returns a dictionary object containing 
        images and the targets strings.
        """
        with open(f_json) as f:
            # Load file
            traj_obj = json.load(f)
            ann_obj = traj_obj
            # Access instructions
            for key in self.src_keys['key_annotations']:
                ann_obj = ann_obj[key]

        traj_dict = {}
        # Get the first image for each first high_idx
        traj_dict['images'] = []
        unique_high = set()
        for image_obj in traj_obj['images']:
            if image_obj['high_idx'] not in unique_high:
                traj_dict['images'].append(
                    image_obj['image_name'].replace('.png', '.jpg')
                )
                unique_high.add(image_obj['high_idx'])

        # Get transcript targets
        traj_dict['targets'] = []
        for ann in ann_obj:
            id_assgn = ann[self.src_keys['key_idx']]
            transcript_list = ann[self.src_keys['key_instructions']]

            traj_dict['targets'].append((id_assgn, transcript_list))

        return traj_dict

    def _encode_image(self,
                      f_image: List[str]) -> torch.tensor:
        """
        Returns a CLIP embedding of the image file.
        """
        with torch.no_grad():
            image = [Image.open(f) for f in f_image]
            image = torch.cat(
                [self.clip_processer(i).unsqueeze(0) for i in image], axis=0
            ).to(self.device)
            image = self.clip_model.encode_image(image)

        return image

    def build_target_and_vision(self,
                                dir_source: str,
                                dir_task_list: List[str],
                                d_vision: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds target and vision from dir_source/dir_task_list[i].
        """
        target_buf = pd.DataFrame([],
                                  columns=['id_assignment',
                                           'idx_instruction',
                                           'transcript_str',
                                           'transcript_tokens',
                                           'f_json'])
        vision_buf = pd.DataFrame([],
                                  columns=['id_assignment',
                                           'idx_instruction'] + np.arange(d_vision).tolist())

        for dir_task in tqdm(dir_task_list):
            for dir_trial in sorted(os.listdir(os.path.join(dir_source, dir_task))):
                dir_trial = os.path.join(dir_task, dir_trial)
                f_json = os.path.join(dir_source, dir_trial, 'traj_data.json')
                dir_image = os.path.join(dir_source, dir_trial, 'raw_images')

                # Read file
                traj_dict = self._open_traj(f_json)
                # Encode image
                clip_feats = self._encode_image(
                    [os.path.join(dir_image, image)
                     for image in traj_dict['images']]
                )

                for id_assgn, transcript_list in traj_dict['targets']:
                    for idx_instr, transcript in enumerate(transcript_list):
                        # Add target to buffer
                        target_buf.loc[len(target_buf)] = [id_assgn,
                                                           idx_instr,
                                                           transcript,
                                                           None,
                                                           f_json]

                        # Add vision to buffer
                        vision_buf.loc[len(vision_buf)] = np.zeros(
                            len(vision_buf.columns))
                        vision_buf.iloc[-1, :2] = [id_assgn, idx_instr]
                        vision_buf.iloc[-1,
                                        2:] = clip_feats[idx_instr].cpu().numpy()

        return target_buf, vision_buf


class DataWriter:
    """
    Writes augmented ALFRED data to local files.
    """

    def __init__(self,
                 dir_out: str,
                 speaker_list: List[Tuple],
                 is_unheard: bool,
                 p_mask_all: List[float],
                 p_mask_nouns: List[float],
                 waveform_sample_rate: int):
        super(DataWriter, self).__init__()

        # Save parameters
        self.dir_out = dir_out
        self.speaker_list = speaker_list
        self.is_unheard = is_unheard
        self.p_mask_all = p_mask_all
        self.p_mask_nouns = p_mask_nouns
        self.waveform_sample_rate = waveform_sample_rate

        # Create buffers
        self.dir_w2v = "audio"
        self.dir_waveform = "waveform"
        self.buffer_noise = None
        self.open()

        if not os.path.exists(dir_out):
            os.mkdir(dir_out)

    def open(self) -> None:
        """
        Creates directories and buffer files for writing.
        """
        os.mkdir(os.path.join(self.dir_out, self.dir_w2v))
        os.mkdir(os.path.join(self.dir_out, self.dir_waveform))

        for label, _, _, accents_unseen in self.speaker_list:
            if label != "":
                label = label + "_"
            if self.is_unheard and len(accents_unseen) == 0:
                continue

            os.mkdir(os.path.join(self.dir_out, self.dir_w2v, f"{label}clean"))
            os.mkdir(os.path.join(self.dir_out,
                     self.dir_waveform, f"{label}clean"))
            for p in self.p_mask_all:
                id_noise = f"{label}mask_{p:.1f}_all"
                os.mkdir(os.path.join(self.dir_out, self.dir_w2v, id_noise))
                os.mkdir(os.path.join(self.dir_out,
                         self.dir_waveform, id_noise))
            for p in self.p_mask_nouns:
                id_noise = f"{label}mask_{p:.1f}_nouns"
                os.mkdir(os.path.join(self.dir_out, self.dir_w2v, id_noise))
                os.mkdir(os.path.join(self.dir_out,
                         self.dir_waveform, id_noise))

        self.buffer_noise = pd.DataFrame([], columns=['id_assignment',
                                                      'idx_instruction',
                                                      'id_noise',
                                                      'idxs'])

    def _write_w2v(self,
                   dir_out: str,
                   id_assgn_list: List[str],
                   idx_instr_list: List[int],
                   w2v_feats_list: torch.tensor) -> None:
        """
        Writes the wav2vec audio embedding to file.
        """
        for id_assgn, idx_instr, w2v_feats in zip(id_assgn_list, idx_instr_list, w2v_feats_list):
            torch.save(w2v_feats.clone(), os.path.join(
                dir_out, f'{id_assgn}_{idx_instr}.pt'))

    def _write_waveforms(self,
                         dir_out: str,
                         id_assgn_list: List[str],
                         idx_instr_list: List[int],
                         waveforms_list: torch.tensor) -> None:
        """
        Writes the waveforms to file.
        """
        for id_assgn, idx_instr, waveform in zip(id_assgn_list, idx_instr_list, waveforms_list):
            torchaudio.save(os.path.join(
                dir_out, f'{id_assgn}_{idx_instr}.wav'), waveform.unsqueeze(0), sample_rate=self.waveform_sample_rate)

    def _write_noise(self,
                     id_noise: str,
                     id_assgn_list: List[str],
                     idx_instr_list: List[int],
                     indices_list: List[List]) -> None:
        """
        Writes audio masking metadata to a data buffer.
        """
        for id_assgn, idx_instr, indices in zip(id_assgn_list, idx_instr_list, indices_list):
            self.buffer_noise.loc[len(self.buffer_noise)] = [
                id_assgn, idx_instr, id_noise, indices]

    def write(self,
              id_assgn_list: List[str],
              idx_instr_list: List[int],
              w2v_clean: List[torch.tensor],
              w2v_masked: List[List],
              waveforms_clean: List[torch.tensor],
              waveforms_masked: List[List],
              indices_masked: List[List]) -> None:
        """
        Writes wav2vec audio embeddings, waveforms, and audio masking metadata to file.
        """
        for i_speaker, (label, _, _, accents_unseen) in enumerate(self.speaker_list):
            if label != "":
                label = label + "_"
            if self.is_unheard and len(accents_unseen) == 0:
                continue

            # Write clean
            self._write_w2v(os.path.join(self.dir_out, self.dir_w2v, f"{label}clean"),
                            id_assgn_list, idx_instr_list, w2v_clean[i_speaker])
            self._write_waveforms(os.path.join(self.dir_out, self.dir_waveform, f"{label}clean"),
                                  id_assgn_list, idx_instr_list, waveforms_clean[i_speaker])
            # Write masked
            for i, p in enumerate(self.p_mask_all + self.p_mask_nouns):
                only_perturb_nouns = i >= len(self.p_mask_all)
                id_noise = f"{label}mask_{p:.1f}_nouns" if only_perturb_nouns else f"{label}mask_{p:.1f}_all"
                self._write_w2v(os.path.join(self.dir_out, self.dir_w2v, id_noise),
                                id_assgn_list, idx_instr_list, w2v_masked[i_speaker][i])
                self._write_waveforms(os.path.join(self.dir_out, self.dir_waveform, id_noise),
                                      id_assgn_list, idx_instr_list, waveforms_masked[i_speaker][i])
                self._write_noise(id_noise,
                                  id_assgn_list,
                                  idx_instr_list,
                                  indices_masked[i_speaker][i])

    def write_buffers(self,
                      target: pd.DataFrame,
                      vision: pd.DataFrame) -> None:
        """
        Writes the target transcripts, clip embeddings, and audio masking metadata to file.
        """
        target.to_csv(os.path.join(
            self.dir_out, "target.csv"), index=False)
        vision.to_csv(os.path.join(
            self.dir_out, "clip.csv"), index=False)
        self.buffer_noise.to_csv(os.path.join(
            self.dir_out, "noise.csv"), index=False)
