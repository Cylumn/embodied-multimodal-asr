import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import clip
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H


class PositionalEncoding(nn.Module):
    """
    Positional encoding of word tokens.
    """
    def __init__(self, 
                 dim_model: int,
                 p_dropout: float, 
                 max_len: int):
        super().__init__()

        self.dropout = nn.Dropout(p_dropout)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(
            0, max_len, dtype=torch.float
        ).view(-1, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, 
                token_embedding: torch.tensor) -> torch.tensor:
        """
        Computes the positional encoding and modifies the token embedding.
        """
        # Residual connection + pos encoding
        token_with_encoding = self.pos_encoding[:token_embedding.shape[1], :]
        token_with_encoding = token_with_encoding.transpose(0, 1)
        token_with_encoding = token_with_encoding.repeat_interleave(token_embedding.shape[0], dim=0)
        return self.dropout(token_embedding + token_with_encoding)


class UnimodalDecoder(nn.Module):
    """
    The Unimodal language decoder that attends over the 
    audio embeddings to generate the next word in the spoken instruction.
    """
    def __init__(self,
                 d_audio: Tuple[int, int],
                 d_out: int,
                 depth: int,
                 max_target_len: int,
                 dropout: float):
        super(UnimodalDecoder, self).__init__()

        self.d_audio_L, self.d_audio_H = d_audio
        self.d_out = d_out
        self.max_target_len = max_target_len

        self.emb = nn.Embedding(d_out, self.d_audio_H)
        self.positional_encoder = PositionalEncoding(
            dim_model=self.d_audio_H,
            p_dropout=0,
            max_len=max_target_len
        )
        decoder_layer = nn.TransformerDecoderLayer(
            self.d_audio_H,
            nhead=8,
            batch_first=True,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.out = nn.Linear(self.d_audio_H, d_out)

    def forward(self,
                mem: torch.tensor,  # audio embeddings
                tgt: torch.tensor,  # decoded tokens
                tgt_mask: torch.tensor = None,
                tgt_pad_mask: torch.tensor = None) -> torch.tensor:
        """
        Generates predictions for the next token in the sequence
        based on the audio embedding and decoded token sequence.
        """
        # Embedding + Positional Encoding
        # (B x L x H)
        tgt = self.emb(tgt) * math.sqrt(self.d_audio_H)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks
        transformer_out = self.decoder(
            tgt,                               # (B, L, E)
            memory=mem,                        # (B, L_S, H_S)
            tgt_mask=tgt_mask,                 # (L, L)
            tgt_key_padding_mask=tgt_pad_mask  # (B, L)
        )
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, 
                     size: int) -> torch.tensor:
        """
        Returns a lower triangular matrix.
        """
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def create_pad_mask(self,
                        matrix: torch.tensor,
                        pad_token: int) -> torch.tensor:
        """
        Returns an indicator matrix for the pad token.
        """
        return (matrix == pad_token)


class MultimodalDecoder(nn.Module):
    """
    The Multimodal language decoder that jointly attends over the 
    audio embeddings and the image embeddings to generate the 
    next word in the spoken instruction.
    """
    def __init__(self,
                 d_audio: Tuple[int, int],
                 d_vision: int,
                 d_out: int,
                 depth: int,
                 max_target_len: int,
                 dropout: float):
        super(MultimodalDecoder, self).__init__()

        self.d_audio_L, self.d_audio_H = d_audio
        self.d_vision = d_vision
        self.d_out = d_out
        self.max_target_len = max_target_len

        self.emb = nn.Embedding(d_out, self.d_audio_H)
        self.fusion = nn.Linear(self.d_audio_H + self.d_vision, self.d_audio_H)
        self.positional_encoder = PositionalEncoding(
            dim_model=self.d_audio_H,
            p_dropout=0,
            max_len=max_target_len
        )
        decoder_layer = nn.TransformerDecoderLayer(
            self.d_audio_H,
            nhead=8,
            batch_first=True,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.out = nn.Linear(self.d_audio_H, d_out)

    def forward(self,
                mem: torch.tensor,  # audio embeddings
                vis: torch.tensor,  # image embeddings
                tgt: torch.tensor,  # decoded tokens
                tgt_mask: torch.tensor = None,
                tgt_pad_mask: torch.tensor = None) -> torch.tensor:
        """
        Generates predictions for the next token in the sequence
        based on the audio embedding and decoded token sequence.
        """
        # Embedding + Positional Encoding
        # (B x L x H)
        tgt = self.emb(tgt)
        # (B x L x H)
        vis = vis.repeat_interleave(tgt.shape[1], dim=1)
        tgt = self.fusion(torch.cat([tgt, vis], axis=2))
        tgt = F.relu(tgt) * math.sqrt(self.d_audio_H)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks
        transformer_out = self.decoder(
            tgt,                               # (B, L, E)
            memory=mem,                        # (B, L_S, H_S)
            tgt_mask=tgt_mask,                 # (L, L)
            tgt_key_padding_mask=tgt_pad_mask  # (B, L)
        )
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, 
                     size: int) -> torch.tensor:
        """
        Returns a lower triangular matrix.
        """
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def create_pad_mask(self,
                        matrix: torch.tensor,
                        pad_token: int) -> torch.tensor:
        """
        Returns an indicator matrix for the pad token.
        """
        return (matrix == pad_token)


class ASRPipeline(nn.Module):
    """
    Helper class for end-to-end inference.
    """
    def __init__(self,
                 decoder: Union[UnimodalDecoder, MultimodalDecoder],
                 tokenizer: LabelEncoder,
                 device: torch.device) -> None:
        super(ASRPipeline, self).__init__()

        self.is_multimodal = isinstance(decoder, MultimodalDecoder)
        self.decoder = decoder

        # Word Tokenizer
        BOS_token, EOS_token, PAD_token = tokenizer.transform(['<BOS>', '<EOS>', '<PAD>'])
        self.tokenizer = tokenizer
        self.BOS_token = BOS_token
        self.EOS_token = EOS_token
        self.PAD_token = PAD_token
        # Load Speech Encoder
        w2v_model = WAV2VEC2_ASR_BASE_960H.get_model().to(device)
        self.w2v_model = w2v_model.eval()
        if self.is_multimodal:
            # Load Image Encoder
            clip_model, clip_processor = clip.load("ViT-B/32", device)
            self.clip_model = clip_model.eval()
            self.clip_processor = clip_processor

        super(ASRPipeline, self).to(device)
        self.device = device

    def forward(self,
                audio: torch.tensor, 
                vision: Image = None, 
                top_k: int = 5) -> str:
        """
        Helper function for prediction during inference,
        from a waveform tensor and a PIL Image to a text transcript.
        """
        with torch.no_grad():
            w2v_feats = self.w2v_model.feature_extractor(audio.to(self.device), len(audio))[0]
            w2v_feats = self.w2v_model.encoder(w2v_feats)
            if self.is_multimodal:
                clip_feats = self.clip_processor(vision).to(self.device)
                clip_feats = self.clip_model.encode_image(clip_feats.unsqueeze(0)).unsqueeze(0)

            # Beam Search
            audio = w2v_feats.repeat_interleave(top_k, dim=0)
            if self.is_multimodal:
                vision = clip_feats.repeat_interleave(top_k, dim=0)
            top_input = torch.tensor(
                [[self.BOS_token]], device=self.device
            ).repeat_interleave(top_k, dim=0)
            top_sequence_list = [([self.BOS_token], 0)]
            for _ in range(self.decoder.max_target_len):
                # Get source mask
                tgt_mask = self.decoder.get_tgt_mask(top_input.shape[1]).to(self.device)
                if self.is_multimodal:
                    pred_list = self.decoder(audio, vision, top_input, tgt_mask)
                else:
                    pred_list = self.decoder(audio, top_input, tgt_mask)
                pred_list = F.log_softmax(pred_list[:, -1, :], dim=-1)
                new_sequences = []
                for top_sequence, pred in zip(top_sequence_list, pred_list):
                    old_seq, old_score = top_sequence
                    p_word, tok_word = pred.topk(top_k)
                    for idx_word in range(top_k):
                        new_seq = old_seq + [tok_word[idx_word].item()]
                        new_score = old_score + p_word[idx_word].item()
                        new_sequences.append((new_seq, new_score))
                top_sequence_list = sorted(new_sequences, key=lambda val: val[1], reverse=True)
                top_sequence_list = top_sequence_list[:top_k]
                top_input = torch.tensor([seq[0] for seq in top_sequence_list], device=self.device)

            pred_str = " ".join(self.tokenizer.inverse_transform(top_sequence_list[0][0]))
            pred_str = pred_str.replace('<BOS> ', "").replace(" <PAD>", "").replace(" <EOS>", "")
        return pred_str.capitalize() + '.'