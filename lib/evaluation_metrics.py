from typing import List, Tuple

import jiwer
import torch.nn as nn


class WordErrorRate(nn.Module):
    """
    Computes Word Error Rate (WER) 
    with the JiWER package.
    """

    def __init__(self):
        super(WordErrorRate, self).__init__()

        self.target_transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])
        self.pred_transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])

    def forward(self,
                pred: List[str],
                target: List[str]) -> float:
        """
        Computes the WER of the prediction transcript.
        """
        return jiwer.wer(
            target,
            pred,
            truth_transform=self.target_transformation,
            hypothesis_transform=self.pred_transformation
        )


class RecoveryRate(nn.Module):
    """
    Computes Recovery Rate (RR).
    """
    def __init__(self):
        super(RecoveryRate, self).__init__()

        self.target_transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])
        self.pred_transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])

    def forward(self,
                pred: List[str],
                target: List[str],
                noise_indices: List[List],
                index_padding: int = 2) -> Tuple[float]:
        """
        Computes the RR of the prediction transcript.
        """
        n_correct = 0
        n_noise = 0
        pred = self.pred_transformation(pred)
        target = self.target_transformation(target)

        for transcript_pred, transcript_actual, indices in zip(pred, target, noise_indices):
            n_noise += len(indices)
            for i_word in indices:
                k = 1
                target_str = transcript_actual[min(len(transcript_actual) - 1, i_word)]
                correct = (
                    i_word < len(transcript_pred) and transcript_pred[i_word] == target_str
                )
                # Give word index some buffer room
                while not correct and k <= index_padding and len(transcript_pred) > 0:
                    correct = (
                        transcript_pred[min(len(transcript_pred) - 1, max(0, i_word - k))] == target_str or \
                        transcript_pred[min(len(transcript_pred) - 1, i_word + k)] == target_str
                    )
                    k += 1
                # Increment number correct
                if correct:
                    n_correct += 1
        return 1 if n_noise == 0 else n_correct / n_noise
