"""
Анализ датасета GLUE MRPC (Paraphrase Detection).

ЗАДАНИЕ:
Загрузите датасет MRPC, анализируйте распределение похожести пар предложений,
статистику их длин и выполните анализ парафраз.

ТРЕБОВАНИЯ:
- Загрузить датасет: load_dataset()
- Провести анализ распределения paraphrases/non-paraphrases
- Вычислить статистику длин предложений
- Провести анализ пересечения слов (word overlap)
- Создать диаграмму распределения
- Сохранить результаты в mrpc_results.json
"""

import json
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt


# TODO: Ваш код здесь


def load_glue_mrpc_dataset():
    # TODO: Реализовать загрузку
    pass


def preprocess_text(text: str):
    # TODO: Реализовать
    pass


def analyze_similarity_distribution(dataset):
    # TODO: Вернуть counts и percentages для paraphrases/non-paraphrases
    pass


def analyze_sentence_lengths(dataset):
    # TODO: Вернуть статистику для sentence1 и sentence2
    pass


def calculate_word_overlap(dataset):
    # TODO: Вернуть avg/median/max/min word overlap (Jaccard)
    pass


def create_visualization(dataset):
    # TODO: Создать график и сохранить в "mrpc_distribution.png"
    pass


def main():
    # TODO: Собрать анализ
    pass


if __name__ == "__main__":
    main()
