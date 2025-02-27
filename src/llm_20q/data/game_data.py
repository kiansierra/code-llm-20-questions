import json
from pathlib import Path
from typing import Literal

import pandas as pd

__all__ = ["build_game_records", "build_df", "TaskType"]

TaskType = Literal["ask", "answer", "guess"]


def build_game_records(folder: str, reward: bool = True) -> list[dict]:
    """
    Builds a dataset of winning games from JSON files in the specified folder.

    Args:
        folder (str): The path to the folder containing the JSON files.

    Returns:
        list[dict]: A list of dictionaries representing the winning games. Each dictionary contains the observation,
                    reward, and additional information about the game.
    """
    all_games = list((Path(folder).glob("*.json")))
    if not all_games:
        raise ValueError(f"No game records found in {folder} directory.")
    winner_games = []
    for game_path in all_games:
        with open(game_path, "r", encoding="utf-8") as f:
            game = json.load(f)
        end_step = game["steps"][-1]
        for elem in end_step:
            reward_condition = (elem["reward"] and elem["reward"] > 0) or not reward
            if "keyword" in elem["observation"] and reward_condition and elem['action']:
                data = {**elem["observation"]}
                data["reward"] = elem["reward"]
                data = {**data, **game["info"]}
                winner_games.append(data)
    return winner_games


def build_question_df(games: list[dict]) -> pd.DataFrame:
    """
    Builds a pandas DataFrame from a list of game dictionaries.

    Args:
        games (list[dict]): A list of game dictionaries containing information about the games.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the game data with additional columns.
    """

    games_df = pd.DataFrame(games)
    games_df["position"] = games_df.apply(lambda x: list(range(len(x["questions"]))), axis=1)
    games_df = games_df.explode("position").reset_index(drop=True)
    games_df["question"] = games_df.apply(lambda x: x["questions"][x["position"]], axis=1).str.strip()
    games_df["questions"] = games_df.apply(lambda x: x["questions"][: x["position"]], axis=1)
    games_df["guesses"] = games_df.apply(lambda x: x["guesses"][: x["position"]], axis=1)
    games_df["answers"] = games_df.apply(lambda x: x["answers"][: x["position"]], axis=1)
    return games_df


def build_answers_df(games: list[dict]) -> pd.DataFrame:

    games_df = pd.DataFrame(games)
    games_df["position"] = games_df.apply(lambda x: list(range(len(x["questions"]))), axis=1)
    games_df = games_df.explode("position").reset_index(drop=True)
    games_df["max_position"] = games_df.apply(lambda x: len(x["questions"]), axis=1)
    games_df = games_df.query("position < max_position-1").reset_index(drop=True)
    games_df["questions"] = games_df.apply(lambda x: x["questions"][: x["position"] + 1], axis=1)
    games_df["guesses"] = games_df.apply(lambda x: x["guesses"][: x["position"]], axis=1)
    games_df["answer"] = games_df.apply(lambda x: x["answers"][x["position"]], axis=1).str.strip()
    games_df["answers"] = games_df.apply(lambda x: x["answers"][: x["position"]], axis=1)
    return games_df


def build_guesses_df(games: list[dict]) -> pd.DataFrame:

    games_df = pd.DataFrame(games)
    games_df["position"] = games_df.apply(lambda x: list(range(len(x["questions"]))), axis=1)
    games_df = games_df.explode("position").reset_index(drop=True)
    games_df["max_position"] = games_df.apply(lambda x: len(x["questions"]), axis=1)
    games_df = games_df.query("position < max_position-1").reset_index(drop=True)
    games_df["questions"] = games_df.apply(lambda x: x["questions"][: x["position"] + 1], axis=1)
    games_df["answers"] = games_df.apply(lambda x: x["answers"][: x["position"] + 1], axis=1)
    games_df["guess"] = games_df.apply(lambda x: x["guesses"][x["position"]], axis=1).str.strip()
    games_df["guesses"] = games_df.apply(lambda x: x["guesses"][: x["position"]], axis=1)
    return games_df


def build_df(games: list[dict], task: TaskType) -> pd.DataFrame:

    match task:
        case "ask":
            return build_question_df(games)
        case "answer":
            return build_answers_df(games)
        case "guess":
            return build_guesses_df(games)
        case _:
            raise ValueError(f"Invalid task type: {task}")
