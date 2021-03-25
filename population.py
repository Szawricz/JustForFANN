"""The Popolation module."""

import sqlite3
from random import choice
from string import ascii_uppercase
from sys import path

from neuronet import Neuronet


def unique_uppcase_strings(strings_number: int, chars_number: int):
    """[summary]

    Args:
        strings_number (int): [description]
        nsymbols (int): [description]

    Returns:
        [type]: [description]
    """
    strings_list = set()
    while len(strings_list) < strings_number:
        string = ''
        for _character_num in range(chars_number):
            string += choice(ascii_uppercase)
        strings_list.add(string)
    return strings_list


class Population(object):
    """Class of ANN population."""

    def __init__(self, database_path: str):
        """Init the Population of neuronets.

        Args:
            database_path (str): population database file path
        """
        self.database_path = database_path

    @classmethod
    def with_parameters(
        cls, popul_name: str, layers: list, popul_size: int,
            ) -> Population:
        """Init the Population of neuronets with the passed parameters.

        Args:
            popul_name (str): a name of the population and database
            layers (list): the population neuronets structure
            popul_size (int): a population neuronets number

        Returns:
            Population: the population of neuronets
        """
        names_anns = unique_uppcase_strings(popul_size, 4)
        connect = sqlite3.connect(f'{path[0]}/{popul_name}.db')
        create_main_popul_table = f"""
        CREATE TABLE IF NOT EXISTS main_{popul_name}_table (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ann_name TEXT NOT NULL,
        layers TEXT NOT NULL,
        succes REAL NOT NULL
        );
        """
        connect.execute(create_main_popul_table)
        connect.commit()
        for ann_name in names_anns:
            add_ann_to_database = f"""
            INSERT INTO main_{popul_name}_table (ann_name, layers, succes)
            VALUES ('{ann_name}', '{layers}', 0);
            """
            connect.execute(add_ann_to_database)
            neuronet = Neuronet.from_layers(layers)
            create_ann_table = f"""
            CREATE TABLE IF NOT EXISTS {ann_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weight_name TEXT NOT NULL,
            weight_value REAL NOT NULL
            );
            """
            connect.execute(create_ann_table)
            for weight_name, weight_value in neuronet.weights.items():
                add_weight = f"""
                    INSERT INTO {ann_name} (weight_name, weight_value)
                    VALUES ('{weight_name}', {weight_value});
                    """
                connect.execute(add_weight)
            connect.commit()
            database_path = f'{path[0]}{popul_name}.db'
        return Population(database_path)

    @classmethod
    def from_database(cls, file_path: str) -> Population:
        """Init the Population from the ready database.

        Args:
            file_path (str): population database file path

        Returns:
            Population: the population of neuronets
        """
        return Population(file_path)


    def tich(self, prop_func: callable, dataset_file: str, error: float, mort: int, mut: int):
        """[summary]

        Args:
            dataset_file (str): dataset filepath
            error (float): [description]
            mort (int): [description]
            mut (int): [description]
        """
