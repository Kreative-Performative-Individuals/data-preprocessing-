""" Message model for the API """

from pydantic import BaseModel
import json
from dataclasses import dataclass


class RealTimeKPI:
    kpi: str
    column: str
    operation: str
    values: list[float]

    def __init__(self, kpi, column, values, operation):
        self.kpi = kpi
        self.column = column
        self.values = values
        self.operation = operation

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)
