""" Message model for the API """

from pydantic import BaseModel
import json


class RealTimeKPI(BaseModel):
    kpi: str
    machine: str
    operation: str
    column: str
    value: float

    @classmethod
    def from_dictionary(cls, data, column):
        return cls(
            kpi=data["kpi"],
            machine=data["machine"],
            operation=data["operation"],
            column=data[column],
            value=data["value"]
        )

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

