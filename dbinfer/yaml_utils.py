# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


from pathlib import Path
from pydantic import BaseModel
import yaml
import json

def save_pyd(pyd_model: BaseModel, path: Path):
    # NOTE: Read/write using JSON to convert enum into string.
    data_dict = json.loads(pyd_model.model_dump_json())
    with open(path, "w") as f:
        f.write(yaml.dump(data_dict))

def load_pyd(pyd_model_class, path: Path) -> BaseModel:
    with open(path, "r") as f:
        return pyd_model_class.model_validate(yaml.safe_load(f))
