#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is used to generate data for meta language models

import sys
import argparse
from metalm import MetaLM

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--vocab_size', type=int, default=20)
    parser.add_argument('--repeat_length', type=int, default=20)
    parser.add_argument('--error_rate', type=float, default=0.15)
    parser.add_argument('--sequence_length', type=int, default=4096)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()
    dataset = MetaLM(
            args.vocab_size,
            args.repeat_length,
            args.error_rate,
            args.sequence_length)
    if(args.output is None):
        dataset.generate_to_file(args.samples, sys.stdout)
    else:
        dataset.generate_to_file(args.samples, args.output)
