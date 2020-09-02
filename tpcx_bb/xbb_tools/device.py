#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import pynvml
import psutil
import re, os


pynvml.nvmlInit()

def visible_devices():
    envvar = os.getenv( 'NVIDIA_VISIBLE_DEVICES',
                        os.getenv('CUDA_VISIBLE_DEVICES', 'all' ))
    if envvar.upper() == 'ALL':
        return tuple(range(pynvml.nvmlDeviceGetCount()))
    return tuple(map( int, envvar.split(',')))


def device_memory_limit( devices=None ):
    indices=[]
    if not devices:
        indices=visible_devices()
    else:
        indices=map(int,re.split( '[, ]+', devices))

    return min( [pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(d)).total for d in indices] )


def memory_limit():
    return psutil.virtual_memory().available
