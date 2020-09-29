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

def visible_devices(devices='all'):
    """returns a tuple of integers representing device indices"""
    envvar = os.getenv( 'NVIDIA_VISIBLE_DEVICES',
                        os.getenv('CUDA_VISIBLE_DEVICES', devices ))
    if envvar.upper() == 'ALL':
        return tuple(ndx for ndx in range(pynvml.nvmlDeviceGetCount()))
    return tuple( int(ndx) if re.match( '^\d+$', ndx.strip())
                           else pynvml.nvmlDeviceGetIndex( pynvml.nvmlDeviceGetHandleByUUID(ndx.encode('UTF-8')))
                           for ndx in re.split('[, ]+',envvar))


def device_memory_limit( devices=[] ):
    return min( [pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(d)).total for d in visible_devices(','.join(devices) if devices else None)] )


def memory_limit():
    return psutil.virtual_memory().available
