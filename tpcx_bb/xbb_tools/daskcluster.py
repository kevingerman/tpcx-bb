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

from . import config
from .cluster_startup import attach_to_cluster
from .device import device_memory_limit, memory_limit, visible_devices

import subprocess

import logging, os, sys, math, time
log=logging.getLogger()

"""
   Thin wrapper around a thin wrapper to use args from xbb_tools.config to call
   Start stop dask cluster for automated benchmarks.
"""

def cli(commandline=None):
    parser = config.get_config().build_argparser()

    parser.add_argument(
        'commands', nargs='*',
        help='one or more of.. start_workers, stop_workers, start_scheduler, stop_scheduler.  More details: help-commands'
    )
    parser.usage="Start and stop, and get data from dask cluster described in config file."
    args = vars(parser.parse_args( commandline ))
    conf=config.get_config( args, fname=args.get('configfile'), envprefix='DASK_')

    env={'CUDA_VISIBLE_DEVICES':conf.get('CUDA_VISIBLE_DEVICES',os.getenv(
            'CUDA_VISIBLE_DEVICES',visible_devices()))}

    if conf.cluster_mode == "TCP":
        env.update({'DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT':conf.get(
                        'distributed__comm__timeouts__connect', "100s"),
                    'DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP':conf.get(
                        'distrubuted__comm__timeouts__tcp', "600s"),
                    'DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN':conf.get(
                        'distributed__comm__retry__delay__min', "1s"),
                    'DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX':conf.get(
                        'distributed__comm__retry__delay__max', "60s")
                    })
    elif conf.cluster_mode=='NVLINK':
        env.update({'tpcxbb_benchmark_sweep_run':bool(conf.get('tpcxbb_benchmark_sweep_run',True)),
                    'DASK_RMM__POOL_SIZE':conf.get('rmm__pool_size','1GB'),
                    'DASK_UCX__CUDA_COPY':conf.get('ucx__cuda_copy',True),
                    'DASK_UCX__TCP':conf.get('ucx__tcp',True),
                    'DASK_UCX__NVLINK':conf.get('ucx__nvlink',True),
                    'DASK_UCX__INFINIBAND':conf.get('ucx_infiniband',False),
                    'DASK_UCX__RDMACM':conf.get('ucx__rdmacm',False),
                    })

    for cmd in conf.commands:
        cmdf = cmd.upper().strip().replace('-','_')
        if cmdf == 'START_WORKERS':
            start_workers( conf, env )
        elif cmdf == 'START_SCHEDULER':
            start_scheduler( conf, env )
        elif cmdf == 'STOP_WORKERS':
            stop_workers( conf, env )
        elif cmdf == 'STOP_SCHEDULER':
            stop_scheduler( conf, env )
        elif cmdf == 'DUMP_TASK_STREAM':
            dump_task_stream( conf, env )
        else:
            print( 'Unknown comand "{}"'.format( cmdf ) )
        time.sleep(2)

def start_scheduler( conf, env ):
    clusterkeys=( 'dashboard_address', 'diagnostics_port', 'dask_host',
                  'dask_interface', 'dask_port', 'dask_preload',
                  'dask_protocol', 'dask_scheduler-file' )

    args=[str(a) for k in filter( lambda x: x in clusterkeys, conf.keys()) for a in [k.replace('dask_','--'),conf.get(k)]]
    return subprocess.Popen([sys.executable, '-m', 'distributed.cli.dask_scheduler'] + args,
                            stdout=open( os.path.join( conf.get('logdir', os.getcwd()), 'scheduler_{}.stdout.log'.format(os.getpid())), 'w'),
                            stderr=open( os.path.join( conf.get('logdir', os.getcwd()), 'scheduler_{}.stderr.log'.format(os.getpid())), 'w'),
                            close_fds=True, env,
                            restore_signals=False, start_new_session=True)

def start_workers( conf, env ):
    nworkers=int(conf.dask_n_workers)
    gpu_max_mem=float(device_memory_limit())
    device_mem_limit="{}MB".format(int(float(conf.get('dask_device-memory-limit',gpu_max_mem))/(1024**2)*.8))
    sys_max_mem="{}MB".format( int(memory_limit()/(int(nworkers)*1024**2)))
    env.update({'DEVICE_MEMORY_LIMIT':device_mem_limit,
                'POOL_SIZE':'{}MB'.format( int(gpu_max_mem/(1024**2))),
                'LOGDIR':conf.get('log_dir', './'),
                'WORKER_DIR':conf.get('dask_worker_dir','./'),
                'MAX_SYSTEM_MEMORY':sys_max_mem})

    args=[sys.executable, '-m', 'dask_cuda.cli.dask_cuda_worker',
          '--device-memory-limit', device_mem_limit, '--no-reconnect',
          '--{}-tcp-over-ucx'.format(conf.get('dask_tcp-over-ucx','disable')), 
          '--memory-limit', conf.get('dask_memory-limit',sys_max_mem), 
          '--rmm-pool-size', conf.get('dask_rmm-pool-size',device_mem_limit),
          '--{}-infiniband'.format( conf.get('dask_infiniband','disable')),
          '--{}-nvlink'.format( conf.get( 'dask_nvlink', 'disable')),
          '--{}-rdmacm'.format( conf.get( 'dask_rdmacm', 'disable'))]

    clusterkeys=( 'dask_diagnostics_port', 'dask_host', 'dask_local-directory',
                  'dask_interface', 'dask_port', 'dask_preload',
                  'dask_scheduler-file', 'dask_net-devices', 'dask_nthreads',
                  'dask_tls-ca-file', 'dask_tls-cert', 'dask_tls-key' )

    args+=[str(a) for k in filter( lambda x: x in clusterkeys, conf.keys()) for a in [k.replace('dask_','--'),conf.get(k)]]
    #print ("EXECUTE:  {}".format( ' '.join( map(str,args))))

    visible_gpus=visible_devices()
    scale=math.ceil(len(visible_gpus)/nworkers)
    
    for i in range( nworkers):
        mygpus=list(visible_gpus*scale)[i%len(visible_gpus):
                                        i+(nworkers*scale):
                                        nworkers]
        env['CUDA_VISIBLE_DEVICES']=','.join(list(map(str,mygpus)))
        env['NVIDIA_VISIBLE_DEVICES']=env['CUDA_VISIBLE_DEVICES']
        #print ("EXECUTE 'worker-{}: {} on GPUs: {}".format( i,' '.join( map(str,args)),env['CUDA_VISIBLE_DEVICES']))

        pid= subprocess.Popen(args + ['--name', 'worker-{}'.format(i)],
                             stdout=open( os.path.join( conf.get('logdir', os.getcwd()), 
                                                       'worker{}_{}.stdout.log'.format(i,os.getpid())), 'w'),
                             stderr=open( os.path.join( conf.get('logdir', os.getcwd()), 
                                                       'worker{}_{}.stderr.log'.format(i,os.getpid())), 'w'),
                             close_fds=True, env=env, restore_signals=False, start_new_session=True)
        #print( "*** WORKER {} PID: {}".format( i, pid.pid ))


def stop_scheduler( conf, env ):
    try:
        client = attach_to_cluster( conf )
        client.shutdown()
    except:
        log.exception( "Failed to stop scheduler", sys.exc_info())


def stop_workers( conf, env ):
    try:
        client = attach_to_cluster( conf )
        client.cancel( client.futures())
    except:
        log.exception( "Failed to stop workers", sys.exc_info())


def dump_task_stream( conf, env ):
    try:
        client = attach_to_cluster( conf )
        client.get_task_stream()
        raise Exception("Not yet implemented!!!")
    except:
        log.exception( "Failed to dump task_stream", sys.exc_info())


if __name__ == '__main__':
    import sys
    cli(sys.argv[1:])
