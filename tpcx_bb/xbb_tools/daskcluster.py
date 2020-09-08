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
   * config uses environment prefix 'DASK_'
"""

def cli(commandline=None):
    parser = config.get_config().build_argparser()

    parser.add_argument(
        'commands', nargs='*',
        help='one or more of.. start_workers, stop_workers, start_scheduler, stop_scheduler.'
    )
    parser.usage="Start and stop, and get data from dask cluster described in config file."
    args = vars(parser.parse_args( commandline ))
    conf=config.get_config( args, fname=args.get('configfile'), envprefix='DASK_')

    env={'CUDA_VISIBLE_DEVICES':conf.get(
            'CUDA_VISIBLE_DEVICES',visible_devices()),
         'DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT':conf.get(
            'distributed__comm__timeouts__connect', "100s"),
         'DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP':conf.get(
            'distrubuted__comm__timeouts__tcp', "600s"),
         'DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN':conf.get(
            'distributed__comm__retry__delay__min', "1s"),
         'DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX':conf.get(
            'distributed__comm__retry__delay__max', "60s")}

    for i, cmd in enumerate(conf.commands):
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
    clusterkeys=( 'dashboard_address', 'diagnostics_port', 'host',
                  'interface', 'port', 'preload',
                  'protocol', 'scheduler-file' )

    if conf.cluster_mode=='NVLINK':
        env.update({'tpcxbb_benchmark_sweep_run':conf.get('tpcxbb_benchmark_sweep_run',True),
                    'DASK_RMM__POOL_SIZE':conf.get('rmm__pool_size','1GB'),
                    'DASK_UCX__CUDA_COPY':conf.get('ucx__cuda_copy',True),
                    'DASK_UCX__TCP':conf.get('ucx__tcp',True),
                    'DASK_UCX__NVLINK':conf.get('ucx__nvlink',True),
                    'DASK_UCX__INFINIBAND':conf.get('ucx_infiniband',False),
                    'DASK_UCX__RDMACM':conf.get('ucx__rdmacm',False),
                    })

    args=[str(a) for k in filter( lambda x: x in clusterkeys, conf.keys()) for a in [f"--{k}",conf.get(k)]]
    return subprocess.Popen([sys.executable, '-m', 'distributed.cli.dask_scheduler'] + args,
                            stdout=open( os.path.join( conf.get('logdir', os.getcwd()), 'scheduler_{}.stdout.log'.format(os.getpid())), 'w'),
                            stderr=open( os.path.join( conf.get('logdir', os.getcwd()), 'scheduler_{}.stderr.log'.format(os.getpid())), 'w'),
                            close_fds=True, env=env, restore_signals=False, start_new_session=True)

def start_workers( conf, env ):
    nworkers=int(conf.n_workers)
    gpu_max_mem_mb=float(device_memory_limit())/(1024**2)

    device_mem_limit_mb=int(conf.get('device-memory-limit',0.8)*gpu_max_mem_mb)
    sys_max_mem_mb=int(memory_limit()/(int(nworkers)*1024**2))

    env.update({'DEVICE_MEMORY_LIMIT':f"{device_mem_limit_mb}MB",
                'POOL_SIZE':F'{gpu_max_mem_mb}MB',
                'LOGDIR':conf.get('log_dir', './'),
                'WORKER_DIR':conf.get('worker_dir','./'),
                'MAX_SYSTEM_MEMORY':f"{sys_max_mem_mb}MB"})

    args=[sys.executable, '-m', 'dask_cuda.cli.dask_cuda_worker',
          '--device-memory-limit', f"{device_mem_limit_mb}MB", '--no-reconnect',
          '--memory-limit', conf.get('memory-limit',f"{sys_max_mem_mb}MB"),
          '--{}-tcp-over-ucx'.format(conf.get('tcp-over-ucx','enable' if conf.cluster_mode.lower() == "NVLINK" else "disable")),
          '--{}-infiniband'.format( conf.get('infiniband','disable' if conf.cluster_mode.lower() == "NVLINK" else "enable")),
          '--{}-nvlink'.format( conf.get( 'nvlink', 'enable' if conf.cluster_mode.lower() == "NVLINK" else "disable")),
          '--{}-rdmacm'.format( conf.get( 'rdmacm', 'disable'))]

    if conf.with_blazing:
        ##with blazing: RMM_POOL_SIZE is reduced from total GPU mem by half the amount that DEVICE_MEMORY_LIMIT
        args+=['--rmm-pool-size',conf.get('rmm-pool-size', "{}MB".format(
                                          int(gpu_mem_max_mb*(float(device_mem_limit_mb)/(2*gpu_max_mem_mb)))))]

    clusterkeys=( 'diagnostics_port', 'host', 'local-directory',
                  'interface', 'port', 'preload',
                  'scheduler-file', 'net-devices', 'nthreads',
                  'tls-ca-file', 'tls-cert', 'tls-key' )

    args+=[str(a) for k in filter( lambda x: x in clusterkeys, conf.keys()) for a in [f"--{k}",conf.get(k)]]
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
