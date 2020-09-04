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

import os
import copy
import json
import pkgutil
import logging

from argparse import ArgumentParser


class Config(dict):

    _default_config=None
    _packagename=__module__.split('.')[0].replace('_','')
    _prefix='{}_'.format(_packagename.upper())
    _default_file_name='{}.config.json'.format(_packagename)

    __doc__="""
Config module takes parameters from command line, json files, environment, and the default config json in this pacakge and
presents a config object with dict interface and direct accessors.  Default config is overidden by config file is overidden
by environment is overidden by command line and c'tor args.  Environment will pick up any environment variables prefixed with
"{}" Prefix can be overidden in module get_config method.
    """.format( _prefix )

    def __init__( self, fname=None, **kwargs ):
        if not fname:
            fname = os.path.join( os.getcwd(), self._default_file_name )
        self.update( self.make_config( self.get_default_config() ))

        if os.path.exists( fname ):
            self.update(self.parse_config_file( fname))
        else:
            logging.getLogger().warning( "Unable to find {}".format(fname))
        self.update(kwargs)

    def parse_config_file( self, fname ):
        assert os.path.exists(fname)
        if fname.lower().endswith('yaml'):
            return {}
        elif fname.lower().endswith( 'json'):
            return json.load(open( fname, 'r'))
        else:
            raise Exception( "unsupported configfile format, {}".format( fname))

    def build_argparser( self, description="Argparser generated from config module" ):
        parser = ArgumentParser(description=description,
                                conflict_handler='resolve')
        #TODO: eval should filter for poison inputs                                                                                                                                                             
        for d in Config.get_default_config():
            name="--{}".format(d.pop("name"))
            if 'default' in d:
                d['default']=self.eval_default_value(d)
                d.pop('type')
            parser.add_argument( name, **d)

        parser.add_argument(
            '-c', '--config',
            default=os.path.join( os.getcwd(),'.'.join([self._prefix.lower()]+["config.json"])),
            action='store', dest='configfile',
            help=self.__doc__    )

        return parser

    def override( self, kvdict, skipdefaults=False ):
        defaults={}
        ff=lambda x: str(defaults.get(x,'__'+str(kvdict[x]))) != str(kvdict[x])
        if skipdefaults:
            defaults=self.make_config( self.get_default_config() )
        if kvdict:
            self.update( {k:kvdict[k] for k in filter( ff, kvdict )} )
        return self

    def as_dict( self ):
        return copy.deepcopy( self )

    def as_environment_dict(self):
        return { self._prefix + k:str(v) for k,v in self.items() }

    def __getattr__(self, name):
        if name in self:
            if isinstance( self.get(name), dict):
                c=Config()
                c._data=self.get(name)
                return c
            else:
                return self.get(name)
        else:
            raise AttributeError( "Attribute {} does not exist in {}".format( name, self.__class__ ))

    @classmethod
    def eval_default_value( clz, elem ):
        defaulttype=elem.get('type','str').strip()
        if 'type' == eval( "type({}).__name__".format( defaulttype)):
            return eval(defaulttype)( elem.get('default'))
        else:
            return elem.get('default',None)

    @classmethod
    def make_config( clz, spec):
        return { d['name']:clz.eval_default_value(d) for d in spec }

    @classmethod
    def get_default_config( clz ):
        if not clz._default_config:
            pkg=clz.__module__.split('.')[0]
            data=pkgutil.get_data( pkg, 'default.config.json')
            clz._default_config=json.loads(data.decode('UTF-8'))

        return copy.deepcopy( clz._default_config )


def get_config( conf={}, fname=None, envprefix=Config._prefix):
    env={k.replace(envprefix,''):os.getenv(k) for k in
         filter( lambda x: x.startswith( envprefix), os.environ )}
    fname=fname if fname and os.path.exists(fname) else conf.get( 'configfile', env.get('configfile'))
    return Config( fname ).override( env ).override( conf, skipdefaults=True )


class TestConfig:
    def test_get_config_is_config_instance(self):
        assert type(get_config()) == type(Config())

    def test_config_field_as_property(self):
        it=get_config( conf={"fish":"trout"} )
        assert it.fish == "trout"

    def test_environment_export(self):
        it=get_config( conf={"fish":"trout"} )
        for k,v in it.as_environment_dict().items():
            os.environ[k]=str(v)
        assert( get_config().fish == 'trout' )

    def test_overrides_configfile_with_environment( self, tmpdir ):
        fname="test_overrides_environment_options_with_configfile.json"
        os.environ["{}fish".format(Config._prefix)]="trout"
        ffix = tmpdir.join(fname).write('{"fish":"perch"}')
        it=get_config( fname=os.path.join( tmpdir, fname))
        assert it.fish == "trout"

    def test_overrides_default_with_file(self, tmpdir):
        fname="test_overrides_environment_options_with_configfile.json"
        ffix = tmpdir.join(fname).write('{"cluster_port":"44"}')
        it=Config( fname=os.path.join( tmpdir, fname) )
        assert it.cluster_port == "44"

    def test_overrides_default_with_kvpairs(self, tmpdir):
        it=get_config( {'cluster_port':'55'})
        assert it.cluster_port == "55"

    def test_overrides_default_with_environment(self, tmpdir):
        os.environ["{}cluster_port".format(Config._prefix)]="trout"
        it=get_config()
        os.environ.pop("{}cluster_port".format(Config._prefix))
        assert it.cluster_port == "trout"

    def test_overrides_environment_with_nvpairs(self):
        fname='test_overrides_configfile_with_nvpairs.json'
        os.environ["{}fish".format(Config._prefix)]="carp"
        it=get_config( {"fish":"bass"} )
        assert it.fish == "bass"

    def test_skipsdefault_values_from_file(self, tmpdir):
        defaults=Config.make_config(Config.get_default_config())
        fname="test_skipsdefault_values_from_nvpairs.json"
        ffix = tmpdir.join(fname).write('{"cluster_port":"99"}')
        it=get_config( fname=os.path.join( tmpdir,fname) )

        assert( int(it.cluster_port) == 99)

        it.override(defaults, skipdefaults=True)
        assert( int(it.cluster_port) == 99)

        it.override(defaults)
        assert( it.cluster_port == defaults.get('cluster_port'))

    def test_skipsdefault_values_from_kvpair(self, tmpdir):
        defaults=Config.make_config(Config.get_default_config())
        it=Config( cluster_port=33 )

        assert( int(it.cluster_port) == 33)

        it.override(defaults, skipdefaults=True)
        assert( int(it.cluster_port) == 33)

        it.override(defaults)
        assert( it.cluster_port == defaults.get('cluster_port'))

    #TODO: additional tests confirm all variants of this command line are supported: 
    # add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])

    def test_yamlformat(self, tmpdir):
        fname="test_yamlformat.yaml"
        ffix = tmpdir.join(fname).write('cluster_port: 44')
        it=Config( fname=os.path.join( tmpdir, fname) )
        assert it.cluster_port == "44"


if __name__ == '__main__':
    #run as `python -m xbb_tools.config`
    import pytest
    pytest.main([__file__])

