import sys
try:
  import yaml
except ImportError:
  print "You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions."
  sys.exit()
"""
from collections import OrderedDict

def OrderedLoad(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
  class OrderedLoader(Loader):
    pass
  def construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return object_pairs_hook(loader.construct_pairs(node))
  OrderedLoader.add_constructor(
      yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
      construct_mapping)
  return yaml.load(stream, OrderedLoader)
"""

def readConfig( filename ):
  print "Tensile::ReadYAML::ReadConfig( %s )" % ( filename )
  try:
    stream = open(filename, "r")
  except IOError:
    print "Tensile::ReadYAML::ReadConfig ERROR: Cannot open config file: %s" % filename
    sys.exit(1)
  config = yaml.load(stream, yaml.SafeLoader)
  stream.close()
  return config

"""
# Candidate Parameter List
- single kernel or allow multiple kernels
- micro tiles
- ppd
- unrolls
# Solution Parameter List
- kernel grid
# Kernel Parameter List Exact
- precision
- transpose
- ppd initial stride
# Kernel Parameter List Same For All Tiles
- branch type
- interleaving
- global memory read/write pattern (para/perp) for C, A, B
  - float vs float4
  - num loads para vs perp
  - each threads reads contiguous 8 elements
  - which C elements in charge of (adjacent or strided)
- do-while vs for loops
- prefetch vs not prefetch (same for all b/c either handles or doesn't)
# Kernel Parameter List Different Per Tile
- work-group {8x8, 16x16}
- micro-tile {2x2, 2x4, 4x4, 4x8, 8x8}
- work-group order for cacheing
- num sub-groups for splitting K
- unroll

"""
