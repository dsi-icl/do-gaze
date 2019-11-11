import pstats
from pstats import SortKey
p = pstats.Stats('profile')
p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats(SortKey.CUMULATIVE)
p.print_stats()