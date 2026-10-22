#!/usr/bin/env python3
import numpy as np
from build_p31_future_reference import yearwise_top25
from p31_core import mean_opposition

rng = np.random.default_rng(20260928)
years = np.repeat([2018, 2019, 2020, 2021], [793, 399, 397, 400])
ids = np.asarray([f'{i:064x}' for i in range(len(years))])
members = rng.uniform(.01, .99, size=(len(years), 4))
p = members.mean(axis=1)
route = yearwise_top25(mean_opposition(members, p), years, ids)
assert [int(route[years == y].sum()) for y in (2018, 2019, 2020, 2021)] == [198, 100, 99, 100]
assert int(route.sum()) == 497
print('P31_C43_FUTURE_SELFTEST_PASS')
print('route_by_year=2018:198,2019:100,2020:99,2021:100 total=497')
