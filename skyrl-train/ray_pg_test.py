#!/usr/bin/env python3
import sys
import ray
from ray.util.placement_group import placement_group, remove_placement_group

ray.init()
pg = placement_group([{"GPU": 4}], strategy="PACK", name="test_pg")

try:
    ray.get(pg.ready(), timeout=60)
    print(f"Placement group created successfully: {pg.id}")
except Exception as e:
    print(f"Failed to create placement group requiring 2 GPUs in 60s. Error: {e}")
    sys.exit(1)
finally:
    try:
        remove_placement_group(pg)
    except Exception:
        pass
    ray.shutdown()