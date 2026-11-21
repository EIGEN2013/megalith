"""The enlarged village of Ellenhar: where the new buildings, the low town wall, its gates and the streets go.

Game axes (metres): x east, z south, y up. The old village sits around (0, 3) within ~28 m; the sea lies to the
north (z < -15) and east (x > ~50). The wall closes the landward side from the north-west shore round to the
east cliff; the sea side stays open to the quay.

Read by tools/build_land.py (clearing, streets in the ground layers), tools/hq_village.py (the models) and,
through assets_src/village_ext/village_ext.json, by the game.
"""
import math

# the wall's centre line, north-west shore -> west -> south -> east cliff; corners are rounded when built
WALL = [(-27.0, -39.0), (-41.0, -26.0), (-43.0, 2.5), (-43.0, 40.0), (-37.0, 61.0), (-16.0, 68.0), (0.4, 68.0),
        (24.0, 68.0), (43.0, 62.0), (50.0, 42.0), (51.5, 16.0), (49.0, -6.0)]
GATES = [  # where a road passes the wall: an arched gatehouse
    {"id": "gate_west", "x": -43.0, "z": 2.5, "road": "west"},
    {"id": "gate_south", "x": 0.4, "z": 68.0, "road": "south"},
]
WALL_HEIGHT = 2.1          # above the ground, to the top of the coping
WALL_THICK = 0.9

# streets inside the wall (dirt and gravel; the old village keeps its cobbles)
STREETS = [
    [(-46.0, 2.5), (-34.0, 3.2), (-22.0, 3.6)],                     # west gate -> plaza
    [(0.4, 71.0), (0.6, 50.0), (0.4, 30.0), (0.0, 24.0)],           # south gate -> plaza
    [(13.0, 15.0), (24.0, 10.0), (34.5, 5.5)],                      # plaza -> Bren's cottage
    [(0.6, 46.0), (-11.0, 46.0)],                                   # to the chapel door
    [(0.6, 40.0), (10.5, 40.0)],                                    # to the thatched cottage
    [(0.6, 58.0), (10.5, 58.0)],                                    # to the bakery
    [(0.5, 31.0), (-9.5, 31.0)],                                    # to the tall house
    [(-29.0, 3.4), (-29.0, -3.5)],                                  # the tavern's yard
    [(-30.0, 3.4), (-30.0, 11.5)],                                  # the smithy's yard
]
STREET_HALF = 1.6

# the new buildings: centre, the way the front faces (game x, z), and the kind of building
BUILDINGS = [
    {"id": "tavern", "kind": "tavern", "x": -29.0, "z": -9.5, "face": (0, 1), "name": "갈매기 여관"},
    {"id": "smithy", "kind": "smithy", "x": -30.0, "z": 16.5, "face": (0, -1), "name": "대장간"},
    {"id": "chapel", "kind": "chapel", "x": -19.5, "z": 46.0, "face": (1, 0), "name": "바닷가 예배당"},
    {"id": "tallhouse", "kind": "tallhouse", "x": -14.0, "z": 31.0, "face": (1, 0), "name": "높은 집"},
    {"id": "cottage", "kind": "cottage", "x": 15.5, "z": 40.0, "face": (-1, 0), "name": "초가집"},
    {"id": "bakery", "kind": "bakery", "x": 15.5, "z": 58.0, "face": (-1, 0), "name": "빵집"},
    {"id": "fisher", "kind": "fisher", "x": 41.0, "z": 5.0, "face": (-1, 0), "name": "브렌의 집"},
]
# an orchard inside the wall (oaks and ashes from the land's tree set, so they shed fruit when shaken)
ORCHARD = [(30.0, 30.0), (37.0, 33.0), (44.0, 29.0), (32.0, 39.0), (40.0, 42.0), (46.0, 37.0), (34.0, 48.0), (42.0, 52.0),
           (27.0, 50.0), (-32.0, 30.0), (-35.0, 40.0), (-31.0, 55.0)]


def yaw_of(face):
    """Blender Z rotation (= the game collider yaw) that turns a front facing -Y toward `face` (game x, z)."""
    return math.atan2(face[0], face[1])


def wall_polygon():
    """The walled area as a closed polygon (game x, z): the wall line, closed along the shore."""
    return WALL + [(40.0, -14.0), (20.0, -26.0), (0.0, -30.0), (-16.0, -38.0)]


def inside(x, z, poly=None):
    poly = poly or wall_polygon()
    c = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, zi = poly[i]
        xj, zj = poly[j]
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi + 1e-12) + xi:
            c = not c
        j = i
    return c


def dist_to_wall(x, z):
    best = 1e9
    for (ax, az), (bx, bz) in zip(WALL[:-1], WALL[1:]):
        dx, dz = bx - ax, bz - az
        t = max(0.0, min(1.0, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz)))
        best = min(best, math.hypot(x - ax - t * dx, z - az - t * dz))
    return best
