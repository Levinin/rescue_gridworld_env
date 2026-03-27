# ---- Tile codes ----
EMPTY = 0
WALL = 5
DOOR_LOCKED = 10
DOOR_UNLOCKED = 20
CUPBOARD_LOCKED = 30
CUPBOARD_UNLOCKED = 40
CUPBOARD_UNLOCKED_KEYCARD = 50
KEY_TILE = 60  # keys are placed "on" open tiles; this tile is passable
PERSON_TILE = 70
EXIT = 80
UNKNOWN = 255



TILE_NAMES = {
    EMPTY: "empty",
    WALL: "wall",
    DOOR_LOCKED: "door_locked",
    DOOR_UNLOCKED: "door_unlocked",
    CUPBOARD_LOCKED: "cupboard_locked",
    CUPBOARD_UNLOCKED: "cupboard_unlocked",
    KEY_TILE: "key",
    EXIT: "exit",
}

# ---- Actions ----
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
PICK_KEY = 4
UNLOCK_CUPBOARD = 5
PICK_KEYCARD = 6
UNLOCK_DOOR = 7
TALK_PERSON = 8

ACTION_NAMES = {
    UP: "up",
    DOWN: "down",
    LEFT: "left",
    RIGHT: "right",
    PICK_KEY: "pick_key",
    UNLOCK_CUPBOARD: "unlock_cupboard",
    PICK_KEYCARD: "pick_keycard",
    UNLOCK_DOOR: "unlock_door",
    TALK_PERSON: "talk_person",
}
