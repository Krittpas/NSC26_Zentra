# modules/safety_zone.py — Safety Zone (stub — module not yet implemented)

zones = []

MP_OK = False


def on_frame(frame, meta, window_title=""):
    pass


def on_data(data, meta, frame=None):
    pass


def mouse_callback(event, x, y, flags, param):
    pass


def toggle_draw_mode():
    pass


def clear_all_zones():
    pass


def get_exclusion_polygons():
    return []


def _load_zones():
    pass


def send_line_notify(msg, image=None, level="warning", **kwargs):
    pass
