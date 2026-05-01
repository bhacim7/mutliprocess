import math

def calculate_obj_gps(lat1, lon1, dist_m, bearing_deg):
    """
    Calculates target GPS coordinates given current GPS, distance(m), and bearing(deg).
    """
    R = 6378137.0  # Earth radius
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    bearing_rad = math.radians(bearing_deg)

    lat2_rad = math.asin(math.sin(lat1_rad) * math.cos(dist_m / R) +
                         math.cos(lat1_rad) * math.sin(dist_m / R) * math.cos(bearing_rad))

    lon2_rad = lon1_rad + math.atan2(math.sin(bearing_rad) * math.sin(dist_m / R) * math.cos(lat1_rad),
                                     math.cos(dist_m / R) - math.sin(lat1_rad) * math.sin(lat2_rad))

    return math.degrees(lat2_rad), math.degrees(lon2_rad)

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth."""
    # Stub for future nav process, just implemented here for completeness
    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates bearing between two GPS points"""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lam1 = math.radians(lon1)
    lam2 = math.radians(lon2)
    y = math.sin(lam2 - lam1) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(lam2 - lam1)
    theta = math.atan2(y, x)
    return (math.degrees(theta) + 360) % 360

def signed_angle_difference(angle1, angle2):
    """Calculates signed angle difference between two angles in degrees"""
    diff = (angle2 - angle1 + 180) % 360 - 180
    return diff
