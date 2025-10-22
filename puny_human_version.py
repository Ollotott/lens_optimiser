import math
import multiprocessing
import time


def get_lens_max(thickness, r1, r2):
    circle1_center_point = (0, -r1 + thickness / 2)
    circle2_center_point = (circle1_center_point[0], r2 - thickness / 2)

    d = circle1_center_point[1] - circle2_center_point[1]
    y_dist_from_center_c1 = abs((d ** 2 - r2 ** 2 + r1 ** 2)/(2*d))
    x_dist_from_center = abs((r1 ** 2 - y_dist_from_center_c1 ** 2)**0.5)

    intersection_point = (x_dist_from_center, circle1_center_point[1] + y_dist_from_center_c1)
    return intersection_point


def get_ray_angle(thickness, r1, r2, ray_shoot_pos):
    lens_max_r = get_lens_max(thickness, r1, r2)
    lens_max_l = (-lens_max_r[0], lens_max_r[1])

    a = lens_max_r[0] * 2
    b = vector_len(vector_gen(ray_shoot_pos, lens_max_r))
    c = vector_len(vector_gen(ray_shoot_pos, lens_max_l))

    return math.acos((a ** 2 - b ** 2 - c ** 2)/(-2*b*c))


def find_min_in_dict(dictionary):
    """Return the key (as float) whose numeric value is minimal."""
    if not dictionary:
        return None
    # dictionary keys already floats, values numeric
    min_key = min(dictionary.items(), key=lambda kv: float(kv[1]))[0]
    return float(min_key)


def average(list_of_values):
    if not list_of_values:
        return 0.0
    total = 0.0
    for value in list_of_values:
        total += value
    return total / len(list_of_values)


def standard_deviation(list_of_values):
    """Note: preserved your original 'deviation' implementation (mean absolute deviation)."""
    if not list_of_values:
        return 0.0
    avg = average(list_of_values)
    total_deviation = 0.0
    for value in list_of_values:
        total_deviation += abs(avg - value)
    return total_deviation / len(list_of_values)


def ray(start_coord, direction, variable):
    return (start_coord[0] + variable * direction[0]), (start_coord[1] + variable * direction[1])


def ray_lens_intersect(ray_start, ray_vector, circle_rad, circle_center):
    """Calculates the Intersect between the Ray line and the Circle. Spits out None if there is none."""
    dx = ray_start[0] - circle_center[0]
    dy = ray_start[1] - circle_center[1]

    A = ray_vector[0] ** 2 + ray_vector[1] ** 2
    B = 2 * (dx * ray_vector[0] + dy * ray_vector[1])
    C = dx ** 2 + dy ** 2 - circle_rad ** 2

    dis = B ** 2 - 4 * A * C
    if dis < 0 or A <= 0:
        return None, None
    else:
        t1 = (-B + dis ** 0.5) / (2 * A)
        t2 = (-B - dis ** 0.5) / (2 * A)

        point1 = ray(ray_start, ray_vector, t1)
        point2 = ray(ray_start, ray_vector, t2)

        # Takes the closer of two points if start is outside of circle
        if t1 > 0 and t2 > 0:
            if t2 > t1:
                return point1, True
            else:
                return point2, True
        # takes the positive of the t's when start is inside the circle
        elif t1 > 0:
            return point1, False
        else:
            return point2, False


def scalar_prod(vector_1, vector_2):
    return vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1]


def vector_len(vektor):
    return (vektor[0] ** 2 + vektor[1] ** 2) ** 0.5


def vector_angle(vector_1, vector_2):
    """Vektor winkel formel"""
    # safe acos guard
    denom = (vector_len(vector_1) * vector_len(vector_2))
    if denom == 0:
        return 0.0
    cosv = scalar_prod(vector_1, vector_2) / denom
    cosv = max(-1.0, min(1.0, cosv))
    return math.acos(cosv)


def turn_90_deg(vector, left):
    if left:
        return -vector[1], vector[0]
    else:
        return vector[1], -vector[0]


def vector_gen(tip, shaft):
    return tip[0] - shaft[0], tip[1] - shaft[1]


def pi_to_deg(number):
    return number / (2 * math.pi) * 360


def deg_to_rad(deg):
    return deg / 360 * math.pi * 2


def normalize_vector(vector):
    L = vector_len(vector)
    if L == 0:
        return 0.0, 0.0
    return vector[0] * (1 / L), vector[1] * (1 / L)


def angle_vector(vector, angle):
    """Turns a vector counterclockwise around its shaft"""
    return (vector[0] * math.cos(angle) - vector[1] * math.sin(angle)), (
            vector[0] * math.sin(angle) + vector[1] * math.cos(angle))


def cross(vector_1, vector_2):
    return vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0]


def get_changed_ray(ray_start, ray_vector, circle_rad, circle_center, r1, r2):
    """Takes a ray and finds the intersection with a circle and calculates how it should continue based on the refractive indexes. (Normalised vector)
    Returns the intersection (new starting point) and the new vector or None on failure."""
    intersect, outside = ray_lens_intersect(ray_start, ray_vector, circle_rad, circle_center)
    if intersect is None:
        return None

    circle_to_int_vector = vector_gen(intersect, circle_center)

    temp = False
    while True:
        tangent_to_circle_at_int = turn_90_deg(circle_to_int_vector, temp)
        incoming_angle = vector_angle(tangent_to_circle_at_int, ray_vector)
        if incoming_angle <= (0.5 * math.pi):
            incoming_angle = 0.5 * math.pi - incoming_angle
            break
        else:
            temp = True

    try:
        int_to_circle_vector = vector_gen(circle_center, intersect)
        # Snell
        sin_arg = (math.sin(incoming_angle) * r1) / r2
        # numeric guard
        if abs(sin_arg) > 1.0:
            return None
        outgoing_angle = math.asin(sin_arg)

        if not outside:
            int_to_circle_vector = -int_to_circle_vector[0], -int_to_circle_vector[1]

        # positive for CCW
        side = cross(normalize_vector(int_to_circle_vector), normalize_vector(ray_vector))

        # if side is negative, we need to have the angle in the function be positive.
        sign = 1 if side >= 0 else -1

        outgoing_vector = angle_vector(normalize_vector(int_to_circle_vector), sign * outgoing_angle)
        outgoing_vector = normalize_vector(outgoing_vector)
    except Exception:
        return None

    return intersect, outgoing_vector


def find_sensor_hit(starting_pos, vector, focal_len):
    # guard division by zero
    if vector[1] == 0:
        return None
    t = (-focal_len - starting_pos[1]) / vector[1]
    return vector[0] * t + starting_pos[0]


def trace_ray(x_start, focal_len, circle1_radius, circle2_radius, lens_thickness, ray_start_point=None,
              ray_dir_vector=None):
    """Takes a ray and shoots it through the lens and records where on the sensor plane it lands."""
    if ray_start_point is None:
        ray_start_point = (x_start, 200)
    if ray_dir_vector is None:
        ray_dir_vector = (0, -1)

    circle1_center_point = (0, -circle1_radius + lens_thickness / 2)
    circle2_center_point = (circle1_center_point[0], circle2_radius - lens_thickness / 2)

    refractive_index_1 = 1.5  # material 1
    refractive_index_2 = 1.3  # material 2

    try:
        out1 = get_changed_ray(ray_start_point, ray_dir_vector, circle1_radius,
                               circle1_center_point, 1, refractive_index_1)
        if out1 is None:
            return None
        intersection_point_1, ray_vector_1 = out1

        out2 = get_changed_ray(intersection_point_1, ray_vector_1, circle2_radius,
                               circle2_center_point, refractive_index_1, 1)
        if out2 is None:
            return None
        intersection_point_2, ray_vector_2 = out2
    except TypeError:
        return None

    return find_sensor_hit(intersection_point_2, ray_vector_2, focal_len)


def ray_trace_flower(start_coordinate, focal_length, lens_thickness, lens_curvature_1, lens_curvature_2,
                     number_of_rays):
    """Returns the sensor hits."""
    # we now get all the vectors we need.

    ray_angle = get_ray_angle(lens_thickness, lens_curvature_1, lens_curvature_2, start_coordinate)
    max_vector = angle_vector(vector_gen(get_lens_max(lens_thickness, lens_curvature_1, lens_curvature_2), start_coordinate), ray_angle / 2)
    step_size = ray_angle / number_of_rays

    count = 0
    ray_list = []

    # this creates a list of rays.
    while count <= number_of_rays:
        ray_list.append(angle_vector(max_vector, -step_size * count))
        count += 1

    sensor_hit_locations = []
    for current_ray in ray_list:
        hit = trace_ray(0, focal_length, lens_curvature_1, lens_curvature_2, lens_thickness, start_coordinate,
                        current_ray)
        if hit is not None:
            sensor_hit_locations.append(hit)

    return sensor_hit_locations


def ray_trace_flower_string_and_score(start_coordinate, focal_length, lens_thickness, lens_curvature_1,
                                      lens_curvature_2,
                                      number_of_rays, distance_to_travel, steps_to_take, q, first_curve_as_key):
    """Returns the score for the lens."""
    scored_standard_dev_list = []
    number_of_hits = []

    step_size = distance_to_travel / steps_to_take
    step_count = -1
    while step_count <= steps_to_take:
        step_count += 1
        current_pos = (start_coordinate[0] + step_size * step_count, start_coordinate[1])

        sensor_hits = ray_trace_flower(current_pos, focal_length, lens_thickness, lens_curvature_1, lens_curvature_2, number_of_rays)

        number_of_hits.append(len(sensor_hits))

        if len(sensor_hits) > 0:
            standard_dev = standard_deviation(sensor_hits)
            scored_standard_dev = standard_dev * (number_of_rays / (0.5 * len(sensor_hits)))
            scored_standard_dev_list.append(scored_standard_dev)
        else:
            continue

    # If we have no scored entries, return inf score so this curvature is ignored.
    if not scored_standard_dev_list:
        if first_curve_as_key:
            q.put({float(lens_curvature_1): float('inf')})
        else:
            q.put({float(lens_curvature_2): float('inf')})
    else:
        if first_curve_as_key:
            q.put({float(lens_curvature_1): (float(average(scored_standard_dev_list)), average(number_of_hits))})
        else:
            q.put({float(lens_curvature_2): (float(average(scored_standard_dev_list)), average(number_of_hits))})


def find_optimal_curvature(focal_length, lens_thickness, ray_start, ray_angle, number_of_rays, resolution):
    """The ray angle has to be in radians."""
    min_curvature = 250
    max_curvature = 400

    max_vector = angle_vector(vector_gen((0, 0), ray_start), ray_angle / 2)
    step_size = ray_angle / number_of_rays

    # generate a list of all the vectors that we will scan.
    count = 0
    ray_list = []
    while count <= number_of_rays:
        ray_list.append(angle_vector(max_vector, -step_size * count))
        count += 1

    f = min_curvature
    curvatures = {}  # Holds the sensor distance results from a single lens so that they can be averaged

    while f <= max_curvature:
        distance_list = []
        for angled_vector in ray_list:
            try:
                sensor_result = trace_ray(0, focal_length, f, f, lens_thickness, ray_start, angled_vector)
                if -18 <= sensor_result <= 18:
                    distance_list.append(sensor_result)
            except TypeError:
                # print(f"Couldn't solve for x={x}, as there is no intersection.")
                continue
            except ValueError:
                # print(f"{x} Light is reflected back into the lens")
                continue

        curvatures[str(f)] = standard_deviation(distance_list), len(distance_list)
        f += resolution
        f = round(f, 2)

    temp_list = []
    for curve, standard_dev in curvatures.items():
        temp_list.append(standard_dev[0])

    min_standard_dev = min(temp_list)
    for curve, standard_dev in curvatures.items():
        if standard_dev[0] == min_standard_dev:
            return float(curve)  # The return file is: (standard deviation, curvature)


def find_optimal_focal_distance(circle1_radius, circle2_radius, lens_thickness):
    min_focal_length = 10
    max_focal_length = 1000
    f = min_focal_length
    focal_lengths = {}

    testing_rays = 100
    max_x = get_lens_max(lens_thickness, circle1_radius, circle2_radius)[0]
    step_size = 2 * max_x / testing_rays
    while f <= max_focal_length:
        distance_list = []

        x = -max_x

        while x <= max_x:
            try:
                sensor_result = trace_ray(x, f, circle1_radius, circle2_radius, lens_thickness)
                # print(f"---For x={x}, {sensor_result}")
                if -18 <= sensor_result <= 18:
                    distance_list.append(sensor_result)
                x += step_size
            except TypeError:
                # print(f"Couldn't solve for x={x}, as there is no intersection.")
                x += step_size
                continue
            except ValueError:
                # print(f"{x} Light is reflected back into the lens")
                x += step_size

        focal_lengths[str(f)] = standard_deviation(distance_list)
        f += 0.5

    temp_list = []
    for focal_len, standard_dev in focal_lengths.items():
        temp_list.append(standard_dev)
        # print(f"{focal_len}mm: {standard_dev} mm of deviation")

    min_standard_dev = min(temp_list)
    for focal_len, standard_dev in focal_lengths.items():
        if standard_dev == min_standard_dev:
            print(f"Minimum Standard deviation is: {min_standard_dev}, With a focal length of {focal_len}")
            break
    return min_standard_dev


if __name__ == "__main__":
    start = time.time()

    # defining starting parameters
    goal_focal_length = 100
    goal_thickness = 40

    object_distance = 3000

    flower_rays = 50
    flower_number = 100

    curve_resolution = 1
    curve_seeking_radius = 100

    cycles = 20
    batch_size = 6

    flower_string_length = math.tan(2*math.atan(36/(2*goal_focal_length))) * object_distance

    # let us commence.
    preliminary_curvature = find_optimal_curvature(goal_focal_length, goal_thickness, (0, object_distance),
                                                   5,
                                                   flower_rays, curve_resolution)
    print(f"Preliminary Starting Curvature: {preliminary_curvature}")
    # find_optimal_focal_distance(preliminary_curvature, preliminary_curvature, goal_thickness)

    best_curve_1 = preliminary_curvature
    best_curve_2 = preliminary_curvature

    result_archive = []

    cycle_counter = 0
    while cycle_counter < cycles:
        if (best_curve_1, best_curve_2) in result_archive:
            curve_resolution = curve_resolution * 0.75
            curve_seeking_radius = curve_seeking_radius * 0.75
            print("New Radii", curve_resolution, curve_seeking_radius)
            better_res_run = True
        else:
            better_res_run = False

        result_archive.append((best_curve_1, best_curve_2))

        curve_1_old = best_curve_1
        curve_2_old = best_curve_2

        # set the lower search bound for this round and make sure it don't get too low
        current_rad_1 = best_curve_1 - curve_seeking_radius
        if current_rad_1 < goal_thickness:
            current_rad_1 = goal_thickness

        current_rad_2 = best_curve_2 - curve_seeking_radius
        if current_rad_2 < goal_thickness:
            current_rad_2 = goal_thickness

        current_results = {}
        current_results_ray_count = {}

        while current_rad_1 <= best_curve_1 + curve_seeking_radius:
            q = multiprocessing.Queue()
            curvatures_to_test = [current_rad_1 + i * curve_resolution for i in range(batch_size)]
            processes = []

            for c in curvatures_to_test:
                p = multiprocessing.Process(target=ray_trace_flower_string_and_score,
                                            args=((0, object_distance), goal_focal_length, goal_thickness,
                                                  c, best_curve_2, flower_rays,
                                                  flower_string_length, flower_number, q, True))

                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            results_batch = [q.get() for _ in processes]
            for item in results_batch:
                for thread_curve, thread_res in item.items():
                    current_results[float(thread_curve)] = float(thread_res[0])
                    current_results_ray_count[float(thread_curve)] = float(thread_res[1])

            current_rad_1 += batch_size * curve_resolution

        # present the best curve
        best_curve_1 = find_min_in_dict(current_results)
        print(f"Best R1 from Cycle Number {cycle_counter}: {best_curve_1} with a score of {current_results[best_curve_1]}, {current_results_ray_count[best_curve_1]} rays hit the back plate.")

        # we check if the current best r1 has already been found with this resolution, if it is true, then we restart.
        if better_res_run is False and (best_curve_1, best_curve_2) in result_archive:
            print("Skipped searching for R2, as we have already had this result.")
            cycle_counter += 1
            continue


        # now let us find the best second curvature!
        current_results = {}
        current_results_ray_count = {}
        while current_rad_2 <= best_curve_2 + curve_seeking_radius:

            q = multiprocessing.Queue()
            curvatures_to_test = [current_rad_2 + i * curve_resolution for i in range(batch_size)]
            processes = []

            for c in curvatures_to_test:
                p = multiprocessing.Process(target=ray_trace_flower_string_and_score,
                                            args=((0, object_distance), goal_focal_length, goal_thickness,
                                                  best_curve_1, c, flower_rays,
                                                  flower_string_length, flower_number, q, False))

                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            results_batch = [q.get() for _ in processes]

            for item in results_batch:
                for thread_curve, thread_res in item.items():
                    current_results[float(thread_curve)] = float(thread_res[0])
                    current_results_ray_count[float(thread_curve)] = float(thread_res[1])

            current_rad_2 += batch_size * curve_resolution

        # present the best curve
        best_curve_2 = find_min_in_dict(current_results)
        print(f"Best R2 from Cycle Number {cycle_counter}: {best_curve_2} with a score of {current_results[best_curve_2]}, {current_results_ray_count[best_curve_2]} rays hit the back plate.")

        cycle_counter += 1

    find_optimal_focal_distance(best_curve_1, best_curve_2, goal_thickness)
    print(time.time() - start)
