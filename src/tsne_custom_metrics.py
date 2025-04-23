import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


class ConvexHull2D:
    def __init__(self, points):
        self.points = np.array(points)
        self.hull = []

    def compute_hull(self):
        points = sorted(self.points.tolist())
        if len(points) <= 1:
            self.hull = points
            return

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        self.hull = lower[:-1] + upper[:-1]

    def plot(self):
        points = self.points
        hull = np.array(self.hull + [self.hull[0]])  # close the hull

        plt.figure()
        plt.plot(points[:, 0], points[:, 1], 'o', label='Points')
        plt.plot(hull[:, 0], hull[:, 1], 'r-', lw=2, label='Convex Hull')
        plt.fill(hull[:, 0], hull[:, 1], color="blue", alpha=0.2, label='Convex Hull')
        plt.legend()
        plt.title("Convex Hull (Graham Scan)")
        plt.show()


class CustomMetrics:
    def __init__(self, X, Y, tolerance_percent=0):
        self.X = X
        self.Y = Y
        self.unique_labels = np.unique(Y)
        self.X_split = {label: X[Y == label] for label in self.unique_labels}
        self.hulls = {}
        self.removed_points = {label: [] for label in self.unique_labels}
        for i in self.unique_labels:
            ch = ConvexHull2D(self.X_split[i])
            ch.compute_hull()
            self.hulls[i] = ch

        self.remove_outliers(tolerance_percent)

    def calculate_cluster_score(self, hulls=None):
        if hulls is None:
            hulls = self.hulls
        total_area = 0
        for i in hulls:
            total_area += self.polygon_area(hulls[i].hull)

        hull_points = [hulls[i].hull for i in hulls]
        intersection_area = 0
        for i in range(len(hull_points)):
            for j in range(i + 1, len(hull_points)):
                intersection_area += self.convex_hull_intersection_area(hull_points[i], hull_points[j])
        score = 1 - intersection_area / total_area
        return score

    def remove_outlier(self, label):
        points = self.X_split[label]
        hull_points = self.hulls[label].hull

        if len(hull_points) <= 3:
            print(f"Not enough points in hull for label {label} to remove an outlier.")
            return

        max_edge_sum = -1
        remove_idx = -1

        for i in range(len(hull_points)):
            prev_point = np.array(hull_points[i - 1])
            curr_point = np.array(hull_points[i])
            next_point = np.array(hull_points[(i + 1) % len(hull_points)])

            left_edge = np.linalg.norm(curr_point - prev_point)
            right_edge = np.linalg.norm(curr_point - next_point)
            edge_sum = left_edge + right_edge

            if edge_sum > max_edge_sum:
                max_edge_sum = edge_sum
                remove_idx = i

        point_to_remove = np.array(hull_points[remove_idx])
        self.removed_points[label].append(list(point_to_remove))

        # Remove this point from the original point set
        mask = ~np.all(points == point_to_remove, axis=1)
        self.X_split[label] = points[mask]

        # Recompute the convex hull
        new_ch = ConvexHull2D(self.X_split[label])
        new_ch.compute_hull()
        self.hulls[label] = new_ch

        # print(f"Removed outlier {point_to_remove} from class {label}")

    def remove_outliers(self, tolerance_percent):
        n = int((tolerance_percent / 100) * len(self.X) / len(self.unique_labels))
        for i in range(n):
            for label in self.unique_labels:
                self.remove_outlier(label)

    def plot_hulls(self, label=False):
        plt.figure()
        for i in self.unique_labels:
            points = self.X_split[i]
            hull = np.array(self.hulls[i].hull + [self.hulls[i].hull[0]])
            plt.plot(points[:, 0], points[:, 1], 'o')
            if len(self.removed_points[i]) > 0:
                removed_arr = np.array(self.removed_points[i])
                if removed_arr.ndim == 1:
                    removed_arr = removed_arr.reshape(1, -1)
                plt.plot(removed_arr[:, 0], removed_arr[:, 1], 'o')
            plt.plot(hull[:, 0], hull[:, 1], 'r-', lw=2, label=(f'Convex Hull{i}' if label else None))
            plt.fill(hull[:, 0], hull[:, 1], alpha=0.2, label=(f'Convex Hull{i}' if label else None))
        plt.legend()
        plt.title("Convex Hull (Graham Scan)")
        plt.show()

    def polygon_area(self, poly):
        """Compute area of a polygon using the shoelace formula."""
        x, y = zip(*poly)
        return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(poly) - 1)))

    def inside(self, p, edge_start, edge_end):
        """Check if point p is inside the edge defined by edge_start -> edge_end."""
        return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) - (edge_end[1] - edge_start[1]) * (
                p[0] - edge_start[0]) >= 0

    def compute_intersection(self, p1, p2, q1, q2):
        """Compute intersection point of lines p1->p2 and q1->q2."""
        A1, B1 = p2[1] - p1[1], p1[0] - p2[0]
        C1 = A1 * p1[0] + B1 * p1[1]

        A2, B2 = q2[1] - q1[1], q1[0] - q2[0]
        C2 = A2 * q1[0] + B2 * q1[1]

        det = A1 * B2 - A2 * B1
        if det == 0:
            return None  # Parallel lines

        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return (x, y)

    def sutherland_hodgman(self, subject_polygon, clip_polygon):
        """Perform polygon clipping using the Sutherland–Hodgman algorithm."""
        output_list = subject_polygon
        for i in range(len(clip_polygon)):
            input_list = output_list
            output_list = []

            A = clip_polygon[i]
            B = clip_polygon[(i + 1) % len(clip_polygon)]

            for j in range(len(input_list)):
                P = input_list[j]
                Q = input_list[(j + 1) % len(input_list)]

                if self.inside(Q, A, B):
                    if not self.inside(P, A, B):
                        intersection = self.compute_intersection(P, Q, A, B)
                        if intersection:
                            output_list.append(intersection)
                    output_list.append(Q)
                elif self.inside(P, A, B):
                    intersection = self.compute_intersection(P, Q, A, B)
                    if intersection:
                        output_list.append(intersection)

        return output_list

    def convex_hull_intersection_area(self, hull1, hull2):
        """Compute the area of intersection of two convex hulls."""
        intersect_poly = self.sutherland_hodgman(hull1, hull2)
        if len(intersect_poly) < 3:
            return 0.0
        return self.polygon_area(intersect_poly)

    def cluster_splitting_without_graph(self):
        self.sub_hulls = {}

        for label in self.unique_labels:
            points = self.X_split[label]
            clustering = DBSCAN(eps=5, min_samples=5).fit(points)
            cluster_labels = np.unique(clustering.labels_)

            for cl in cluster_labels:
                if cl == -1:
                    continue

                cluster_points = points[clustering.labels_ == cl]
                if len(cluster_points) >= 3:
                    ch = ConvexHull2D(cluster_points)
                    ch.compute_hull()
                    if label not in self.sub_hulls:
                        self.sub_hulls[label] = []
                    self.sub_hulls[label].append(ch)
        score_hulls = {}
        index = 0
        for i in self.sub_hulls.values():
            for j in i:
                score_hulls[index] = j
                index += 1
        return self.calculate_cluster_score(score_hulls)

    def cluster_splitting(self):
        self.sub_hulls = {}

        for label in self.unique_labels:
            points = self.X_split[label]
            clustering = DBSCAN(eps=5, min_samples=5).fit(points)
            cluster_labels = np.unique(clustering.labels_)
            print(cluster_labels)

            for cl in cluster_labels:
                if cl == -1:
                    continue  # skip noise

                cluster_points = points[clustering.labels_ == cl]
                if len(cluster_points) >= 3:
                    ch = ConvexHull2D(cluster_points)
                    ch.compute_hull()
                    if label not in self.sub_hulls:
                        self.sub_hulls[label] = []
                    self.sub_hulls[label].append(ch)

        plt.figure()
        label = False
        for i in self.unique_labels:
            points = self.X_split[i]
            plt.plot(points[:, 0], points[:, 1], 'o')
            if i in self.sub_hulls:
                for h in self.sub_hulls[i]:
                    hull = np.array(h.hull + [h.hull[0]])
                    plt.plot(hull[:, 0], hull[:, 1], 'r-', lw=2, label=(f'Convex Hull{i}' if label else None))
                    plt.fill(hull[:, 0], hull[:, 1], alpha=0.2, label=(f'Convex Hull{i}' if label else None))
        plt.legend()
        plt.title("Convex Hull (Graham Scan)")
        plt.show()
        score_hulls = {}
        index = 0
        for i in self.sub_hulls.values():
            for j in i:
                score_hulls[index] = j
                index += 1
        return self.calculate_cluster_score(score_hulls)
