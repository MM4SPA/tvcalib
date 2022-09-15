import numpy as np

from SoccerNet.Evaluation.utils_calibration import Camera, SoccerPitch


def compound_score(completeness, acc_5, acc_10, acc_20):
    # input values in range [0, 1] not percentage
    return (1 - np.exp(-4 * completeness)) * (0.5 * acc_5 + 0.35 * acc_10 + 0.15 * acc_20)


def _get_cam_from_decomp(h, width, height, wrap_delta, zeta):
    # for decomposition and refinement we can only use planar points
    field = SoccerPitch2D()
    cam_d = Camera(width, height)

    sucsess = cam_d.from_homography(np.linalg.inv(h))
    if not sucsess:
        # print("Cannot perform Homography Decomposition")
        return None

    cam_h = HomographySN(h)
    # field.point_dict() -> minimum set of key points is enough as we are not restricted to detected points
    point_matches = []
    for _, p3D in field.point_dict.items():
        # project key point from estimated homography
        p2D_h = cam_h.project_point(p3D.copy())
        # project key point from initial camera estimation
        p2D_c = cam_d.project_point(p3D.copy())

        # we introduce a delta to catch some keypoints already close to the visible (image) area
        wrap_delta_h = wrap_delta * height  # in px
        wrap_delta_w = wrap_delta * width
        if (
            p2D_h[0] < width + wrap_delta_w
            and p2D_h[0] >= 0 - wrap_delta_w
            and p2D_h[1] < height + wrap_delta_h
            and p2D_h[1] >= 0 - wrap_delta_h
        ):
            dist = np.linalg.norm(p2D_h - p2D_c)
            try:
                if dist < zeta:
                    point_matches.append((p3D, p2D_h[:2]))  # ([Nx3], [Nx2]) i.e. ([X,Y,0], [x,y,0])
            except:
                print(dist, zeta, type(dist), type(zeta))
    if len(point_matches) > 3:
        cam_d.refine_camera(point_matches)
        return cam_d
    else:
        # print(prediction["image_ids"], "point_matches <= 3")
        # reject entire sample
        # ALTERNATIVE: return before refinement
        return None


class HomographySN:
    def __init__(self, prediction: np.ndarray) -> None:
        """Project flat points from a homography matrix.

        Args:
            prediction (np.ndarray): Homography matrix H that maps from image to world (template) according to the template and axis definitions from SN
        """
        assert len(prediction.shape) == 2
        assert prediction.shape[0] == 3
        assert prediction.shape[1] == 3

        self.H = prediction  # maps from 2D image -> 2D world (template)
        self.H_inv = np.linalg.inv(prediction)  #  maps from 2D world (template) -> 2D image

    def project_point(self, p, distort=False):
        assert distort == False
        # we need to manually set p_z = 1 for plane-to-plane mapping
        p[-1] = 1.0
        # projets a template point (x, y, z) to image
        # assume that z == 0.0
        projected = self.H_inv @ p
        projected /= projected[2]
        return projected  # [x, y, 1]


class SoccerPitch2D:
    """SN SoccerPitch that contains only segments that lie on one plane.
    Note: Used for evaluation only.
    """

    GOAL_LINE_TO_PENALTY_MARK = 11.0
    PENALTY_AREA_WIDTH = 40.32
    PENALTY_AREA_LENGTH = 16.5
    GOAL_AREA_WIDTH = 18.32
    GOAL_AREA_LENGTH = 5.5
    CENTER_CIRCLE_RADIUS = 9.15
    GOAL_HEIGHT = 2.44
    GOAL_LENGTH = 7.32

    lines_classes = [
        "Big rect. left bottom",
        "Big rect. left main",
        "Big rect. left top",
        "Big rect. right bottom",
        "Big rect. right main",
        "Big rect. right top",
        "Circle central",
        "Circle left",
        "Circle right",
        "Line unknown",
        "Middle line",
        "Side line bottom",
        "Side line left",
        "Side line right",
        "Side line top",
        "Small rect. left bottom",
        "Small rect. left main",
        "Small rect. left top",
        "Small rect. right bottom",
        "Small rect. right main",
        "Small rect. right top",
    ]

    symetric_classes = {
        "Side line top": "Side line bottom",
        "Side line bottom": "Side line top",
        "Side line left": "Side line right",
        "Middle line": "Middle line",
        "Side line right": "Side line left",
        "Big rect. left top": "Big rect. right bottom",
        "Big rect. left bottom": "Big rect. right top",
        "Big rect. left main": "Big rect. right main",
        "Big rect. right top": "Big rect. left bottom",
        "Big rect. right bottom": "Big rect. left top",
        "Big rect. right main": "Big rect. left main",
        "Small rect. left top": "Small rect. right bottom",
        "Small rect. left bottom": "Small rect. right top",
        "Small rect. left main": "Small rect. right main",
        "Small rect. right top": "Small rect. left bottom",
        "Small rect. right bottom": "Small rect. left top",
        "Small rect. right main": "Small rect. left main",
        "Circle left": "Circle right",
        "Circle central": "Circle central",
        "Circle right": "Circle left",
        "Line unknown": "Line unknown",
    }

    # RGB values
    palette = {
        "Big rect. left bottom": (127, 0, 0),
        "Big rect. left main": (102, 102, 102),
        "Big rect. left top": (0, 0, 127),
        "Big rect. right bottom": (86, 32, 39),
        "Big rect. right main": (48, 77, 0),
        "Big rect. right top": (14, 97, 100),
        "Circle central": (0, 0, 255),
        "Circle left": (255, 127, 0),
        "Circle right": (0, 255, 255),
        "Line unknown": (0, 0, 0),
        "Middle line": (255, 255, 0),
        "Side line bottom": (255, 0, 255),
        "Side line left": (0, 255, 150),
        "Side line right": (0, 230, 0),
        "Side line top": (230, 0, 0),
        "Small rect. left bottom": (0, 150, 255),
        "Small rect. left main": (254, 173, 225),
        "Small rect. left top": (87, 72, 39),
        "Small rect. right bottom": (122, 0, 255),
        "Small rect. right main": (255, 255, 255),
        "Small rect. right top": (153, 23, 153),
    }

    def __init__(self, pitch_length=105.0, pitch_width=68.0):
        """
        Initialize 3D coordinates of all elements of the soccer pitch.
        :param pitch_length: According to FIFA rules, length belong to [90,120] meters
        :param pitch_width: According to FIFA rules, length belong to [45,90] meters
        """
        self.PITCH_LENGTH = pitch_length
        self.PITCH_WIDTH = pitch_width

        self.center_mark = np.array([0, 0, 0], dtype="float")
        self.halfway_and_bottom_touch_line_mark = np.array([0, pitch_width / 2.0, 0], dtype="float")
        self.halfway_and_top_touch_line_mark = np.array([0, -pitch_width / 2.0, 0], dtype="float")
        self.halfway_line_and_center_circle_top_mark = np.array(
            [0, -SoccerPitch2D.CENTER_CIRCLE_RADIUS, 0], dtype="float"
        )
        self.halfway_line_and_center_circle_bottom_mark = np.array(
            [0, SoccerPitch2D.CENTER_CIRCLE_RADIUS, 0], dtype="float"
        )
        self.bottom_right_corner = np.array(
            [pitch_length / 2.0, pitch_width / 2.0, 0], dtype="float"
        )
        self.bottom_left_corner = np.array(
            [-pitch_length / 2.0, pitch_width / 2.0, 0], dtype="float"
        )
        self.top_right_corner = np.array([pitch_length / 2.0, -pitch_width / 2.0, 0], dtype="float")
        self.top_left_corner = np.array([-pitch_length / 2.0, -34, 0], dtype="float")

        self.left_goal_bottom_left_post = np.array(
            [-pitch_length / 2.0, SoccerPitch2D.GOAL_LENGTH / 2.0, 0.0], dtype="float"
        )
        self.left_goal_top_left_post = np.array(
            [-pitch_length / 2.0, SoccerPitch2D.GOAL_LENGTH / 2.0, -SoccerPitch2D.GOAL_HEIGHT],
            dtype="float",
        )
        self.left_goal_bottom_right_post = np.array(
            [-pitch_length / 2.0, -SoccerPitch2D.GOAL_LENGTH / 2.0, 0.0], dtype="float"
        )
        self.left_goal_top_right_post = np.array(
            [-pitch_length / 2.0, -SoccerPitch2D.GOAL_LENGTH / 2.0, -SoccerPitch2D.GOAL_HEIGHT],
            dtype="float",
        )

        self.right_goal_bottom_left_post = np.array(
            [pitch_length / 2.0, -SoccerPitch2D.GOAL_LENGTH / 2.0, 0.0], dtype="float"
        )
        self.right_goal_top_left_post = np.array(
            [pitch_length / 2.0, -SoccerPitch2D.GOAL_LENGTH / 2.0, -SoccerPitch2D.GOAL_HEIGHT],
            dtype="float",
        )
        self.right_goal_bottom_right_post = np.array(
            [pitch_length / 2.0, SoccerPitch2D.GOAL_LENGTH / 2.0, 0.0], dtype="float"
        )
        self.right_goal_top_right_post = np.array(
            [pitch_length / 2.0, SoccerPitch2D.GOAL_LENGTH / 2.0, -SoccerPitch2D.GOAL_HEIGHT],
            dtype="float",
        )

        self.left_penalty_mark = np.array(
            [-pitch_length / 2.0 + SoccerPitch2D.GOAL_LINE_TO_PENALTY_MARK, 0, 0], dtype="float"
        )
        self.right_penalty_mark = np.array(
            [pitch_length / 2.0 - SoccerPitch2D.GOAL_LINE_TO_PENALTY_MARK, 0, 0], dtype="float"
        )

        self.left_penalty_area_top_right_corner = np.array(
            [
                -pitch_length / 2.0 + SoccerPitch2D.PENALTY_AREA_LENGTH,
                -SoccerPitch2D.PENALTY_AREA_WIDTH / 2.0,
                0,
            ],
            dtype="float",
        )
        self.left_penalty_area_top_left_corner = np.array(
            [-pitch_length / 2.0, -SoccerPitch2D.PENALTY_AREA_WIDTH / 2.0, 0], dtype="float"
        )
        self.left_penalty_area_bottom_right_corner = np.array(
            [
                -pitch_length / 2.0 + SoccerPitch2D.PENALTY_AREA_LENGTH,
                SoccerPitch2D.PENALTY_AREA_WIDTH / 2.0,
                0,
            ],
            dtype="float",
        )
        self.left_penalty_area_bottom_left_corner = np.array(
            [-pitch_length / 2.0, SoccerPitch2D.PENALTY_AREA_WIDTH / 2.0, 0], dtype="float"
        )
        self.right_penalty_area_top_right_corner = np.array(
            [pitch_length / 2.0, -SoccerPitch2D.PENALTY_AREA_WIDTH / 2.0, 0], dtype="float"
        )
        self.right_penalty_area_top_left_corner = np.array(
            [
                pitch_length / 2.0 - SoccerPitch2D.PENALTY_AREA_LENGTH,
                -SoccerPitch2D.PENALTY_AREA_WIDTH / 2.0,
                0,
            ],
            dtype="float",
        )
        self.right_penalty_area_bottom_right_corner = np.array(
            [pitch_length / 2.0, SoccerPitch2D.PENALTY_AREA_WIDTH / 2.0, 0], dtype="float"
        )
        self.right_penalty_area_bottom_left_corner = np.array(
            [
                pitch_length / 2.0 - SoccerPitch2D.PENALTY_AREA_LENGTH,
                SoccerPitch2D.PENALTY_AREA_WIDTH / 2.0,
                0,
            ],
            dtype="float",
        )

        self.left_goal_area_top_right_corner = np.array(
            [
                -pitch_length / 2.0 + SoccerPitch2D.GOAL_AREA_LENGTH,
                -SoccerPitch2D.GOAL_AREA_WIDTH / 2.0,
                0,
            ],
            dtype="float",
        )
        self.left_goal_area_top_left_corner = np.array(
            [-pitch_length / 2.0, -SoccerPitch2D.GOAL_AREA_WIDTH / 2.0, 0], dtype="float"
        )
        self.left_goal_area_bottom_right_corner = np.array(
            [
                -pitch_length / 2.0 + SoccerPitch2D.GOAL_AREA_LENGTH,
                SoccerPitch2D.GOAL_AREA_WIDTH / 2.0,
                0,
            ],
            dtype="float",
        )
        self.left_goal_area_bottom_left_corner = np.array(
            [-pitch_length / 2.0, SoccerPitch2D.GOAL_AREA_WIDTH / 2.0, 0], dtype="float"
        )
        self.right_goal_area_top_right_corner = np.array(
            [pitch_length / 2.0, -SoccerPitch2D.GOAL_AREA_WIDTH / 2.0, 0], dtype="float"
        )
        self.right_goal_area_top_left_corner = np.array(
            [
                pitch_length / 2.0 - SoccerPitch2D.GOAL_AREA_LENGTH,
                -SoccerPitch2D.GOAL_AREA_WIDTH / 2.0,
                0,
            ],
            dtype="float",
        )
        self.right_goal_area_bottom_right_corner = np.array(
            [pitch_length / 2.0, SoccerPitch2D.GOAL_AREA_WIDTH / 2.0, 0], dtype="float"
        )
        self.right_goal_area_bottom_left_corner = np.array(
            [
                pitch_length / 2.0 - SoccerPitch2D.GOAL_AREA_LENGTH,
                SoccerPitch2D.GOAL_AREA_WIDTH / 2.0,
                0,
            ],
            dtype="float",
        )

        x = -pitch_length / 2.0 + SoccerPitch2D.PENALTY_AREA_LENGTH
        dx = SoccerPitch2D.PENALTY_AREA_LENGTH - SoccerPitch2D.GOAL_LINE_TO_PENALTY_MARK
        y = -np.sqrt(
            SoccerPitch2D.CENTER_CIRCLE_RADIUS * SoccerPitch2D.CENTER_CIRCLE_RADIUS - dx * dx
        )
        self.top_left_16M_penalty_arc_mark = np.array([x, y, 0], dtype="float")

        x = pitch_length / 2.0 - SoccerPitch2D.PENALTY_AREA_LENGTH
        dx = SoccerPitch2D.PENALTY_AREA_LENGTH - SoccerPitch2D.GOAL_LINE_TO_PENALTY_MARK
        y = -np.sqrt(
            SoccerPitch2D.CENTER_CIRCLE_RADIUS * SoccerPitch2D.CENTER_CIRCLE_RADIUS - dx * dx
        )
        self.top_right_16M_penalty_arc_mark = np.array([x, y, 0], dtype="float")

        x = -pitch_length / 2.0 + SoccerPitch2D.PENALTY_AREA_LENGTH
        dx = SoccerPitch2D.PENALTY_AREA_LENGTH - SoccerPitch2D.GOAL_LINE_TO_PENALTY_MARK
        y = np.sqrt(
            SoccerPitch2D.CENTER_CIRCLE_RADIUS * SoccerPitch2D.CENTER_CIRCLE_RADIUS - dx * dx
        )
        self.bottom_left_16M_penalty_arc_mark = np.array([x, y, 0], dtype="float")

        x = pitch_length / 2.0 - SoccerPitch2D.PENALTY_AREA_LENGTH
        dx = SoccerPitch2D.PENALTY_AREA_LENGTH - SoccerPitch2D.GOAL_LINE_TO_PENALTY_MARK
        y = np.sqrt(
            SoccerPitch2D.CENTER_CIRCLE_RADIUS * SoccerPitch2D.CENTER_CIRCLE_RADIUS - dx * dx
        )
        self.bottom_right_16M_penalty_arc_mark = np.array([x, y, 0], dtype="float")

        # self.set_elevations(elevation)

        self.point_dict = {}
        self.point_dict["CENTER_MARK"] = self.center_mark
        self.point_dict["L_PENALTY_MARK"] = self.left_penalty_mark
        self.point_dict["R_PENALTY_MARK"] = self.right_penalty_mark
        self.point_dict["TL_PITCH_CORNER"] = self.top_left_corner
        self.point_dict["BL_PITCH_CORNER"] = self.bottom_left_corner
        self.point_dict["TR_PITCH_CORNER"] = self.top_right_corner
        self.point_dict["BR_PITCH_CORNER"] = self.bottom_right_corner
        self.point_dict["L_PENALTY_AREA_TL_CORNER"] = self.left_penalty_area_top_left_corner
        self.point_dict["L_PENALTY_AREA_TR_CORNER"] = self.left_penalty_area_top_right_corner
        self.point_dict["L_PENALTY_AREA_BL_CORNER"] = self.left_penalty_area_bottom_left_corner
        self.point_dict["L_PENALTY_AREA_BR_CORNER"] = self.left_penalty_area_bottom_right_corner

        self.point_dict["R_PENALTY_AREA_TL_CORNER"] = self.right_penalty_area_top_left_corner
        self.point_dict["R_PENALTY_AREA_TR_CORNER"] = self.right_penalty_area_top_right_corner
        self.point_dict["R_PENALTY_AREA_BL_CORNER"] = self.right_penalty_area_bottom_left_corner
        self.point_dict["R_PENALTY_AREA_BR_CORNER"] = self.right_penalty_area_bottom_right_corner

        self.point_dict["L_GOAL_AREA_TL_CORNER"] = self.left_goal_area_top_left_corner
        self.point_dict["L_GOAL_AREA_TR_CORNER"] = self.left_goal_area_top_right_corner
        self.point_dict["L_GOAL_AREA_BL_CORNER"] = self.left_goal_area_bottom_left_corner
        self.point_dict["L_GOAL_AREA_BR_CORNER"] = self.left_goal_area_bottom_right_corner

        self.point_dict["R_GOAL_AREA_TL_CORNER"] = self.right_goal_area_top_left_corner
        self.point_dict["R_GOAL_AREA_TR_CORNER"] = self.right_goal_area_top_right_corner
        self.point_dict["R_GOAL_AREA_BL_CORNER"] = self.right_goal_area_bottom_left_corner
        self.point_dict["R_GOAL_AREA_BR_CORNER"] = self.right_goal_area_bottom_right_corner

        # self.point_dict["L_GOAL_TL_POST"] = self.left_goal_top_left_post
        # self.point_dict["L_GOAL_TR_POST"] = self.left_goal_top_right_post
        # self.point_dict["L_GOAL_BL_POST"] = self.left_goal_bottom_left_post
        # self.point_dict["L_GOAL_BR_POST"] = self.left_goal_bottom_right_post

        # self.point_dict["R_GOAL_TL_POST"] = self.right_goal_top_left_post
        # self.point_dict["R_GOAL_TR_POST"] = self.right_goal_top_right_post
        # self.point_dict["R_GOAL_BL_POST"] = self.right_goal_bottom_left_post
        # self.point_dict["R_GOAL_BR_POST"] = self.right_goal_bottom_right_post

        self.point_dict[
            "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION"
        ] = self.halfway_and_top_touch_line_mark
        self.point_dict[
            "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION"
        ] = self.halfway_and_bottom_touch_line_mark
        self.point_dict[
            "T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION"
        ] = self.halfway_line_and_center_circle_top_mark
        self.point_dict[
            "B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION"
        ] = self.halfway_line_and_center_circle_bottom_mark
        self.point_dict[
            "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"
        ] = self.top_left_16M_penalty_arc_mark
        self.point_dict[
            "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"
        ] = self.bottom_left_16M_penalty_arc_mark
        self.point_dict[
            "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"
        ] = self.top_right_16M_penalty_arc_mark
        self.point_dict[
            "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"
        ] = self.bottom_right_16M_penalty_arc_mark

        self.line_extremities = dict()
        self.line_extremities["Big rect. left bottom"] = (
            self.point_dict["L_PENALTY_AREA_BL_CORNER"],
            self.point_dict["L_PENALTY_AREA_BR_CORNER"],
        )
        self.line_extremities["Big rect. left top"] = (
            self.point_dict["L_PENALTY_AREA_TL_CORNER"],
            self.point_dict["L_PENALTY_AREA_TR_CORNER"],
        )
        self.line_extremities["Big rect. left main"] = (
            self.point_dict["L_PENALTY_AREA_TR_CORNER"],
            self.point_dict["L_PENALTY_AREA_BR_CORNER"],
        )
        self.line_extremities["Big rect. right bottom"] = (
            self.point_dict["R_PENALTY_AREA_BL_CORNER"],
            self.point_dict["R_PENALTY_AREA_BR_CORNER"],
        )
        self.line_extremities["Big rect. right top"] = (
            self.point_dict["R_PENALTY_AREA_TL_CORNER"],
            self.point_dict["R_PENALTY_AREA_TR_CORNER"],
        )
        self.line_extremities["Big rect. right main"] = (
            self.point_dict["R_PENALTY_AREA_TL_CORNER"],
            self.point_dict["R_PENALTY_AREA_BL_CORNER"],
        )

        self.line_extremities["Small rect. left bottom"] = (
            self.point_dict["L_GOAL_AREA_BL_CORNER"],
            self.point_dict["L_GOAL_AREA_BR_CORNER"],
        )
        self.line_extremities["Small rect. left top"] = (
            self.point_dict["L_GOAL_AREA_TL_CORNER"],
            self.point_dict["L_GOAL_AREA_TR_CORNER"],
        )
        self.line_extremities["Small rect. left main"] = (
            self.point_dict["L_GOAL_AREA_TR_CORNER"],
            self.point_dict["L_GOAL_AREA_BR_CORNER"],
        )
        self.line_extremities["Small rect. right bottom"] = (
            self.point_dict["R_GOAL_AREA_BL_CORNER"],
            self.point_dict["R_GOAL_AREA_BR_CORNER"],
        )
        self.line_extremities["Small rect. right top"] = (
            self.point_dict["R_GOAL_AREA_TL_CORNER"],
            self.point_dict["R_GOAL_AREA_TR_CORNER"],
        )
        self.line_extremities["Small rect. right main"] = (
            self.point_dict["R_GOAL_AREA_TL_CORNER"],
            self.point_dict["R_GOAL_AREA_BL_CORNER"],
        )

        self.line_extremities["Side line top"] = (
            self.point_dict["TL_PITCH_CORNER"],
            self.point_dict["TR_PITCH_CORNER"],
        )
        self.line_extremities["Side line bottom"] = (
            self.point_dict["BL_PITCH_CORNER"],
            self.point_dict["BR_PITCH_CORNER"],
        )
        self.line_extremities["Side line left"] = (
            self.point_dict["TL_PITCH_CORNER"],
            self.point_dict["BL_PITCH_CORNER"],
        )
        self.line_extremities["Side line right"] = (
            self.point_dict["TR_PITCH_CORNER"],
            self.point_dict["BR_PITCH_CORNER"],
        )
        self.line_extremities["Middle line"] = (
            self.point_dict["T_TOUCH_AND_HALFWAY_LINES_INTERSECTION"],
            self.point_dict["B_TOUCH_AND_HALFWAY_LINES_INTERSECTION"],
        )

        self.line_extremities["Circle right"] = (
            self.point_dict["TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"],
            self.point_dict["BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"],
        )
        self.line_extremities["Circle left"] = (
            self.point_dict["TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"],
            self.point_dict["BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"],
        )

        self.line_extremities_keys = dict()
        self.line_extremities_keys["Big rect. left bottom"] = (
            "L_PENALTY_AREA_BL_CORNER",
            "L_PENALTY_AREA_BR_CORNER",
        )
        self.line_extremities_keys["Big rect. left top"] = (
            "L_PENALTY_AREA_TL_CORNER",
            "L_PENALTY_AREA_TR_CORNER",
        )
        self.line_extremities_keys["Big rect. left main"] = (
            "L_PENALTY_AREA_TR_CORNER",
            "L_PENALTY_AREA_BR_CORNER",
        )
        self.line_extremities_keys["Big rect. right bottom"] = (
            "R_PENALTY_AREA_BL_CORNER",
            "R_PENALTY_AREA_BR_CORNER",
        )
        self.line_extremities_keys["Big rect. right top"] = (
            "R_PENALTY_AREA_TL_CORNER",
            "R_PENALTY_AREA_TR_CORNER",
        )
        self.line_extremities_keys["Big rect. right main"] = (
            "R_PENALTY_AREA_TL_CORNER",
            "R_PENALTY_AREA_BL_CORNER",
        )

        self.line_extremities_keys["Small rect. left bottom"] = (
            "L_GOAL_AREA_BL_CORNER",
            "L_GOAL_AREA_BR_CORNER",
        )
        self.line_extremities_keys["Small rect. left top"] = (
            "L_GOAL_AREA_TL_CORNER",
            "L_GOAL_AREA_TR_CORNER",
        )
        self.line_extremities_keys["Small rect. left main"] = (
            "L_GOAL_AREA_TR_CORNER",
            "L_GOAL_AREA_BR_CORNER",
        )
        self.line_extremities_keys["Small rect. right bottom"] = (
            "R_GOAL_AREA_BL_CORNER",
            "R_GOAL_AREA_BR_CORNER",
        )
        self.line_extremities_keys["Small rect. right top"] = (
            "R_GOAL_AREA_TL_CORNER",
            "R_GOAL_AREA_TR_CORNER",
        )
        self.line_extremities_keys["Small rect. right main"] = (
            "R_GOAL_AREA_TL_CORNER",
            "R_GOAL_AREA_BL_CORNER",
        )

        self.line_extremities_keys["Side line top"] = ("TL_PITCH_CORNER", "TR_PITCH_CORNER")
        self.line_extremities_keys["Side line bottom"] = ("BL_PITCH_CORNER", "BR_PITCH_CORNER")
        self.line_extremities_keys["Side line left"] = ("TL_PITCH_CORNER", "BL_PITCH_CORNER")
        self.line_extremities_keys["Side line right"] = ("TR_PITCH_CORNER", "BR_PITCH_CORNER")
        self.line_extremities_keys["Middle line"] = (
            "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
            "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
        )

        self.line_extremities_keys["Circle right"] = (
            "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
            "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
        )
        self.line_extremities_keys["Circle left"] = (
            "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
            "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
        )

        @staticmethod
        def get_segment_keys():
            return set(
                [
                    "Big rect. left bottom",
                    "Big rect. left top",
                    "Big rect. left main",
                    "Big rect. right bottom",
                    "Big rect. right top",
                    "Big rect. right main",
                    "Small rect. left bottom" "Small rect. left top",
                    "Small rect. left main",
                    "Small rect. right bottom",
                    "Small rect. right top",
                    "Small rect. right main",
                    "Side line top",
                    "Side line bottom",
                    "Side line left",
                    "Side line right",
                    "Middle line",
                    "Circle right",
                    "Circle left",
                ]
            )

    def points(self):
        return [
            self.center_mark,
            self.halfway_and_bottom_touch_line_mark,
            self.halfway_and_top_touch_line_mark,
            self.halfway_line_and_center_circle_top_mark,
            self.halfway_line_and_center_circle_bottom_mark,
            self.bottom_right_corner,
            self.bottom_left_corner,
            self.top_right_corner,
            self.top_left_corner,
            self.left_penalty_mark,
            self.right_penalty_mark,
            self.left_penalty_area_top_right_corner,
            self.left_penalty_area_top_left_corner,
            self.left_penalty_area_bottom_right_corner,
            self.left_penalty_area_bottom_left_corner,
            self.right_penalty_area_top_right_corner,
            self.right_penalty_area_top_left_corner,
            self.right_penalty_area_bottom_right_corner,
            self.right_penalty_area_bottom_left_corner,
            self.left_goal_area_top_right_corner,
            self.left_goal_area_top_left_corner,
            self.left_goal_area_bottom_right_corner,
            self.left_goal_area_bottom_left_corner,
            self.right_goal_area_top_right_corner,
            self.right_goal_area_top_left_corner,
            self.right_goal_area_bottom_right_corner,
            self.right_goal_area_bottom_left_corner,
            self.top_left_16M_penalty_arc_mark,
            self.top_right_16M_penalty_arc_mark,
            self.bottom_left_16M_penalty_arc_mark,
            self.bottom_right_16M_penalty_arc_mark,
            # self.left_goal_top_left_post,
            # self.left_goal_top_right_post,
            # self.left_goal_bottom_left_post,
            # self.left_goal_bottom_right_post,
            # self.right_goal_top_left_post,
            # self.right_goal_top_right_post,
            # self.right_goal_bottom_left_post,
            # self.right_goal_bottom_right_post,
        ]

    def sample_field_points(self, dist=0.1, dist_circles=0.2):
        """
        Samples each pitch element every dist meters, returns a dictionary associating the class of the element with a list of points sampled along this element.
        :param dist: the distance in meters between each point sampled
        :param dist_circles: the distance in meters between each point sampled on circles
        :return:  a dictionary associating the class of the element with a list of points sampled along this element.
        """
        polylines = dict()
        center = self.point_dict["CENTER_MARK"]
        fromAngle = 0.0
        toAngle = 2 * np.pi

        if toAngle < fromAngle:
            toAngle += 2 * np.pi
        x1 = center[0] + np.cos(fromAngle) * SoccerPitch2D.CENTER_CIRCLE_RADIUS
        y1 = center[1] + np.sin(fromAngle) * SoccerPitch2D.CENTER_CIRCLE_RADIUS
        z1 = 0.0
        point = np.array((x1, y1, z1))
        polyline = [point]
        length = SoccerPitch2D.CENTER_CIRCLE_RADIUS * (toAngle - fromAngle)
        nb_pts = int(length / dist_circles)
        dangle = dist_circles / SoccerPitch2D.CENTER_CIRCLE_RADIUS
        for i in range(1, nb_pts):
            angle = fromAngle + i * dangle
            x = center[0] + np.cos(angle) * SoccerPitch2D.CENTER_CIRCLE_RADIUS
            y = center[1] + np.sin(angle) * SoccerPitch2D.CENTER_CIRCLE_RADIUS
            z = 0
            point = np.array((x, y, z))
            polyline.append(point)
        polylines["Circle central"] = polyline
        for key, line in self.line_extremities.items():

            if "Circle" in key:
                if key == "Circle right":
                    top = self.point_dict["TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
                    bottom = self.point_dict["BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
                    center = self.point_dict["R_PENALTY_MARK"]
                    toAngle = np.arctan2(top[1] - center[1], top[0] - center[0]) + 2 * np.pi
                    fromAngle = np.arctan2(bottom[1] - center[1], bottom[0] - center[0]) + 2 * np.pi
                elif key == "Circle left":
                    top = self.point_dict["TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
                    bottom = self.point_dict["BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION"]
                    center = self.point_dict["L_PENALTY_MARK"]
                    fromAngle = np.arctan2(top[1] - center[1], top[0] - center[0]) + 2 * np.pi
                    toAngle = np.arctan2(bottom[1] - center[1], bottom[0] - center[0]) + 2 * np.pi
                if toAngle < fromAngle:
                    toAngle += 2 * np.pi
                x1 = center[0] + np.cos(fromAngle) * SoccerPitch2D.CENTER_CIRCLE_RADIUS
                y1 = center[1] + np.sin(fromAngle) * SoccerPitch2D.CENTER_CIRCLE_RADIUS
                z1 = 0.0
                xn = center[0] + np.cos(toAngle) * SoccerPitch2D.CENTER_CIRCLE_RADIUS
                yn = center[1] + np.sin(toAngle) * SoccerPitch2D.CENTER_CIRCLE_RADIUS
                zn = 0.0
                start = np.array((x1, y1, z1))
                end = np.array((xn, yn, zn))
                polyline = [start]
                length = SoccerPitch2D.CENTER_CIRCLE_RADIUS * (toAngle - fromAngle)
                nb_pts = int(length / dist_circles)
                dangle = dist_circles / SoccerPitch2D.CENTER_CIRCLE_RADIUS
                for i in range(1, nb_pts + 1):
                    angle = fromAngle + i * dangle
                    x = center[0] + np.cos(angle) * SoccerPitch2D.CENTER_CIRCLE_RADIUS
                    y = center[1] + np.sin(angle) * SoccerPitch2D.CENTER_CIRCLE_RADIUS
                    z = 0
                    point = np.array((x, y, z))
                    polyline.append(point)
                polyline.append(end)
                polylines[key] = polyline
            else:
                start = line[0]
                end = line[1]

                polyline = [start]

                total_dist = np.sqrt(np.sum(np.square(start - end)))
                nb_pts = int(total_dist / dist - 1)

                v = end - start
                v /= np.linalg.norm(v)
                prev_pt = start
                for i in range(nb_pts):
                    pt = prev_pt + dist * v
                    prev_pt = pt
                    polyline.append(pt)
                polyline.append(end)
                polylines[key] = polyline
        return polylines

    def get_2d_homogeneous_line(self, line_name):
        """
        For lines belonging to the pitch lawn plane returns its 2D homogenous equation coefficients
        :param line_name
        :return: an array containing the three coefficients of the line
        """
        # ensure line in football pitch plane
        if (
            line_name in self.line_extremities.keys()
            and "post" not in line_name
            and "crossbar" not in line_name
            and "Circle" not in line_name
        ):
            extremities = self.line_extremities[line_name]
            p1 = np.array([extremities[0][0], extremities[0][1], 1], dtype="float")
            p2 = np.array([extremities[1][0], extremities[1][1], 1], dtype="float")
            line = np.cross(p1, p2)

            return line
        return None


def get_polylines(
    prediction,
    width,
    height,
    sampling_factor,
    project_from: str,
    evaluate_planar: bool,
    distort=False,
    zeta=100,  # in pixels
    wrap_delta=0.1,  # relative to image dim
):
    """Modified version from SoccerNet.Evaluation.utils_calibration.get_polylines() to use the custom SoccerPitch2D that excludes all segments z != 0.0

    Given a set of camera parameters, this function adapts the camera to the desired image resolution and then
    projects the 3D points belonging to the terrain model in order to give a dictionary associating the classes
    observed and the points projected in the image.

    :param prediction: camera parameters in their json/dictionary format or homography matrix
    :param width: image width for evaluation
    :param height: image height for evaluation
    :return: a dictionary with keys corresponding to a class observed in the image ( a line of the 3D model whose
    projection falls in the image) and values are then the list of 2D projected points.
    """

    cam = None
    if project_from == "HDecomp":
        cam = _get_cam_from_decomp(
            np.array(prediction["homography"]), width, height, wrap_delta, zeta
        )
        if cam is None:
            return None
    elif project_from == "Camera":
        cam = Camera(width, height)
        cam.from_json_parameters(prediction)
    elif project_from == "Homography":
        h = np.array(prediction["homography"])
        cam = HomographySN(h)
    else:
        raise KeyError()

    assert cam is not None
    if evaluate_planar:
        field = SoccerPitch2D()
    else:
        field = SoccerPitch()  # 3D Soccer Pitch

    # lazy implementation to project all 3d points to the visible image
    projections = dict()
    sides = [
        np.array([1, 0, 0]),
        np.array([1, 0, -width + 1]),
        np.array([0, 1, 0]),
        np.array([0, 1, -height + 1]),
    ]

    for key, points in field.sample_field_points(sampling_factor).items():

        projections_list = []
        in_img = False
        prev_proj = np.zeros(3)
        for i, point in enumerate(points):
            ext = cam.project_point(point, distort=distort)
            if ext[2] < 1e-5:
                # point at infinity or behind camera
                continue
            if 0 <= ext[0] < width and 0 <= ext[1] < height:

                if not in_img and i > 0:

                    line = np.cross(ext, prev_proj)
                    in_img_intersections = []
                    dist_to_ext = []
                    for side in sides:
                        intersection = np.cross(line, side)
                        intersection /= intersection[2]
                        if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
                            in_img_intersections.append(intersection)
                            dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
                    if in_img_intersections:
                        intersection = in_img_intersections[np.argmin(dist_to_ext)]

                        projections_list.append({"x": intersection[0], "y": intersection[1]})

                projections_list.append({"x": ext[0], "y": ext[1]})
                in_img = True
            elif in_img:
                # first point out
                line = np.cross(ext, prev_proj)

                in_img_intersections = []
                dist_to_ext = []
                for side in sides:
                    intersection = np.cross(line, side)
                    intersection /= intersection[2]
                    if 0 <= intersection[0] < width and 0 <= intersection[1] < height:
                        in_img_intersections.append(intersection)
                        dist_to_ext.append(np.sqrt(np.sum(np.square(intersection - ext))))
                if in_img_intersections:
                    intersection = in_img_intersections[np.argmin(dist_to_ext)]

                    projections_list.append({"x": intersection[0], "y": intersection[1]})
                in_img = False
            prev_proj = ext
        if len(projections_list):
            projections[key] = projections_list

    return projections
