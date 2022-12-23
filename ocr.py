# Import required packages
import configparser
from datetime import datetime

import cv2
import pytesseract
import numpy as np
import const

from util import rotate, write_json

zero_to_nine = "0123456789"
a_to_z = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = "D:\Program Files\Tesseract-OCR\\tesseract.exe"

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

config = configparser.ConfigParser()
config.read("config.ini")

coordinates = const.HD1920by1080  # Defualt constants to use


def crop_top_bottom(img):
    """crop_top_bottom"""
    return img[2:-2, 0:-0]


def str_to_number(str):
    """str_to_number"""
    str = str.replace(",", "").replace("\n", "")
    if len(str) <= 0:
        return 0
    return int(str)


def debug_json(name, json):
    """debug_json"""
    if config.getboolean("debug", "json"):
        write_json(name, json)


def debug_image(name, im):
    """debug_image"""
    if config.getboolean("debug", "images"):
        cv2.imwrite(name, im)


def debug(*values):
    """debug"""
    if config.getboolean("debug", "log"):
        print(*values)


def process_self_name(im):
    """process_self_name"""
    rotated = rotate(im, -4)  # rotate to make the text straight
    # rotated = im
    debug_image(f"dbg/name_rotated.jpg", rotated)
    cropped = rotated[16:38, 5:240]
    debug_image(f"dbg/name_cropped.jpg", cropped)
    name = ocr(cropped, 0, "--psm 7 -c load_system_dawg=0").replace("\n", "")
    debug("name", name)

    if len(name) <= 0:
        raise Exception("empty name")

    return {"name": name}


def process_self_hero(im):
    """process_self_hero"""
    debug_image(f"dbg/hero.jpg", im)
    name = ocr(
        im,
        0,
        f"--psm 7 -c load_system_dawg=0 tessedit_char_whitelist={zero_to_nine}",
        inv=True,
    ).replace("\n", "")
    debug("hero", name)

    if len(name) <= 0:
        raise Exception("empty hero")

    return {"hero": name}


def process_match_info(im):
    """process_match_info"""
    mode_map_img = im[0 : coordinates["mode_height"], coordinates["mode_offset"] :]
    debug_image(f"dbg/mode.jpg", mode_map_img)
    mode_map = ocr(
        mode_map_img, 0, f"--psm 7  -c tessedit_char_whitelist=|:-{a_to_z}", crop=False
    ).replace("\n", "")
    debug("mode+map", mode_map)

    mode, map, *rest = mode_map.split("|")

    if len(mode) <= 0:
        raise Exception("empty mode")

    if len(map) <= 0:
        raise Exception("empty map")

    is_comp = "COMPETITIVE" in mode
    if "-COMPETITIVE" in mode:
        mode = mode[: -len("-COMPETITIVE")]
    if "COMPETITIVE" in mode:
        mode = mode[: -len("COMPETITIVE")]

    time_img = im[
        coordinates["time_offset_y"] : coordinates["time_offset_y"]
        + coordinates["time_height"],
        coordinates["time_offset_x"] : coordinates["time_offset_y"]
        + coordinates["time_width"],
    ]
    debug_image(f"dbg/time.jpg", time_img)
    time = ocr(
        time_img, 0, f"--psm 7  -c tessedit_char_whitelist={zero_to_nine}:", crop=False
    ).replace("\n", "")
    debug("time", time)

    datetime.strptime(time, "%M:%S")

    return {
        "mode_map": mode_map,
        "mode": mode,
        "map": map,
        "competitive": is_comp,
        "time": time,
    }


# https://stackoverflow.com/questions/33497736/opencv-adjusting-photo-with-skew-angle-tilt
# https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
def deskew_player_name(name_img):
    """deskew_player_name"""
    rows, cols = name_img.shape
    pts1 = np.float32([[5, 0], [95, 0], [0, 25], [90, 25]])
    pts2 = np.float32([[-5, 0], [95, 0], [-3, 25], [97, 25]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(name_img, M, (cols, rows))

    return dst


def process_player_list(im, name_offs):
    """process_player_list"""
    out = []
    for i in range(0, 5):
        row_y = coordinates["row_height"] * i

        name_img = im[
            row_y
            + coordinates["row_padding"] : row_y
            + coordinates["row_height"]
            - coordinates["row_padding"],
            name_offs : name_offs + coordinates["name_width"],
        ]
        # name_img = crop_top_bottom(name_img)
        debug_image(f"dbg/name{i}.jpg", name_img)
        name_img = deskew_player_name(name_img)
        debug_image(f"dbg/name{i}_deskew.jpg", name_img)
        name = ocr(name_img, i, "--psm 7 -c load_system_dawg=0").replace("\n", "")
        debug("name", name)

        elims_img = im[
            row_y
            + coordinates["row_padding"] : row_y
            + coordinates["row_height"]
            - coordinates["row_padding"],
            coordinates["elims_offset"] : coordinates["elims_offset"]
            + coordinates["elims_width"],
        ]
        debug_image(f"dbg/elims{i}.jpg", elims_img)
        elims = str_to_number(
            ocr(elims_img, i, f"--psm 8 -c tessedit_char_whitelist={zero_to_nine}")
        )
        debug("elims", elims)

        assist_img = im[
            row_y
            + coordinates["row_padding"] : row_y
            + coordinates["row_height"]
            - coordinates["row_padding"],
            coordinates["assist_offset"] : coordinates["assist_offset"]
            + coordinates["assist_width"],
        ]
        debug_image(f"dbg/assist{i}.jpg", assist_img)
        assists = str_to_number(
            ocr(assist_img, i, f"--psm 8 -c tessedit_char_whitelist={zero_to_nine}")
        )
        debug("assists", assists)

        deaths_img = im[
            row_y
            + coordinates["row_padding"] : row_y
            + coordinates["row_height"]
            - coordinates["row_padding"],
            coordinates["deaths_offset"] : coordinates["deaths_offset"]
            + coordinates["deaths_width"],
        ]
        debug_image(f"dbg/deaths{i}.jpg", deaths_img)
        deaths = str_to_number(
            ocr(deaths_img, i, f"--psm 8 -c tessedit_char_whitelist={zero_to_nine}")
        )
        debug("deaths", deaths)

        dmg_img = im[
            row_y
            + coordinates["row_padding"] : row_y
            + coordinates["row_height"]
            - coordinates["row_padding"],
            coordinates["dmg_offset"] : coordinates["dmg_offset"]
            + coordinates["dmg_width"],
        ]
        debug_image(f"dbg/dmg{i}.jpg", dmg_img)
        dmg = str_to_number(
            ocr(dmg_img, i, f"--psm 8 -c tessedit_char_whitelist={zero_to_nine},")
        )
        debug("dmg", dmg)

        heal_img = im[
            row_y
            + coordinates["row_padding"] : row_y
            + coordinates["row_height"]
            - coordinates["row_padding"],
            coordinates["heals_offset"] : coordinates["heals_offset"]
            + coordinates["heals_width"],
        ]
        debug_image(f"dbg/heal{i}.jpg", heal_img)
        heal = str_to_number(
            ocr(heal_img, i, f"--psm 8 -c tessedit_char_whitelist={zero_to_nine},")
        )
        debug("heal", heal)

        mit_img = im[
            row_y
            + coordinates["row_padding"] : row_y
            + coordinates["row_height"]
            - coordinates["row_padding"],
            coordinates["mit_offset"] : coordinates["mit_offset"]
            + coordinates["mit_width"],
        ]
        debug_image(f"dbg/mit{i}.jpg", mit_img)
        mit = str_to_number(
            ocr(mit_img, i, f"--psm 8 -c tessedit_char_whitelist={zero_to_nine},")
        )
        debug("mit", mit)

        out.append(
            {
                "name": name,
                "elims": elims,
                "assists": assists,
                "deaths": deaths,
                "dmg": dmg,
                "heal": heal,
                "mit": mit,
            }
        )
    return out


def ocr(img, offs, args, crop=True, inv=False):
    """ocr"""
    # Performing OTSU threshold
    thr = cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV if inv else cv2.THRESH_OTSU
    ret, thresh1 = cv2.threshold(img, 0, 255, thr)
    debug_image(f"dbg/thresh{offs}.jpg", thresh1)

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    # # Drawing a rectangle on copied image
    im2 = img.copy()
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    debug_image(f"dbg/contour{offs}.jpg", im2)

    if crop:
        # Cropping the text block for giving input to OCR
        cropped = img[y + 2 : y + h - 2, x + 2 : x + w - 2]
    else:
        cropped = img

    debug_image(f"dbg/cropped{offs}.jpg", cropped)

    text = pytesseract.image_to_string(cropped, config=args)
    return text


def process_screenshot(img):
    """process_screenshot"""
    # Read image from which text needs to be extracted
    # img = cv2.imread("7b3da7fc15_Overwatch.png")

    # Preprocessing the image starts

    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = gray.shape
    width = int(shape[1])
    height = int(shape[0])
    print(width)
    print(height)

    global coordinates
    if width == 1920 and height == 1080:
        coordinates = const.HD1920by1080
    elif width == 3440 and height == 1440:
        coordinates = const.wide3440by1440

    print("Test")
    print(coordinates["self_hero_start"][0])

    debug_image("gray.jpg", gray)

    self_hero = gray[
        coordinates["self_hero_start"][1] : coordinates["self_hero_end"][1],
        coordinates["self_hero_start"][0] : coordinates["self_hero_end"][0],
    ]
    debug_image("dbg/hero.jpg", self_hero)
    self_hero_info = process_self_hero(self_hero)
    debug(self_hero_info)
    debug_json("hero.json", self_hero_info)

    self_name = gray[
        coordinates["self_name_start"][1] : coordinates["self_name_end"][1],
        coordinates["self_name_start"][0] : coordinates["self_name_end"][0],
    ]
    # print(self_name)
    # self_name = "Test"
    debug_image("dbg/name.jpg", self_name)
    self_name_info = process_self_name(self_name)
    debug(self_name_info)
    debug_json("name.json", self_name_info)

    match = gray[
        coordinates["match_info_start"][1] : coordinates["match_info_end"][1],
        coordinates["match_info_start"][0] : coordinates["match_info_end"][0],
    ]
    debug_image("dbg/match.jpg", match)
    match_info = process_match_info(match)
    debug(match_info)
    debug_json("match.json", match_info)

    allies = gray[
        coordinates["allies_start"][1] : coordinates["allies_end"][1],
        coordinates["allies_start"][0] : coordinates["allies_end"][0],
    ]
    debug_image("dbg/allies.jpg", allies)
    allies_info = process_player_list(allies, coordinates["name_offset"])
    debug(allies_info)
    debug_json("allies.json", allies_info)

    enemies = gray[
        coordinates["enemies_start"][1] : coordinates["enemies_end"][1],
        coordinates["enemies_start"][0] : coordinates["enemies_end"][0],
    ]
    debug_image("dbg/enemies.jpg", enemies)
    enemies_info = process_player_list(enemies, coordinates["name_offset"])
    debug(enemies_info)
    debug_json("enemies.json", enemies_info)

    return {
        "time": datetime.now().timestamp(),
        "state": "in_progress",
        "match": match_info,
        "self": self_hero_info | self_name_info,
        "players": {"allies": allies_info, "enemies": enemies_info},
    }


def process_screenshot_file(filename):
    """process_screenshot_file"""
    img = cv2.imread(filename)
    return process_screenshot(img)
