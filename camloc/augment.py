"""Synthetic appearance changes for robustness tests (applied to the grayscale image before features).

These are controlled stress tests, not a realistic night-to-day translation: real daylight also changes
shading, shadows and which structures are visible.
"""
import cv2
import numpy as np


def _gamma(g, gamma):
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(g, lut)


def bright(g):
    """Strong overall brightening (gamma 0.4), like a daytime exposure of the same scene."""
    return _gamma(g, 0.4)


def overexpose(g):
    """Exposure x3 with clipping: highlights saturate."""
    return cv2.convertScaleAbs(g, alpha=3.0, beta=0)


def lowcontrast(g):
    """Half contrast plus a grey veil (haze / backlight)."""
    return cv2.convertScaleAbs(g, alpha=0.5, beta=100)


def lights_off(g):
    """Remove lamps and other bright spots (inpainting), then brighten: lights are off in daylight and the
    scene is lit evenly. Removes the strongest night landmarks."""
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    spots = ((tophat > 35) | (g > 200)).astype(np.uint8)          # small spots much brighter than around
    spots = cv2.dilate(spots, np.ones((9, 9), np.uint8))
    small = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    msk = cv2.resize(spots, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    small = cv2.inpaint(small, msk, 5, cv2.INPAINT_TELEA)
    filled = cv2.resize(small, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = np.where(spots > 0, filled, g).astype(np.uint8)
    return _gamma(out, 0.45)


def blur(g):
    """Horizontal motion blur over 15 px (higher speed / longer exposure)."""
    k = np.zeros((1, 15), np.float32)
    k[0, :] = 1 / 15
    return cv2.filter2D(g, -1, k)


def invert(g):
    """Photographic negative: bright and dark swap. A crude stand-in for daylight reversing the night shading
    (lit stands and lamps dark, unlit surfaces bright)."""
    return 255 - g


TRANSFORMS = {'none': lambda g: g, 'bright': bright, 'overexpose': overexpose, 'lowcontrast': lowcontrast,
              'lights_off': lights_off, 'blur': blur, 'invert': invert}
