"""
This module provides utility functions for converting fighter statistics into 
standardized formats. It includes methods for converting height, weight, reach, 
and date of birth into their respective metric equivalents or usable formats.

Functions:
    - convert_height: Converts height from feet and inches to centimeters.
    - convert_weight: Converts weight from pounds to kilograms.
    - convert_reach: Converts reach from inches to centimeters.
    - convert_date_of_birth: Calculates age based on the date of birth.
"""

import re
from datetime import datetime


def convert_height(height_inches):
    """
    Converts height from feet and inches to centimeters.

    Args:
        height_inches (str): Height in the format 'Height:*feet*'*inches*"'.

    Returns:
        float or None: Height in centimeters, or None if input is invalid.
    """
    if height_inches != '--':
        # Assuming height_inches is in the format 'Height:*number*\' *number*"'
        height_match = re.match(r'(\d+)\'(\d+)"', height_inches)

        if height_match is not None:
            feet, inches = map(int, height_match.groups())
            # Convert height to centimeters (1 foot = 30.48 cm, 1 inch = 2.54 cm)
            height_cm = (feet * 30.48) + (inches * 2.54)
        else:
            height_cm = None
    else:
        height_cm = None

    return height_cm


def convert_weight(weight_pounds):
    """
    Converts weight from pounds to kilograms.

    Args:
        weight_pounds (str): Weight in the format 'Weight:*number* lbs.'.

    Returns:
        float or None: Weight in kilograms, or None if input is invalid.
    """
    if weight_pounds != '--':
        # Assuming weight_pounds is in the format 'Weight:*number* lbs.'
        weight_match = re.match(r'(\d+)lbs\.', weight_pounds)
        if weight_match:
            weight_in_lbs = int(weight_match.group(1))
            # Convert weight to kilograms (1 lb = 0.453592 kg)
            weight_kg = weight_in_lbs * 0.453592
        else:
            weight_kg = None
    else:
        weight_kg = None

    return weight_kg


def convert_reach(reach_inches):
    """
    Converts reach from inches to centimeters.

    Args:
        reach_inches (str): Reach in the format '*number*"' (inches).

    Returns:
        float or None: Reach in centimeters, or None if input is invalid.
    """
    if reach_inches != '--':
        reach_inches = reach_inches.replace('"', '').strip()
        reach_cm = int(reach_inches) * 2.54
    else:
        reach_cm = None

    return reach_cm


def convert_date_of_birth(dob):
    """
    Calculates age from the date of birth.

    Args:
        dob (str): Date of birth in the format '%b%d,%Y', e.g., 'Jan01,1990'.

    Returns:
        int or None: Age in years, or None if input is invalid.
    """
    if dob != '--':
        # Convert the date of birth string to a datetime object
        dob = datetime.strptime(dob, '%b%d,%Y')
        # Get the current date
        current_date = datetime.now()
        # Calculate the age
        age = current_date.year - dob.year - \
            ((current_date.month, current_date.day) < (dob.month, dob.day))
    else:
        age = None

    return age
