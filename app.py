"""
This module provides a Flask application for predicting UFC fight outcomes 
based on fighter statistics scraped from the UFC website. It includes utilities 
for extracting and processing fighter and event data, managing images, and 
integrating machine learning models for predictions.

Dependencies:
    - os
    - onnxruntime
    - datetime
    - urllib.parse
    - requests
    - pycountry
    - numpy
    - bs4 (BeautifulSoup)
    - unidecode
    - flask
    - flask_cors
    - util (custom utilities for data conversion)

Flask Routes:
    - '/predict/<event_path>' : Predicts outcomes of matches in an event.
    - '/' : Fetches and displays upcoming events and their details.
"""

import os
import gc

from datetime import datetime
from urllib.parse import urlparse

import requests
import pycountry
import numpy as np
import onnxruntime as ort
from bs4 import BeautifulSoup
from unidecode import unidecode

from flask_cors import CORS
from flask import Flask, jsonify
from util import convert_height, convert_weight, convert_date_of_birth, convert_reach

app = Flask(__name__)
CORS(app)

stance_lable = ['Open Stance', 'Orthodox', 'Southpaw', 'Switch']

model = ort.InferenceSession('model/RandomForest-arp.onnx')
input_name = model.get_inputs()[0].name


def download_image(path, image_name, image_url, over_write):
    """
    Downloads an image from a given URL and saves it to the specified path.

    Args:
        path (str): Directory path where the image will be saved.
        image_name (str): Name of the image file.
        image_url (str): URL of the image to download.
        over_write (bool): Whether to overwrite the file if it already exists.

    Returns:
        str: The complete path of the saved image.
    """
    if over_write:
        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(path+image_name) and over_write:
            img_response = requests.get(image_url, timeout=10)
            img_response.raise_for_status()

            with open(path+image_name, 'wb') as file:
                file.write(img_response.content)

    return path+image_name


def extract_fighter_image(img_url_):
    """
    Constructs a complete image URL for a fighter.

    Args:
        img_url_ (ParseResult): A parsed URL object of the image.

    Returns:
        str: The complete URL of the fighter image.
    """
    if img_url_.scheme == '' and img_url_.netloc == '':
        img_url = 'https://www.ufc.com' + img_url_.path
    else:
        img_url = img_url_.scheme+'://' + img_url_.netloc + img_url_.path

    return img_url


def get_fighter_url(fighter_name):
    """
    Searches for a fighter's UFC stats URL by name.

    Args:
        fighter_name (str): The full name of the fighter.

    Returns:
        str: The URL to the fighter's stats page, or None if not found.
    """
    fighter_url = None
    url='http://www.ufcstats.com/statistics/fighters/search?query='

    response = requests.get(url+fighter_name.split(' ')[-1], timeout=10) #Search By Famliy name
    soup = BeautifulSoup(response.content, 'lxml')

    fighters_ = soup.find_all('tr', class_='b-statistics__table-row')

    for fighter_ in fighters_:
        if unidecode(fighter_name.split(' ')[0]) in fighter_.text:
            try:
                fighter_url = fighter_.findChild('a')['href']
            except:
                continue

    if fighter_url is None:
        response = requests.get(url+fighter_name.split(' ')[0], timeout=10) #Search By Given name
        soup = BeautifulSoup(response.content, 'lxml')

        fighters_ = soup.find_all('tr', class_='b-statistics__table-row')

        for fighter_ in fighters_:
            if unidecode(fighter_name.split(' ')[0]) in fighter_.text:
                del response, soup
                gc.collect()
                return fighter_.findChild('a')['href']
    else:
        del response, soup
        gc.collect()
        return fighter_url


def get_fighter_stat(fighter_stat_):
    """
    Extracts and converts fighter statistics into a NumPy array.

    Args:
        fighter_stat_ (BeautifulSoup): HTML content of the fighter's stats section.

    Returns:
        np.ndarray: A NumPy array of processed fighter statistics.
    """
    stats_ = fighter_stat_.find_all('li', class_='b-list__box-list-item b-list__box-list-item_type_block')
    stat = dict()
    for stat_ in stats_:
        stat_item = stat_.text.replace(' ', '').replace('\n', '').split(':')
        if len(stat_item) == 2:
            stat[stat_item[0]] = stat_item[1]

    stat_np = np.zeros(13)
    for key, value in stat.items():
        if key == 'Height':
            stat_np[1] = convert_height(value)
        elif key == 'Weight':
            stat_np[2] = convert_weight(value)
        elif key == 'Reach':
            stat_np[3] = convert_reach(value)
        elif key == 'STANCE':
            try:
                stat_np[4] = stance_lable.index(value)
            except ValueError:
                stat_np[4] = -1
        elif key == 'DOB':
            stat_np[0] = convert_date_of_birth(value)
        elif key == 'SLpM':
            stat_np[5] = float(value)
        elif key == 'Str.Acc.':
            stat_np[7] = float(value.replace('%', '')) * 0.01
        elif key == 'SApM':
            stat_np[6] = float(value)
        elif key == 'Str.Def':
            stat_np[9] = float(value.replace('%', '')) * 0.01
        elif key == 'TDAvg.':
            stat_np[12] = float(value)
        elif key == 'TDAcc.':
            stat_np[8] = float(value.replace('%', '')) * 0.01
        elif key == 'TDDef.':
            stat_np[10] = float(value.replace('%', '')) * 0.01
        elif key == 'Sub.Avg.':
            stat_np[11] = float(value)
        else:
            raise ValueError

    return stat_np


def get_fighter_record(fighter_stat_):
    """
    Extracts the win-loss record of a fighter.

    Args:
        fighter_stat_ (BeautifulSoup): HTML content of the fighter's stats section.

    Returns:
        np.ndarray: An array with the number of wins and losses.
    """
    loss_record_ = fighter_stat_.find_all('a', class_='b-flag b-flag_style_bordered')
    win_record_ = fighter_stat_.find_all('a', class_='b-flag b-flag_style_green')
    return np.array([len(win_record_), len(loss_record_)])


def prepare_data(fighter_red, fighter_blue):
    """
    Prepares input data for the prediction model by combining statistics
    and records for two fighters.

    Args:
        fighter_red (str): The name of the red corner fighter.
        fighter_blue (str): The name of the blue corner fighter.

    Returns:
        np.ndarray: A NumPy array containing the combined data of both fighters.
    """
    response = requests.get(get_fighter_url(fighter_red), timeout=10)
    fighter_red_stat_ = BeautifulSoup(response.content, 'lxml')
    fighter_red_stat = get_fighter_stat(fighter_red_stat_)
    fighter_red_record = get_fighter_record(fighter_red_stat_)

    response = requests.get(get_fighter_url(fighter_blue), timeout=10)
    fighter_blue_stat_ = BeautifulSoup(response.content, 'lxml')
    fighter_blue_stat = get_fighter_stat(fighter_blue_stat_)
    fighter_blue_record = get_fighter_record(fighter_blue_stat_)

    del response
    gc.collect()
    return np.concatenate((fighter_red_record, fighter_red_stat, fighter_blue_record, fighter_blue_stat))



def extract_fighter_names(tag):
    """
    Extracts the full name of a fighter from an HTML tag.

    Args:
        tag (Tag): An HTML tag containing the fighter's name.

    Returns:
        str: The extracted full name of the fighter.
    """
    given_name = tag.find('span', class_='details-content__corner-given-name')
    family_name = tag.find('span', class_='details-content__corner-family-name')
    if given_name and family_name:
        fighter_name = f"{given_name.text.strip()} {family_name.text.strip()}"
    else:
        fighter_name = tag.text.strip()
    fighter_name = fighter_name.replace('\n', ' ')

    return fighter_name


def get_match(event_url):
    """
    Retrieves and processes all matches from a given event.

    Args:
        event_url (str): The URL of the UFC event.

    Returns:
        tuple: A list of matches and a NumPy array of input data for predictions.
    """
    response = requests.get(event_url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'lxml')

    matchlist = list()
    input_data = list()
    matchlist_ = soup.find_all('li', class_='l-listing__item')
    for item in matchlist_:
        image_ = item.find_all('img')
        country_ = item.find_all('div', class_='c-listing-fight__country-text')

        img_url_ = urlparse(image_[0].get('src'))
        red_name_ = item.find('div', class_='details-content__name--red')
        red_name = extract_fighter_names(red_name_)
        red = {'Name': red_name,
            'height': image_[0].get('height'),
            'width': image_[0].get('width'),
            'image': extract_fighter_image(img_url_),
            'country': country_[0].text,
            'country_code': pycountry.countries.search_fuzzy(country_[0].text)[0].alpha_2
        }

        img_url_ = urlparse(image_[1].get('src'))
        blue_name_ = item.find('div', class_='details-content__name--blue')
        blue_name = extract_fighter_names(blue_name_)
        blue = {
            'Name': blue_name,
            'height': image_[1].get('height'),
            'width': image_[1].get('width'),
            'image': extract_fighter_image(img_url_),
            'country': country_[1].text,
            'country_code': pycountry.countries.search_fuzzy(country_[1].text)[0].alpha_2
        }

        red['odd'] = item.find_all('span', class_='c-listing-fight__odds-amount')[0].text
        blue['odd'] = item.find_all('span', class_='c-listing-fight__odds-amount')[1].text

        input_data.append(prepare_data(red_name, blue_name))

        match = {'red': red, 'blue': blue}
        matchlist.append(match)

    del response, soup
    gc.collect()
    return matchlist, np.array(input_data)


async def _predict(input_data):
    """
    Predicts the outcome probabilities for matches using the loaded model.

    Args:
        input_data (np.ndarray): Preprocessed input data for the model.

    Returns:
        np.ndarray: An array of prediction probabilities for each match.
    """
    output_data = model.run(None, {input_name: input_data.astype(np.float32)})

    return output_data[1]


@app.route('/predict/<event_path>', methods=['GET'])
async def predict(event_path):
    """
    Flask route to predict outcomes for a given UFC event.

    Args:
        event_path (str): The path segment of the UFC event URL.

    Returns:
        Response: A JSON response containing event details and predictions.
    """
    event = get_event('https://www.ufc.com/event/'+event_path)
    event['match'], input_data = get_match('https://www.ufc.com/event/'+event_path)

    output_data = await _predict(input_data)

    for i, match in enumerate(event['match']):
        match['red']['outcome'] = str(round(output_data[i]['Red']*100, 1))+'%'
        match['blue']['outcome'] = str(round(output_data[i]['Blue']*100, 1))+'%'

    del output_data
    gc.collect()
    return jsonify(event)


def get_event_image(soup):
    """
    Extracts and saves event images from the given HTML content.

    Args:
        soup (BeautifulSoup): Parsed HTML content of the event page.

    Returns:
        list: A list of dictionaries containing image details.
    """
    picture_ = soup.find('picture')

    if picture_ is None:
        return None
    sources_ = picture_.find_all('source')

    image_list = list()
    for source_ in sources_:
        resolution = source_.get('width')+'x'+source_.get('height')
        image = {
            'width': source_.get('width'),
            'height': source_.get('height'),
            'resolution': resolution,
        }

        for image_url in source_['srcset'].split(','):
            img_url_ = urlparse(image_url)
            img_url = img_url_.scheme+'://'+ img_url_.netloc + img_url_.path

            if img_url_.query.endswith('1x'):
                image['x1'] = img_url
            elif img_url_.query.endswith('2x'):
                image['x2'] = img_url

        image_list.append(image)

    return image_list


def get_event(event_url):
    """
    Retrieves details of a specific UFC event.

    Args:
        event_url (str): The URL of the UFC event.

    Returns:
        dict: A dictionary containing event details.
    """
    event_path = event_url.split('event/')[-1]

    response = requests.get(event_url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'lxml')

    event_name = soup.find('div', class_='field field--name-node-title field--type-ds field--label-hidden field__item').text.strip()

    main_event_red = soup.find('span', class_='e-divider__top').text.strip()
    main_event_blue = soup.find('span', class_='e-divider__bottom').text.strip()

    event_time = soup.find('div', class_='c-hero__headline-suffix tz-change-inner').text.strip()
    event_field = soup.find('div', class_='field field--name-venue field--type-entity-reference field--label-hidden field__item').text.strip().replace('\n', ' ')

    event = {
        'event_path': event_path,
        'event_name': event_name,
        'main_event_red': main_event_red,
        'main_event_blue': main_event_blue,
        'event_time': event_time,
        'event_field': event_field,
        'event_image': get_event_image(soup)
    }

    del response, soup
    gc.collect()
    return event


def get_event_list():
    """
    Retrieves a list of upcoming UFC events.

    Returns:
        list: A list of URLs for upcoming events.
    """
    url = 'https://www.ufc.com/events'
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'lxml')
    cards = soup.find_all('div', class_='views-infinite-scroll-content-wrapper clearfix')
    upcoming_events = cards[0].find_all('div', class_='l-listing__item views-row')

    event_list = list()
    for card in upcoming_events:
        event_list.append('https://www.ufc.com'+card.find('h3').find('a').get('href'))

    del response, soup
    gc.collect()
    return event_list


def get_index():
    """
    Fetches and saves promotional images from the UFC homepage.

    Returns:
        list: A list of image details.
    """
    url = "https://www.ufc.com"
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.content, 'lxml')

    index_image = list()
    sources_ = soup.find('picture').find_all('source')
    for source_ in sources_:
        image_urls = source_['srcset'].split(',')
        resolution = source_.get('width')+'x'+source_.get('height')
        image = {
            'width': source_.get('width'),
            'height': source_.get('height'),
            'resolution': resolution,
        }
        for image_url in image_urls:
            img_url_ = urlparse(image_url)
            img_url = img_url_.scheme+'://'+ img_url_.netloc + img_url_.path
            img_path = 'static/images/index/'

            over_write = False if datetime.now().weekday() == 0 else False
            if img_url_.query.endswith('1x'):
                image['x1'] = download_image(img_path, resolution+'_1x.jpg', img_url, over_write)
            elif img_url_.query.endswith('2x'):
                image['x2'] = download_image(img_path, resolution+'_2x.jpg', img_url, over_write)
            index_image.append(image_urls)

    del response, soup
    gc.collect()
    return index_image


@app.route('/', methods=['GET'])
def onload():
    """
    Flask route to load the index page and list of upcoming events.

    Returns:
        Response: A JSON response containing event details and images.
    """

    homepage = dict()
    homepage['wallpaper'] = get_index()

    event_list = list()
    event_url_list = get_event_list()
    for event_url in event_url_list:
        event_list.append(get_event(event_url))
    homepage['event_list'] = event_list

    return jsonify(homepage)


if __name__ == '__main__':
    app.run()
