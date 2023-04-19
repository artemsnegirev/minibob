from typing import List, Dict
from dataclasses import dataclass

from selenium import webdriver
from selenium.webdriver.common.by import By


@dataclass
class DataRecord:
    prompt: str
    answer: str


class BaseParser:
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
    
    def get_page_data(self, driver: webdriver.Chrome) -> List[DataRecord]:
        records = self._parse_page(driver)
        return [self._postprocess(r) for r in records]

    def _postprocess(self, record: DataRecord) -> DataRecord:
        record.answer = record.answer.strip().lower()
        record.prompt = record.prompt.strip()

        return record

    def _parse_page(self, driver: webdriver.Chrome) -> List[DataRecord]:
        pass

    def get_page_links(self, driver: webdriver.Chrome) -> List[str]:
        return [self.base_url]


class BygameBasic(BaseParser):
    def get_page_links(self, driver: webdriver.Chrome) -> List[str]:
        elements = driver.find_elements(By.CSS_SELECTOR, 'ul.uk-list li a')
        return [el.get_attribute('href') for el in elements]
    
    def _parse_page(self, driver: webdriver.Chrome) -> List[DataRecord]:    
        elements = driver.find_elements(By.CSS_SELECTOR, 'p strong')

        items = [
            el.find_element(By.XPATH, '..').text.split('Ответ:')
            for el in elements
        ]

        return [DataRecord(item[0], item[1]) for item in items]


class GuessAnswer(BaseParser):
    def _parse_page(self, driver: webdriver.Chrome) -> List[DataRecord]:
        prompts = [el.text for el in driver.find_elements(By.CSS_SELECTOR, 'h4')]
        
        answers_list = [
            [a.text for a in el.find_elements(By.TAG_NAME, 'li')]
            for el in driver.find_elements(By.CSS_SELECTOR, 'ol.uk-list')
        ]

        # flatten {prompt: str, answers: []} to [{prompt: str, answer: str}]

        result = [
            [DataRecord(prompt, answer) for answer in answers]
            for prompt, answers in zip(prompts, answers_list)
        ]

        result = sum(result, [])
        
        return result


class Zagadki(BaseParser):
    def _parse_page(self, driver: webdriver.Chrome) -> List[Dict]:    
        elements = driver.find_elements(By.CSS_SELECTOR, 'li strong')

        items = [
            el.find_element(By.XPATH, '..').text.split('Ответ:')
            for el in elements
        ]

        return [DataRecord(item[0], item[1]) for item in items]


class Crossword(BaseParser):
    def get_page_links(self, driver: webdriver.Chrome) -> List[str]:
        elements = driver.find_elements(By.CLASS_NAME, 'crossword-list__item-title')
        return [el.get_attribute('href') for el in elements]
    
    def _parse_page(self, driver: webdriver.Chrome) -> List[Dict]:
        items = driver.execute_script("return window.cw.data.grid.words")
        
        return [DataRecord(item['question'], item['word']) for item in items]


class Bashnya(BaseParser):
    def get_page_links(self, driver: webdriver.Chrome) -> List[str]:
        elements = driver.find_elements(By.CSS_SELECTOR, 'a.uk-button')
        return [el.get_attribute('href') for el in elements]
    
    def _parse_page(self, driver: webdriver.Chrome) -> List[Dict]:
        prompt = driver.find_element(By.TAG_NAME, 'h4').text

        answers = [
            el.text
            for el in driver.find_elements(By.CSS_SELECTOR, 'ol.uk-list-disc li')
        ]

        return [DataRecord(prompt, answer) for answer in answers]


class TopSeven(BaseParser):
    def get_page_links(self, driver: webdriver.Chrome) -> List[str]:
        elements = driver.find_elements(By.CSS_SELECTOR, 'a.uk-button')
        return [el.get_attribute('href') for el in elements]
    
    def _parse_page(self, driver: webdriver.Chrome) -> List[DataRecord]:
        prompt = driver.find_element(By.TAG_NAME, 'h2').text

        answers = [
            el.text
            for el in driver.find_elements(By.CSS_SELECTOR, 'ul.uk-list-disc li')
        ]

        return [DataRecord(prompt, answer) for answer in answers]
    
PARSERS: List[BaseParser] = [
    Zagadki("350_zagadok", "https://bygame.ru/otvety/350-zagadok"),
    Bashnya("bashnya_slov", "https://bygame.ru/otvety/na-igru-bashnya-slov"),
    Crossword("crosswords", "https://baza-otvetov.ru/crosswords"),
    GuessAnswer("guess_answer", "https://bygame.ru/otvety/guess-their-answer-vse-voprosy"),
    BygameBasic("ostrova", "https://bygame.ru/otvety/na-igru-ostrova-krossvordov"),
    TopSeven("top_seven", "https://bygame.ru/otvety/top-7"),
    BygameBasic("ugadaj_slova", "https://bygame.ru/otvety/ugadaj-slova-vse-urovni"),
    BygameBasic("umnyasha", "https://bygame.ru/otvety/umnyasha-vse-urovni")
]