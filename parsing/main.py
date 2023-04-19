import os
import json
import tqdm
import traceback

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from parsers import PARSERS


DRIVER_PATH = "chromedriver/chromedriver"

DATASET_PATH = "data/dataset.json"
PROGRESS_PATH = "data/progress.json"

WAIT_SECS = 2


def load_progress():
    if not os.path.exists(PROGRESS_PATH):
        save_progress(0, 0, 0)
    
    with open(PROGRESS_PATH) as fr:
        progress = json.load(fr)
    
    return progress["parser"], progress["page"], progress["item"]

def save_progress(parser_id, page_id, item_id):
    with open(PROGRESS_PATH, "w") as fw:
        payload = {"parser": parser_id, "page": page_id, "item": item_id}
        json.dump(payload, fw)

def parsing_loop(driver: webdriver.Chrome, last_parser_id, last_page_id, last_item_id):
    for parser_id in range(last_parser_id, len(PARSERS)):
        parser = PARSERS[parser_id]

        driver.get(parser.base_url)
        driver.implicitly_wait(WAIT_SECS)

        pages_link = parser.get_page_links(driver)
        
        # start from last visited page if current parser is last visited
        offset_page_id = last_page_id if last_parser_id == parser_id else 0

        for page_id in tqdm.tqdm(range(offset_page_id, len(pages_link)), desc=parser.name):
            link = pages_link[page_id]
            
            driver.get(link)
            driver.implicitly_wait(WAIT_SECS)

            data_items = parser.get_page_data(driver)

            # start from next after last visited item and 
            if last_parser_id == parser_id and last_page_id == page_id:
                offset_item_id = last_item_id + 1
            else:
                offset_item_id = 0

            for item_id in range(offset_item_id, len(data_items)):
                data_item = data_items[item_id]
                
                payload = {
                    "subset": parser.name,
                    "answer": data_item.answer,
                    "prompt": data_item.prompt
                }
                
                yield (payload, (parser_id, page_id, item_id))


def main(driver: webdriver.Chrome):
    progress = load_progress()

    try:
        with open(DATASET_PATH, "a") as fw:
            for payload, last_progress in parsing_loop(driver, *progress):
                print(json.dumps(payload, ensure_ascii=False), file=fw)
    except Exception:
        print(traceback.format_exc())
    finally:
        save_progress(*last_progress)
        print(f"Saved progress to {PROGRESS_PATH}")

    print(f"Saved dataset to {DATASET_PATH}")

    
if __name__ == '__main__':
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--host-resolver-rules=MAP www.google-analytics.com 127.0.0.1")
    chrome_options.add_argument('user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36')

    driver = webdriver.Chrome(executable_path=DRIVER_PATH, options=chrome_options)
    driver.implicitly_wait(2)

    main(driver)