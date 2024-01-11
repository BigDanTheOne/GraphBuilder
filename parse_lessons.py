from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os
from lesson_parser import parse_content


# Initialize the WebDriver (Chrome in this case)
driver = webdriver.Chrome()

# Navigate to the web page
driver.get('https://interneturok.ru/book/physics/11-klass/fizika-11-klass-myakishev-g-ya')

# Wait for the dynamic content to load
time.sleep(5)  # Adjust the sleep time according to your needs

# Locate the div with the specified class or ID
content_div = driver.find_element(By.CLASS_NAME, "subject-theme-textbook")
section_numbers = list(range(20))
for section_number in section_numbers:
    section = None
    try:
        section = content_div.find_element(By.ID, 'section_' + str(section_number))
    except:
        break
    # if section.get_attribute('class') != \
    #         'row justify-content-between align-items-start subject-theme__item has-sublevels ember-view':
    #     continue
    title = section.find_element(By.CLASS_NAME, 'subject-theme__title').text
    print(title)
    os.mkdir(title)
    for theme in section.find_element(By.CLASS_NAME, 'subject-theme__sublist').find_elements(By.CLASS_NAME, 'has-sublevels'):
        theme_title = theme.find_element(By.CLASS_NAME, 'subject-theme__subtitle').text
        print('-' + theme_title)
        os.mkdir(os.path.join(title, theme_title))
        for topic in theme.find_elements(By.TAG_NAME, 'li'):
            if topic.get_attribute('class') == 'row justify-content-between has-paragraph ember-view':
                try:
                    lesson_a = topic.find_element(By.TAG_NAME, 'a')
                except:
                    continue
                lesson = lesson_a.text
                url = lesson_a.get_attribute("href")
                print('--' + lesson + ' : ' + url)
                parse_content(url, f'{title}/{theme_title}/{lesson}')
            else:
                topic_name = topic.find_element(By.TAG_NAME, 'a').text
                if topic_name == '':
                    continue
                print('--' + topic_name)
                os.mkdir(f'{title}/{theme_title}/{topic_name}')
                lessons = topic.find_elements(By.TAG_NAME, 'li')
                for lesson in lessons:
                    lesson_a = lesson.find_element(By.TAG_NAME, 'a')
                    lesson = lesson_a.get_attribute("text")
                    url = lesson_a.get_attribute("href")
                    print('---' + lesson + ' : ' + url)
                    parse_content(url, f'{title}/{theme_title}/{topic_name}/{lesson}')

