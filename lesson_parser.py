from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup, Tag
import time
import copy

# Initialize the WebDriver (Chrome in this case)
def parse_content(url, filename):
    driver = webdriver.Chrome()
    # Navigate to the web page
    driver.get(url)

    # Wait for the dynamic content to load
    time.sleep(3)  # Adjust the sleep time according to your needs

    # Locate the div with the specified class or ID
    content_div = driver.find_element(By.ID, "content")  # or By.CLASS_NAME for class

    with open(filename + '.html', 'w') as f_html, open(filename, 'w') as f:
        f_html.write('<!DOCTYPE html>\n<head><meta charset="utf-8"></head>\n')
        f_html.write('<html lang="ru">')
        # Extract text from each paragraph within the div
        paragraphs = content_div.find_elements(By.XPATH, "./*")
        for paragraph in paragraphs:
            res = paragraph.text
            html_content = paragraph.get_attribute('outerHTML')
            if res == '&nbsp;' or (res == '' and html_content.find('<img') == -1):
                continue
            soup = BeautifulSoup(html_content, 'html.parser')
            # Create a new <p> tag
            new_p = soup.new_tag("p")

            # Find all <p> tags and images within the original HTML
            for element in soup.find_all(['p', 'img', 'a', 'strong', 'span', 'li', 'td'], recursive=True):
                # If it's a paragraph, extract all its contents
                if isinstance(element, Tag) and element.name == "p":
                    content_list = copy.copy(element.contents)
                    for content in content_list:
                        new_p.append(content)
                # If it's an image, append it directly
                else:  # element.name in ['a', 'img', 'li']:
                    if element not in new_p.contents:
                        new_p.append(element)
                # elif element.name in ['strong', 'span']:
                #     if element not in new_p.contents:
                #         new_p.append(element)
                # else:
                #     if element not in new_p.contents:
                #         if
                #         new_p.append(element.text)
            new_contents = soup.new_tag("p")
            contents_copy = copy.copy(new_p.contents)
            for content in contents_copy:
                if content.name in ['p', 'strong', 'span']:
                    new_contents.append(content.text)
                else:
                    new_contents.append(content)
            # Convert the new <p> tag back to string

            new_html_content = str(new_contents)
            f_html.write(new_html_content)
            f_html.write('\n')
            res = new_contents.text
            if paragraph.get_attribute('class') == 'lesson-subtitle':
                res = "Подтема: " + res
            f.write(res)
            f.write('\n')
        f_html.write('</html>')
        f_html.close(), f.close()
    driver.quit()
    # Close the browser
